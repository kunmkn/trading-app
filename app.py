import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    gspread = None
    Credentials = None


DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "trades.csv"
SETTINGS_FILE = DATA_DIR / "settings.json"

CHECKLIST_FIELDS = ["趋势结构确认", "量仓配合", "基本面支持", "事件风险可控", "盈亏比合格"]
MARKETS = ["国内期货", "海外期货", "A股", "美股", "期权"]
DIRECTIONS = ["多", "空"]
ACTIONS = ["开仓", "加仓", "减仓", "平仓"]
OPEN_ACTIONS = {"开仓", "加仓"}
CLOSE_ACTIONS = {"减仓", "平仓"}
SETUPS = ["趋势突破", "回调入场", "区间反转", "基本面事件", "价差/套利"]

TRADE_COLUMNS = [
    "date", "market", "symbol", "direction", "action", "setup", "entry", "stop", "target1", "target2",
    "exit_price", "quantity", "contract_multiplier", "max_risk_rmb", "thesis",
] + CHECKLIST_FIELDS


def to_number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else 0.0
    try:
        parsed = float(str(value).replace(",", "").strip())
        return parsed if math.isfinite(parsed) else 0.0
    except (TypeError, ValueError):
        return 0.0


def direction_sign(direction: str) -> int:
    return -1 if direction == "空" else 1


@dataclass
class Metrics:
    risk_per_unit: float = 0.0
    total_risk: float = 0.0
    max_qty_by_risk: int = 0
    pnl: float = 0.0
    r_multiple: float = 0.0
    reward_risk_1: float = 0.0
    reward_risk_2: float = 0.0
    checklist_score: int = 0


def calc_trade(trade: Dict[str, Any]) -> Metrics:
    """Single-row preview calculation. The portfolio uses the LIFO engine below."""
    trade = trade or {}
    entry = to_number(trade.get("entry"))
    stop = to_number(trade.get("stop"))
    exit_price = to_number(trade.get("exit_price"))
    target1 = to_number(trade.get("target1"))
    target2 = to_number(trade.get("target2"))
    qty = max(0.0, to_number(trade.get("quantity")))
    multiplier_raw = to_number(trade.get("contract_multiplier"))
    multiplier = multiplier_raw if multiplier_raw > 0 else 1.0
    max_risk = max(0.0, to_number(trade.get("max_risk_rmb")))
    sign = direction_sign(trade.get("direction", "多"))

    risk_per_unit = abs(entry - stop) * multiplier if entry != 0 and stop != 0 else 0.0
    total_risk = risk_per_unit * qty
    max_qty_by_risk = int(max_risk // risk_per_unit) if risk_per_unit > 0 else 0
    pnl = (exit_price - entry) * sign * multiplier * qty if entry != 0 and exit_price != 0 and qty > 0 else 0.0
    r_multiple = pnl / total_risk if total_risk > 0 else 0.0
    reward_risk_1 = abs(target1 - entry) * multiplier / risk_per_unit if risk_per_unit > 0 and target1 != 0 else 0.0
    reward_risk_2 = abs(target2 - entry) * multiplier / risk_per_unit if risk_per_unit > 0 and target2 != 0 else 0.0
    checklist_score = sum(1 for field in CHECKLIST_FIELDS if bool(trade.get(field, False)))

    return Metrics(risk_per_unit, total_risk, max_qty_by_risk, pnl, r_multiple, reward_risk_1, reward_risk_2, checklist_score)


def chronological_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(reversed(trades))


def make_lot(trade: Dict[str, Any], lot_id: int) -> Dict[str, Any]:
    entry = to_number(trade.get("entry"))
    stop = to_number(trade.get("stop"))
    qty = max(0.0, to_number(trade.get("quantity")))
    multiplier_raw = to_number(trade.get("contract_multiplier"))
    multiplier = multiplier_raw if multiplier_raw > 0 else 1.0
    risk_per_unit = abs(entry - stop) * multiplier if entry != 0 and stop != 0 else 0.0
    return {
        "lot_id": lot_id,
        "date": trade.get("date", ""),
        "market": trade.get("market", ""),
        "symbol": str(trade.get("symbol", "")).strip(),
        "direction": trade.get("direction", "多"),
        "entry": entry,
        "stop": stop,
        "qty": qty,
        "remaining_qty": qty,
        "contract_multiplier": multiplier,
        "risk_per_unit": risk_per_unit,
        "initial_risk": risk_per_unit * qty,
        "thesis": trade.get("thesis", ""),
    }


def build_position_engine(trades: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], List[Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    stacks: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    realized_events: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    lot_id = 0

    for trade_no, trade in enumerate(chronological_trades(trades), start=1):
        action = trade.get("action", "开仓") or "开仓"
        symbol = str(trade.get("symbol", "")).strip()
        direction = trade.get("direction", "多")
        key = (symbol, direction)
        qty = max(0.0, to_number(trade.get("quantity")))
        if not symbol or qty <= 0:
            continue

        if action in OPEN_ACTIONS:
            lot_id += 1
            stacks.setdefault(key, []).append(make_lot(trade, lot_id))
            continue

        if action in CLOSE_ACTIONS:
            close_price = to_number(trade.get("exit_price"))
            if close_price == 0:
                warnings.append({"trade_no": trade_no, "symbol": symbol, "reason": "减仓/平仓缺少退出价，无法计算已实现PnL。"})
                continue

            remaining_to_close = qty
            stack = stacks.setdefault(key, [])
            while remaining_to_close > 0 and stack:
                lot = stack[-1]
                matched_qty = min(remaining_to_close, lot["remaining_qty"])
                sign = direction_sign(direction)
                pnl = (close_price - lot["entry"]) * sign * lot["contract_multiplier"] * matched_qty
                matched_initial_risk = lot["risk_per_unit"] * matched_qty
                realized_events.append(
                    {
                        "trade_no": trade_no,
                        "close_date": trade.get("date", ""),
                        "symbol": symbol,
                        "direction": direction,
                        "action": action,
                        "lot_id": lot["lot_id"],
                        "open_date": lot["date"],
                        "entry": lot["entry"],
                        "close_price": close_price,
                        "qty": matched_qty,
                        "contract_multiplier": lot["contract_multiplier"],
                        "initial_risk": matched_initial_risk,
                        "realized_pnl": pnl,
                        "realized_r": pnl / matched_initial_risk if matched_initial_risk > 0 else 0.0,
                    }
                )
                lot["remaining_qty"] -= matched_qty
                remaining_to_close -= matched_qty
                if lot["remaining_qty"] <= 1e-12:
                    stack.pop()

            if remaining_to_close > 1e-12:
                warnings.append({"trade_no": trade_no, "symbol": symbol, "reason": f"平仓数量超过当前持仓，未匹配数量：{remaining_to_close:g}"})

    return stacks, realized_events, warnings


def latest_prices_by_symbol(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    latest: Dict[str, float] = {}
    for trade in trades:
        symbol = str(trade.get("symbol", "")).strip()
        if not symbol or symbol in latest:
            continue
        price = to_number(trade.get("exit_price")) or to_number(trade.get("entry"))
        if price != 0:
            latest[symbol] = price
    return latest


def build_open_positions(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    stacks, _, _ = build_position_engine(trades)
    latest_prices = latest_prices_by_symbol(trades)
    rows: List[Dict[str, Any]] = []

    for (symbol, direction), lots in stacks.items():
        open_lots = [lot for lot in lots if lot["remaining_qty"] > 1e-12]
        if not open_lots:
            continue
        total_qty = sum(lot["remaining_qty"] for lot in open_lots)
        weighted_entry = sum(lot["entry"] * lot["remaining_qty"] for lot in open_lots) / total_qty if total_qty > 0 else 0.0
        initial_risk_left = sum(lot["risk_per_unit"] * lot["remaining_qty"] for lot in open_lots)
        mark_price = latest_prices.get(symbol, 0.0)
        sign = direction_sign(direction)
        unrealized_pnl = 0.0
        nominal_exposure = 0.0
        current_risk_to_stop = 0.0
        for lot in open_lots:
            price = mark_price if mark_price != 0 else lot["entry"]
            unrealized_pnl += (price - lot["entry"]) * sign * lot["contract_multiplier"] * lot["remaining_qty"]
            nominal_exposure += abs(price * lot["contract_multiplier"] * lot["remaining_qty"])
            risk_at_stop = (lot["entry"] - lot["stop"]) * sign * lot["contract_multiplier"] * lot["remaining_qty"]
            current_risk_to_stop += max(0.0, risk_at_stop)

        rows.append(
            {
                "品种": symbol,
                "方向": direction,
                "未平仓数量": round(total_qty, 6),
                "子仓位数": len(open_lots),
                "加权开仓价": round(weighted_entry, 4),
                "最新价/标记价": round(mark_price, 4) if mark_price else "",
                "剩余初始风险": round(initial_risk_left, 2),
                "当前止损风险": round(current_risk_to_stop, 2),
                "名义敞口": round(nominal_exposure, 2),
                "未实现PnL": round(unrealized_pnl, 2),
                "未实现R": round(unrealized_pnl / initial_risk_left, 2) if initial_risk_left > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_lot_detail(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    stacks, _, _ = build_position_engine(trades)
    rows: List[Dict[str, Any]] = []
    for (symbol, direction), lots in stacks.items():
        for lot in lots:
            if lot["remaining_qty"] <= 1e-12:
                continue
            rows.append(
                {
                    "Lot ID": lot["lot_id"],
                    "日期": lot["date"],
                    "品种": symbol,
                    "方向": direction,
                    "开仓价": lot["entry"],
                    "止损价": lot["stop"],
                    "剩余数量": round(lot["remaining_qty"], 6),
                    "合约乘数": lot["contract_multiplier"],
                    "剩余初始风险": round(lot["risk_per_unit"] * lot["remaining_qty"], 2),
                }
            )
    return pd.DataFrame(rows)


def build_realized_df(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    _, realized_events, _ = build_position_engine(trades)
    return pd.DataFrame(realized_events)


def build_pnl_curve(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    realized_df = build_realized_df(trades)
    if realized_df.empty:
        return pd.DataFrame(columns=["close_no", "date", "symbol", "realized_pnl", "cumulative_pnl", "peak", "drawdown"])
    rows: List[Dict[str, Any]] = []
    cumulative = 0.0
    peak = 0.0
    for idx, row in realized_df.iterrows():
        pnl = float(row["realized_pnl"])
        cumulative += pnl
        peak = max(peak, cumulative)
        rows.append(
            {
                "close_no": idx + 1,
                "date": row["close_date"],
                "symbol": row["symbol"],
                "realized_pnl": pnl,
                "cumulative_pnl": cumulative,
                "peak": peak,
                "drawdown": cumulative - peak,
            }
        )
    return pd.DataFrame(rows)


def build_account_curve(trades: List[Dict[str, Any]], initial_equity: float) -> pd.DataFrame:
    chronological = chronological_trades(trades)
    rows: List[Dict[str, Any]] = []
    for idx in range(1, len(chronological) + 1):
        partial = list(reversed(chronological[:idx]))
        open_df = build_open_positions(partial)
        realized_df = build_realized_df(partial)
        realized_pnl = float(realized_df["realized_pnl"].sum()) if not realized_df.empty else 0.0
        unrealized_pnl = float(open_df["未实现PnL"].sum()) if not open_df.empty else 0.0
        current_risk = float(open_df["当前止损风险"].sum()) if not open_df.empty else 0.0
        nominal_exposure = float(open_df["名义敞口"].sum()) if not open_df.empty else 0.0
        realized_equity = initial_equity + realized_pnl
        total_equity = realized_equity + unrealized_pnl
        trade = chronological[idx - 1]
        rows.append(
            {
                "trade_no": idx,
                "date": trade.get("date", ""),
                "symbol": trade.get("symbol", ""),
                "action": trade.get("action", ""),
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "realized_equity": realized_equity,
                "total_equity": total_equity,
                "current_risk": current_risk,
                "risk_usage": current_risk / total_equity if total_equity > 0 else 0.0,
                "nominal_exposure": nominal_exposure,
                "leverage": nominal_exposure / total_equity if total_equity > 0 else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["equity_peak"] = df["total_equity"].cummax()
    df["equity_drawdown"] = df["total_equity"] - df["equity_peak"]
    df["equity_drawdown_pct"] = df.apply(lambda r: r["equity_drawdown"] / r["equity_peak"] if r["equity_peak"] > 0 else 0.0, axis=1)
    return df


def calc_portfolio(trades: List[Dict[str, Any]], initial_equity: float) -> Dict[str, float]:
    open_df = build_open_positions(trades)
    realized_df = build_realized_df(trades)
    pnl_curve = build_pnl_curve(trades)
    account_curve = build_account_curve(trades, initial_equity)

    total_realized_pnl = float(realized_df["realized_pnl"].sum()) if not realized_df.empty else 0.0
    total_unrealized_pnl = float(open_df["未实现PnL"].sum()) if not open_df.empty else 0.0
    current_equity = initial_equity + total_realized_pnl + total_unrealized_pnl
    open_current_risk = float(open_df["当前止损风险"].sum()) if not open_df.empty else 0.0
    nominal_exposure = float(open_df["名义敞口"].sum()) if not open_df.empty else 0.0

    return {
        "open_qty": float(open_df["未平仓数量"].sum()) if not open_df.empty else 0.0,
        "open_initial_risk": float(open_df["剩余初始风险"].sum()) if not open_df.empty else 0.0,
        "open_current_risk": open_current_risk,
        "total_realized_pnl": total_realized_pnl,
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_pnl": total_realized_pnl + total_unrealized_pnl,
        "current_equity": current_equity,
        "risk_usage": open_current_risk / current_equity if current_equity > 0 else 0.0,
        "nominal_exposure": nominal_exposure,
        "leverage": nominal_exposure / current_equity if current_equity > 0 else 0.0,
        "avg_realized_r": float(realized_df["realized_r"].mean()) if not realized_df.empty else 0.0,
        "win_rate": float((realized_df["realized_pnl"] > 0).sum() / len(realized_df)) if not realized_df.empty else 0.0,
        "max_drawdown": float(pnl_curve["drawdown"].min()) if not pnl_curve.empty else 0.0,
        "equity_max_drawdown": float(account_curve["equity_drawdown"].min()) if not account_curve.empty else 0.0,
        "equity_max_drawdown_pct": float(account_curve["equity_drawdown_pct"].min()) if not account_curve.empty else 0.0,
        "closed_events": float(len(realized_df)),
    }


def build_rows(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, trade in enumerate(trades, start=1):
        m = calc_trade(trade)
        rows.append(
            {
                "序号": idx,
                "日期": trade.get("date", ""),
                "市场": trade.get("market", ""),
                "品种": trade.get("symbol", ""),
                "方向": trade.get("direction", ""),
                "动作": trade.get("action", ""),
                "形态": trade.get("setup", ""),
                "入场": trade.get("entry", ""),
                "止损": trade.get("stop", ""),
                "目标1": trade.get("target1", ""),
                "目标2": trade.get("target2", ""),
                "退出价/最新价": trade.get("exit_price", ""),
                "数量": trade.get("quantity", ""),
                "合约乘数": trade.get("contract_multiplier", ""),
                "单行预估风险": round(m.total_risk, 2),
                "单行预估PnL": round(m.pnl, 2),
                "单行预估R": round(m.r_multiple, 2),
                "检查得分": f"{m.checklist_score}/5",
                "交易逻辑": trade.get("thesis", ""),
            }
        )
    return rows


def load_settings() -> Dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {"initial_equity": 1000000.0}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"initial_equity": max(0.0, to_number(data.get("initial_equity", 1000000.0)))}
    except Exception:
        return {"initial_equity": 1000000.0}


def save_settings(settings: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"initial_equity": max(0.0, to_number(settings.get("initial_equity")))}, f, ensure_ascii=False, indent=2)


def normalize_trade_for_storage(trade: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for col in TRADE_COLUMNS:
        value = trade.get(col, "开仓" if col == "action" else (False if col in CHECKLIST_FIELDS else ""))
        normalized[col] = bool(value) if col in CHECKLIST_FIELDS else ("" if value is None else str(value))
    return normalized


def load_trades() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    try:
        df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)
    except Exception:
        return []
    trades: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        trade: Dict[str, Any] = {}
        for col in TRADE_COLUMNS:
            if col in CHECKLIST_FIELDS:
                trade[col] = str(row.get(col, "False")).lower() in {"true", "1", "yes", "y"}
            elif col == "action":
                trade[col] = row.get(col, "开仓") or "开仓"
            else:
                trade[col] = row.get(col, "")
        trades.append(trade)
    return trades


def save_trades(trades: List[Dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([normalize_trade_for_storage(t) for t in trades], columns=TRADE_COLUMNS).to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


def google_sheets_enabled() -> bool:
    return (
        gspread is not None
        and Credentials is not None
        and "gcp" in st.secrets
        and "service_account_json" in st.secrets["gcp"]
        and "google_sheets" in st.secrets
    )


def get_gspread_client():
    if not google_sheets_enabled():
        return None
    service_account_info = json.loads(st.secrets["gcp"]["service_account_json"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    return gspread.authorize(creds)


def get_google_sheet():
    client = get_gspread_client()
    if client is None:
        return None

    sheet_url = st.secrets["google_sheets"].get("spreadsheet_url")
    return client.open_by_url("https://docs.google.com/spreadsheets/d/1X2nHTspWfBtGLl1SZ8OCwfj7lqCVqlVNR6_MAs5pv_o/edit?gid=0#gid=0)


def get_or_create_worksheet(spreadsheet, title: str, rows: int = 1000, cols: int = 30):
    try:
        return spreadsheet.worksheet(title)
    except Exception:
        return spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)


def sync_to_google_sheets(trades: List[Dict[str, Any]], settings: Dict[str, Any]) -> bool:
    spreadsheet = get_google_sheet()
    if spreadsheet is None:
        return False

    trades_sheet_name = st.secrets["google_sheets"].get("trades_sheet", "trades")
    settings_sheet_name = st.secrets["google_sheets"].get("settings_sheet", "settings")

    trades_ws = get_or_create_worksheet(spreadsheet, trades_sheet_name)
    settings_ws = get_or_create_worksheet(spreadsheet, settings_sheet_name, rows=20, cols=5)

    normalized_trades = [normalize_trade_for_storage(t) for t in trades]
    trades_values = [TRADE_COLUMNS] + [[row.get(col, "") for col in TRADE_COLUMNS] for row in normalized_trades]
    trades_ws.clear()
    if trades_values:
        trades_ws.update(trades_values)

    settings_values = [
        ["key", "value"],
        ["initial_equity", str(max(0.0, to_number(settings.get("initial_equity", 1000000.0))))],
    ]
    settings_ws.clear()
    settings_ws.update(settings_values)
    return True


def load_from_google_sheets() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    spreadsheet = get_google_sheet()
    if spreadsheet is None:
        return [], {"initial_equity": 1000000.0}

    trades_sheet_name = st.secrets["google_sheets"].get("trades_sheet", "trades")
    settings_sheet_name = st.secrets["google_sheets"].get("settings_sheet", "settings")

    trades: List[Dict[str, Any]] = []
    try:
        trades_ws = spreadsheet.worksheet(trades_sheet_name)
        records = trades_ws.get_all_records()
        trades = [normalize_trade_for_storage(normalize_loaded_row(record)) for record in records]
    except Exception:
        trades = []

    settings = {"initial_equity": 1000000.0}
    try:
        settings_ws = spreadsheet.worksheet(settings_sheet_name)
        rows = settings_ws.get_all_values()
        for row in rows[1:]:
            if len(row) >= 2 and row[0] == "initial_equity":
                settings["initial_equity"] = max(0.0, to_number(row[1]))
    except Exception:
        pass

    return trades, settings


def normalize_loaded_row(raw_trade: Dict[str, Any]) -> Dict[str, Any]:
    trade: Dict[str, Any] = {}
    for col in TRADE_COLUMNS:
        if col in CHECKLIST_FIELDS:
            value = raw_trade.get(col, False)
            trade[col] = value if isinstance(value, bool) else str(value).lower() in {"true", "1", "yes", "y"}
        elif col == "action":
            trade[col] = raw_trade.get(col, "开仓") or "开仓"
        else:
            trade[col] = "" if raw_trade.get(col) is None else str(raw_trade.get(col, ""))
    return trade


def assert_almost_equal(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def run_tests() -> None:
    long_open = {"date": "2026-01-01", "symbol": "CU", "direction": "多", "action": "开仓", "entry": "100", "stop": "95", "quantity": "1", "contract_multiplier": "10", "趋势结构确认": True, "量仓配合": True, "盈亏比合格": True}
    long_add = {"date": "2026-01-02", "symbol": "CU", "direction": "多", "action": "加仓", "entry": "110", "stop": "105", "quantity": "1", "contract_multiplier": "10"}
    long_reduce = {"date": "2026-01-03", "symbol": "CU", "direction": "多", "action": "减仓", "exit_price": "108", "quantity": "1", "contract_multiplier": "10"}
    long_close = {"date": "2026-01-04", "symbol": "CU", "direction": "多", "action": "平仓", "exit_price": "120", "quantity": "1", "contract_multiplier": "10"}

    m = calc_trade({**long_open, "exit_price": "110"})
    assert_almost_equal(m.total_risk, 50, "single row total_risk")
    assert_almost_equal(m.pnl, 100, "single row pnl")
    assert_almost_equal(m.r_multiple, 2, "single row r_multiple")
    assert_almost_equal(m.checklist_score, 3, "checklist_score")

    stacks, realized, warnings = build_position_engine([long_close, long_reduce, long_add, long_open])
    assert_equal(len(warnings), 0, "lifo no warnings")
    assert_equal(len(realized), 2, "lifo realized events")
    assert_almost_equal(realized[0]["entry"], 110, "lifo closes last add first")
    assert_almost_equal(realized[0]["realized_pnl"], -20, "lifo first close pnl")
    assert_almost_equal(realized[1]["entry"], 100, "lifo then closes initial lot")
    assert_almost_equal(realized[1]["realized_pnl"], 200, "lifo second close pnl")
    assert_equal(len(stacks.get(("CU", "多"), [])), 0, "all lots closed")

    curve = build_pnl_curve([long_close, long_reduce, long_add, long_open])
    assert_equal(len(curve), 2, "pnl curve realized count")
    assert_almost_equal(float(curve.iloc[-1]["cumulative_pnl"]), 180, "pnl curve cumulative")
    assert_almost_equal(float(curve["drawdown"].min()), -20, "pnl curve drawdown")

    account_curve = build_account_curve([long_close, long_reduce, long_add, long_open], 100000)
    assert_equal(len(account_curve), 4, "account curve row count")
    assert_almost_equal(float(account_curve.iloc[-1]["total_equity"]), 100180, "account final equity")
    assert_almost_equal(float(account_curve.iloc[-1]["leverage"]), 0, "account final leverage")

    short_open = {"date": "2026-01-05", "symbol": "AL", "direction": "空", "action": "开仓", "entry": "100", "stop": "105", "quantity": "1", "contract_multiplier": "10"}
    short_close = {"date": "2026-01-06", "symbol": "AL", "direction": "空", "action": "平仓", "exit_price": "90", "quantity": "1", "contract_multiplier": "10"}
    realized_df = build_realized_df([short_close, short_open])
    assert_almost_equal(float(realized_df.iloc[0]["realized_pnl"]), 100, "short realized pnl")

    _, _, warnings = build_position_engine([{"date": "2026-01-07", "symbol": "ZN", "direction": "多", "action": "平仓", "exit_price": "120", "quantity": "3", "contract_multiplier": "10"}])
    assert_equal(len(warnings), 1, "over close warning")


run_tests()


st.set_page_config(page_title="交易系统录入台", layout="wide")
st.title("交易系统录入台")
st.caption("纯记录工具：LIFO 持仓引擎 + 账户资金曲线 + 风险占用 + 杠杆监控。当前版本不接实时行情。")

if "trades" not in st.session_state:
    st.session_state.trades = load_trades()
if "settings" not in st.session_state:
    st.session_state.settings = load_settings()

with st.sidebar:
    st.header("账户设置")
    initial_equity_input = st.number_input("初始权益", min_value=0.0, value=float(st.session_state.settings.get("initial_equity", 1000000.0)), step=10000.0)
    if st.button("保存账户设置", use_container_width=True):
        st.session_state.settings["initial_equity"] = initial_equity_input
        save_settings(st.session_state.settings)
        st.success("账户设置已保存。")
        st.rerun()
    st.caption("Equity = 初始权益 + 已实现PnL + 未实现PnL。")
    st.caption(f"Google Sheets secrets: {'已读取' if google_sheets_enabled() else '未启用'}")

    if st.button("一键同步到 Google Sheets", use_container_width=True):
        if sync_to_google_sheets(st.session_state.trades, st.session_state.settings):
            st.success("已同步到 Google Sheets。")
        else:
            st.error("Google Sheets 未启用。请检查 secrets、requirements.txt 和表格分享权限。")

    if st.button("从 Google Sheets 读取", use_container_width=True):
        if google_sheets_enabled():
            cloud_trades, cloud_settings = load_from_google_sheets()
            st.session_state.trades = cloud_trades
            st.session_state.settings = cloud_settings
            save_trades(st.session_state.trades)
            save_settings(st.session_state.settings)
            st.success("已从 Google Sheets 读取。")
            st.rerun()
        else:
            st.error("Google Sheets 未启用。请检查 secrets、requirements.txt 和表格分享权限。")

initial_equity = float(st.session_state.settings.get("initial_equity", 1000000.0))
portfolio = calc_portfolio(st.session_state.trades, initial_equity)

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("当前权益", f"¥{portfolio['current_equity']:,.0f}")
kpi2.metric("当前止损风险", f"¥{portfolio['open_current_risk']:,.0f}")
kpi3.metric("风险占用", f"{portfolio['risk_usage'] * 100:.2f}%")
kpi4.metric("名义杠杆", f"{portfolio['leverage']:.2f}x")
kpi5.metric("已实现PnL", f"¥{portfolio['total_realized_pnl']:,.0f}")
kpi6.metric("权益最大回撤", f"{portfolio['equity_max_drawdown_pct'] * 100:.2f}%")

kpi7, kpi8, kpi9, kpi10, kpi11, kpi12 = st.columns(6)
kpi7.metric("未平仓数量", f"{portfolio['open_qty']:,.2f}")
kpi8.metric("未实现PnL", f"¥{portfolio['total_unrealized_pnl']:,.0f}")
kpi9.metric("总PnL", f"¥{portfolio['total_pnl']:,.0f}")
kpi10.metric("平均已实现R", f"{portfolio['avg_realized_r']:.2f}R")
kpi11.metric("已实现胜率", f"{portfolio['win_rate'] * 100:.1f}%")
kpi12.metric("已实现最大回撤", f"¥{portfolio['max_drawdown']:,.0f}")

st.divider()
left, right = st.columns([2.15, 1])

with left:
    st.subheader("新增交易")
    with st.form("trade_form", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        trade_date = c1.date_input("日期", value=date.today())
        market = c2.selectbox("市场", MARKETS)
        symbol = c3.text_input("品种/代码", placeholder="如 CU2509 / 豆粕 / GEV")
        direction = c4.selectbox("方向", DIRECTIONS)
        action = c5.selectbox("开仓/平仓", ACTIONS)

        c1, c2, c3, c4 = st.columns(4)
        entry = c1.text_input("入场价", help="开仓/加仓时填写；减仓/平仓可以留空。")
        stop = c2.text_input("止损价", help="开仓/加仓时填写，用于计算该 lot 初始风险。")
        target1 = c3.text_input("目标1")
        target2 = c4.text_input("目标2")

        c1, c2, c3, c4 = st.columns(4)
        exit_price = c1.text_input("退出价 / 最新价", help="减仓/平仓时作为成交价；未平仓时可作为标记价。")
        quantity = c2.text_input("手数/股数")
        contract_multiplier = c3.text_input("合约乘数", value="10")
        max_risk_rmb = c4.text_input("单笔最大风险 ¥", value="5000")

        setup = st.selectbox("交易形态", SETUPS)
        thesis = st.text_area("交易逻辑 / 复盘备注", placeholder="例如：趋势结构、库存/基差/月差、资金行为、事件风险、失效条件。", height=120)

        st.write("系统检查")
        checklist_cols = st.columns(5)
        checklist_values: Dict[str, bool] = {}
        for idx, field in enumerate(CHECKLIST_FIELDS):
            checklist_values[field] = checklist_cols[idx].checkbox(field)

        submitted = st.form_submit_button("保存交易", use_container_width=True)
        trade = {
            "date": str(trade_date), "market": market, "symbol": symbol, "direction": direction, "action": action,
            "setup": setup, "entry": entry, "stop": stop, "target1": target1, "target2": target2,
            "exit_price": exit_price, "quantity": quantity, "contract_multiplier": contract_multiplier,
            "max_risk_rmb": max_risk_rmb, "thesis": thesis, **checklist_values,
        }
        if submitted:
            if not symbol.strip():
                st.error("请先填写品种/代码。")
            elif to_number(quantity) <= 0:
                st.error("请填写大于0的数量。")
            elif action in OPEN_ACTIONS and to_number(entry) == 0:
                st.error("开仓/加仓需要填写入场价。")
            elif action in CLOSE_ACTIONS and to_number(exit_price) == 0:
                st.error("减仓/平仓需要填写退出价。")
            else:
                st.session_state.trades.insert(0, trade)
                save_trades(st.session_state.trades)
                st.success("已保存交易。")
                st.rerun()

with right:
    st.subheader("单行预估")
    preview_metrics = calc_trade(trade if "trade" in locals() else {})
    st.metric("每手/每单位风险", f"¥{preview_metrics.risk_per_unit:,.0f}")
    st.metric("本笔总风险", f"¥{preview_metrics.total_risk:,.0f}")
    st.metric("按风险上限可开", f"{preview_metrics.max_qty_by_risk} 手/单位")
    st.metric("目标1盈亏比", f"{preview_metrics.reward_risk_1:.2f}R")
    st.metric("目标2盈亏比", f"{preview_metrics.reward_risk_2:.2f}R")
    st.metric("单行预估PnL", f"¥{preview_metrics.pnl:,.0f}")
    st.metric("单行预估R", f"{preview_metrics.r_multiple:.2f}R")
    st.info(f"系统检查得分：{preview_metrics.checklist_score}/5")

st.divider()
st.subheader("当前持仓 / Open Position")
open_df = build_open_positions(st.session_state.trades)
lot_df = build_lot_detail(st.session_state.trades)
_, _, engine_warnings = build_position_engine(st.session_state.trades)
if engine_warnings:
    with st.expander("持仓引擎警告", expanded=True):
        st.dataframe(pd.DataFrame(engine_warnings), use_container_width=True, hide_index=True)
if not open_df.empty:
    st.dataframe(open_df, use_container_width=True, hide_index=True)
    with st.expander("子仓位明细（LIFO lots）"):
        st.dataframe(lot_df, use_container_width=True, hide_index=True)
else:
    st.info("当前无未平仓持仓。")

st.divider()
st.subheader("账户权益 / 风险占用 / 杠杆")
account_curve = build_account_curve(st.session_state.trades, initial_equity)
if not account_curve.empty:
    st.line_chart(account_curve.set_index("trade_no")[["realized_equity", "total_equity"]], use_container_width=True)
    st.line_chart(account_curve.set_index("trade_no")[["risk_usage", "leverage"]], use_container_width=True)
    with st.expander("账户曲线明细"):
        display_account = account_curve.copy()
        display_account["risk_usage"] = display_account["risk_usage"].map(lambda x: f"{x * 100:.2f}%")
        display_account["equity_drawdown_pct"] = display_account["equity_drawdown_pct"].map(lambda x: f"{x * 100:.2f}%")
        st.dataframe(display_account, use_container_width=True, hide_index=True)
else:
    st.info("暂无账户曲线。保存交易后会生成 equity、risk usage 和 leverage 曲线。")

st.divider()
st.subheader("已实现PnL曲线与回撤")
pnl_curve = build_pnl_curve(st.session_state.trades)
realized_df = build_realized_df(st.session_state.trades)
if not pnl_curve.empty:
    st.line_chart(pnl_curve.set_index("close_no")[["cumulative_pnl", "drawdown"]], use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前累计已实现PnL", f"¥{pnl_curve.iloc[-1]['cumulative_pnl']:,.0f}")
    c2.metric("最高累计已实现PnL", f"¥{pnl_curve['peak'].max():,.0f}")
    c3.metric("最大回撤", f"¥{pnl_curve['drawdown'].min():,.0f}")
    c4.metric("已实现胜率", f"{portfolio['win_rate'] * 100:.1f}%")
    with st.expander("LIFO 平仓匹配明细"):
        st.dataframe(realized_df, use_container_width=True, hide_index=True)
else:
    st.info("暂无已实现平仓记录，减仓/平仓后会自动生成 PnL 曲线。")

st.divider()
st.subheader("交易记录")
if st.session_state.trades:
    df = pd.DataFrame(build_rows(st.session_state.trades))
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    market_filter = c1.selectbox("筛选市场", ["全部"] + MARKETS)
    direction_filter = c2.selectbox("筛选方向", ["全部"] + DIRECTIONS)
    action_filter = c3.selectbox("筛选动作", ["全部"] + ACTIONS)
    symbol_filter = c4.text_input("筛选品种/代码包含")
    filtered_df = df.copy()
    if market_filter != "全部":
        filtered_df = filtered_df[filtered_df["市场"] == market_filter]
    if direction_filter != "全部":
        filtered_df = filtered_df[filtered_df["方向"] == direction_filter]
    if action_filter != "全部":
        filtered_df = filtered_df[filtered_df["动作"] == action_filter]
    if symbol_filter.strip():
        filtered_df = filtered_df[filtered_df["品种"].astype(str).str.contains(symbol_filter.strip(), case=False, na=False)]
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    st.download_button("下载当前筛选结果 CSV", data=filtered_df.to_csv(index=False).encode("utf-8-sig"), file_name="trade_journal.csv", mime="text/csv", use_container_width=True)
    col1, col2 = st.columns(2)
    if col1.button("保存当前全部交易到 data/trades.csv", type="secondary", use_container_width=True):
        save_trades(st.session_state.trades)
        st.success("已保存。")
    if col2.button("清空全部交易", type="secondary", use_container_width=True):
        st.session_state.trades = []
        save_trades(st.session_state.trades)
        st.rerun()
else:
    st.info("暂无交易记录。先录入一笔交易。")

st.caption(f"数据文件：{DATA_FILE.as_posix()}；账户设置文件：{SETTINGS_FILE.as_posix()}。Streamlit Cloud 本地文件存储适合轻量使用；长期稳定存储建议接 Supabase、Google Sheets 或数据库。")
