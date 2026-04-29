import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Storage config
# -----------------------------

DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "trades.csv"


# -----------------------------
# Static config
# -----------------------------

CHECKLIST_FIELDS = [
    "趋势结构确认",
    "量仓配合",
    "基本面支持",
    "事件风险可控",
    "盈亏比合格",
]

MARKETS = ["国内期货", "海外期货", "A股", "美股", "期权"]
DIRECTIONS = ["多", "空"]
ACTIONS = ["开仓", "加仓", "减仓", "平仓"]
OPEN_ACTIONS = {"开仓", "加仓"}
CLOSE_ACTIONS = {"减仓", "平仓"}
SETUPS = ["趋势突破", "回调入场", "区间反转", "基本面事件", "价差/套利"]

TRADE_COLUMNS = [
    "date",
    "market",
    "symbol",
    "direction",
    "action",
    "setup",
    "entry",
    "stop",
    "target1",
    "target2",
    "exit_price",
    "quantity",
    "contract_multiplier",
    "account_equity",
    "max_risk_rmb",
    "thesis",
] + CHECKLIST_FIELDS


# -----------------------------
# Calculation layer
# -----------------------------

def to_number(value: Any) -> float:
    """Safely parse user input into a finite float."""
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else 0.0
    try:
        cleaned = str(value).replace(",", "").strip()
        parsed = float(cleaned)
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
    risk_pct: float = 0.0
    reward_risk_1: float = 0.0
    reward_risk_2: float = 0.0
    checklist_score: int = 0


def calc_trade(trade: Dict[str, Any]) -> Metrics:
    """Single-row preview calculation.

    This is still useful for the form preview. Portfolio realized PnL and open
    positions are computed by the LIFO position engine below.
    """
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
    equity = max(0.0, to_number(trade.get("account_equity")))
    direction = trade.get("direction", "多")
    sign = direction_sign(direction)

    has_entry = entry != 0
    has_stop = stop != 0
    has_exit = exit_price != 0

    risk_per_unit = abs(entry - stop) * multiplier if has_entry and has_stop else 0.0
    total_risk = risk_per_unit * qty
    max_qty_by_risk = int(max_risk // risk_per_unit) if risk_per_unit > 0 else 0
    pnl = (exit_price - entry) * sign * multiplier * qty if has_entry and has_exit and qty > 0 else 0.0
    r_multiple = pnl / total_risk if total_risk > 0 else 0.0
    risk_pct = total_risk / equity if equity > 0 else 0.0
    reward_risk_1 = abs(target1 - entry) * multiplier / risk_per_unit if risk_per_unit > 0 and target1 != 0 else 0.0
    reward_risk_2 = abs(target2 - entry) * multiplier / risk_per_unit if risk_per_unit > 0 and target2 != 0 else 0.0
    checklist_score = sum(1 for field in CHECKLIST_FIELDS if bool(trade.get(field, False)))

    return Metrics(
        risk_per_unit=risk_per_unit,
        total_risk=total_risk,
        max_qty_by_risk=max_qty_by_risk,
        pnl=pnl,
        r_multiple=r_multiple,
        risk_pct=risk_pct,
        reward_risk_1=reward_risk_1,
        reward_risk_2=reward_risk_2,
        checklist_score=checklist_score,
    )


# -----------------------------
# LIFO position engine
# -----------------------------

def chronological_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Session stores newest first. Engine consumes oldest first."""
    return list(reversed(trades))


def make_lot(trade: Dict[str, Any], lot_id: int) -> Dict[str, Any]:
    entry = to_number(trade.get("entry"))
    stop = to_number(trade.get("stop"))
    qty = max(0.0, to_number(trade.get("quantity")))
    multiplier_raw = to_number(trade.get("contract_multiplier"))
    multiplier = multiplier_raw if multiplier_raw > 0 else 1.0
    risk_per_unit = abs(entry - stop) * multiplier if entry != 0 and stop != 0 else 0.0
    initial_risk = risk_per_unit * qty
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
        "initial_risk": initial_risk,
        "risk_per_unit": risk_per_unit,
        "thesis": trade.get("thesis", ""),
    }


def build_position_engine(trades: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], List[Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build LIFO open lots and realized close events.

    Key = (symbol, direction). A close row closes the most recently opened lot
    for the same symbol and same direction first.
    """
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
            lot = make_lot(trade, lot_id)
            stacks.setdefault(key, []).append(lot)
            continue

        if action in CLOSE_ACTIONS:
            close_price = to_number(trade.get("exit_price"))
            if close_price == 0:
                warnings.append({"trade_no": trade_no, "symbol": symbol, "reason": "减仓/平仓记录缺少退出价，无法计算已实现PnL。"})
                continue

            remaining_to_close = qty
            stack = stacks.setdefault(key, [])

            while remaining_to_close > 0 and stack:
                lot = stack[-1]
                matched_qty = min(remaining_to_close, lot["remaining_qty"])
                sign = direction_sign(direction)
                pnl = (close_price - lot["entry"]) * sign * lot["contract_multiplier"] * matched_qty
                matched_initial_risk = lot["risk_per_unit"] * matched_qty
                realized_r = pnl / matched_initial_risk if matched_initial_risk > 0 else 0.0

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
                        "realized_r": realized_r,
                    }
                )

                lot["remaining_qty"] -= matched_qty
                remaining_to_close -= matched_qty

                if lot["remaining_qty"] <= 1e-12:
                    stack.pop()

            if remaining_to_close > 1e-12:
                warnings.append(
                    {
                        "trade_no": trade_no,
                        "symbol": symbol,
                        "reason": f"平仓数量超过当前持仓，未匹配数量：{remaining_to_close:g}",
                    }
                )

    return stacks, realized_events, warnings


def latest_prices_by_symbol(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    latest: Dict[str, float] = {}
    for trade in trades:
        symbol = str(trade.get("symbol", "")).strip()
        if not symbol or symbol in latest:
            continue
        price = to_number(trade.get("exit_price"))
        if price == 0:
            price = to_number(trade.get("entry"))
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
        if mark_price != 0:
            unrealized_pnl = sum((mark_price - lot["entry"]) * sign * lot["contract_multiplier"] * lot["remaining_qty"] for lot in open_lots)

        current_risk_to_stop = 0.0
        for lot in open_lots:
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
    if not realized_events:
        return pd.DataFrame(columns=["trade_no", "close_date", "symbol", "direction", "action", "lot_id", "open_date", "entry", "close_price", "qty", "initial_risk", "realized_pnl", "realized_r"])
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
        drawdown = cumulative - peak
        rows.append(
            {
                "close_no": idx + 1,
                "date": row["close_date"],
                "symbol": row["symbol"],
                "realized_pnl": pnl,
                "cumulative_pnl": cumulative,
                "peak": peak,
                "drawdown": drawdown,
            }
        )
    return pd.DataFrame(rows)


def calc_portfolio(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    open_df = build_open_positions(trades)
    realized_df = build_realized_df(trades)
    pnl_curve = build_pnl_curve(trades)

    total_realized_pnl = float(realized_df["realized_pnl"].sum()) if not realized_df.empty else 0.0
    total_unrealized_pnl = float(open_df["未实现PnL"].sum()) if not open_df.empty else 0.0
    open_qty = float(open_df["未平仓数量"].sum()) if not open_df.empty else 0.0
    open_initial_risk = float(open_df["剩余初始风险"].sum()) if not open_df.empty else 0.0
    open_current_risk = float(open_df["当前止损风险"].sum()) if not open_df.empty else 0.0
    realized_r = float(realized_df["realized_r"].mean()) if not realized_df.empty else 0.0
    win_rate = float((realized_df["realized_pnl"] > 0).sum() / len(realized_df)) if not realized_df.empty else 0.0
    max_drawdown = float(pnl_curve["drawdown"].min()) if not pnl_curve.empty else 0.0

    return {
        "open_qty": open_qty,
        "open_initial_risk": open_initial_risk,
        "open_current_risk": open_current_risk,
        "total_realized_pnl": total_realized_pnl,
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_pnl": total_realized_pnl + total_unrealized_pnl,
        "avg_realized_r": realized_r,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
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
                "风险占权益": f"{m.risk_pct * 100:.2f}%",
                "检查得分": f"{m.checklist_score}/5",
                "交易逻辑": trade.get("thesis", ""),
            }
        )
    return rows


# -----------------------------
# Persistence layer
# -----------------------------

def normalize_trade_for_storage(trade: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for col in TRADE_COLUMNS:
        if col == "action":
            value = trade.get(col, "开仓") or "开仓"
        else:
            value = trade.get(col, False if col in CHECKLIST_FIELDS else "")
        if col in CHECKLIST_FIELDS:
            normalized[col] = bool(value)
        else:
            normalized[col] = "" if value is None else str(value)
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
    normalized = [normalize_trade_for_storage(t) for t in trades]
    df = pd.DataFrame(normalized, columns=TRADE_COLUMNS)
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


# -----------------------------
# Tests
# -----------------------------

def assert_almost_equal(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def run_tests() -> None:
    long_open = {
        "date": "2026-01-01",
        "symbol": "CU",
        "direction": "多",
        "action": "开仓",
        "entry": "100",
        "stop": "95",
        "exit_price": "",
        "quantity": "1",
        "contract_multiplier": "10",
        "max_risk_rmb": "5000",
        "account_equity": "100000",
        "趋势结构确认": True,
        "量仓配合": True,
        "盈亏比合格": True,
    }
    long_add = {
        "date": "2026-01-02",
        "symbol": "CU",
        "direction": "多",
        "action": "加仓",
        "entry": "110",
        "stop": "105",
        "exit_price": "",
        "quantity": "1",
        "contract_multiplier": "10",
    }
    long_reduce = {
        "date": "2026-01-03",
        "symbol": "CU",
        "direction": "多",
        "action": "减仓",
        "entry": "",
        "stop": "",
        "exit_price": "108",
        "quantity": "1",
        "contract_multiplier": "10",
    }
    long_close = {
        "date": "2026-01-04",
        "symbol": "CU",
        "direction": "多",
        "action": "平仓",
        "entry": "",
        "stop": "",
        "exit_price": "120",
        "quantity": "1",
        "contract_multiplier": "10",
    }

    m = calc_trade({**long_open, "exit_price": "110"})
    assert_almost_equal(m.risk_per_unit, 50, "single row risk_per_unit")
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

    open_df = build_open_positions([long_add, long_open])
    assert_equal(len(open_df), 1, "open position row count")
    assert_almost_equal(float(open_df.iloc[0]["未平仓数量"]), 2, "open position qty")
    assert_almost_equal(float(open_df.iloc[0]["加权开仓价"]), 105, "weighted entry")

    short_open = {
        "date": "2026-01-05",
        "symbol": "AL",
        "direction": "空",
        "action": "开仓",
        "entry": "100",
        "stop": "105",
        "quantity": "1",
        "contract_multiplier": "10",
    }
    short_close = {
        "date": "2026-01-06",
        "symbol": "AL",
        "direction": "空",
        "action": "平仓",
        "exit_price": "90",
        "quantity": "1",
        "contract_multiplier": "10",
    }
    realized_df = build_realized_df([short_close, short_open])
    assert_almost_equal(float(realized_df.iloc[0]["realized_pnl"]), 100, "short realized pnl")

    over_close = {
        "date": "2026-01-07",
        "symbol": "ZN",
        "direction": "多",
        "action": "平仓",
        "exit_price": "120",
        "quantity": "3",
        "contract_multiplier": "10",
    }
    _, _, warnings = build_position_engine([over_close])
    assert_equal(len(warnings), 1, "over close warning")

    normalized = normalize_trade_for_storage(long_open)
    assert_equal(normalized["action"], "开仓", "storage action")
    assert_equal(normalized["趋势结构确认"], True, "storage bool true")
    assert_equal(normalized["事件风险可控"], False, "storage bool false default")


run_tests()


# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(page_title="交易系统录入台", layout="wide")

st.title("交易系统录入台")
st.caption("纯记录工具：手动录入交易参数，使用 LIFO 持仓引擎计算未平仓、已实现PnL、资金曲线和R倍数。当前版本不接实时行情。")

if "trades" not in st.session_state:
    st.session_state.trades = load_trades()

portfolio = calc_portfolio(st.session_state.trades)

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("未平仓数量", f"{portfolio['open_qty']:,.2f}")
kpi2.metric("当前止损风险", f"¥{portfolio['open_current_risk']:,.0f}")
kpi3.metric("已实现PnL", f"¥{portfolio['total_realized_pnl']:,.0f}")
kpi4.metric("未实现PnL", f"¥{portfolio['total_unrealized_pnl']:,.0f}")
kpi5.metric("平均已实现R", f"{portfolio['avg_realized_r']:.2f}R")
kpi6.metric("最大回撤", f"¥{portfolio['max_drawdown']:,.0f}")

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
        stop = c2.text_input("止损价", help="开仓/加仓时填写，用于计算该子仓位初始风险。")
        target1 = c3.text_input("目标1")
        target2 = c4.text_input("目标2")

        c1, c2, c3, c4 = st.columns(4)
        exit_price = c1.text_input("退出价 / 最新价", help="减仓/平仓时作为成交价；未平仓时可作为最新标记价。")
        quantity = c2.text_input("手数/股数")
        contract_multiplier = c3.text_input("合约乘数", value="10")
        max_risk_rmb = c4.text_input("单笔最大风险 ¥", value="5000")

        c1, c2 = st.columns(2)
        account_equity = c1.text_input("账户权益 ¥")
        setup = c2.selectbox("交易形态", SETUPS)

        thesis = st.text_area(
            "交易逻辑 / 复盘备注",
            placeholder="例如：趋势结构、库存/基差/月差、资金行为、事件风险、失效条件。",
            height=120,
        )

        st.write("系统检查")
        checklist_cols = st.columns(5)
        checklist_values: Dict[str, bool] = {}
        for idx, field in enumerate(CHECKLIST_FIELDS):
            checklist_values[field] = checklist_cols[idx].checkbox(field)

        submitted = st.form_submit_button("保存交易", use_container_width=True)

        trade = {
            "date": str(trade_date),
            "market": market,
            "symbol": symbol,
            "direction": direction,
            "action": action,
            "setup": setup,
            "entry": entry,
            "stop": stop,
            "target1": target1,
            "target2": target2,
            "exit_price": exit_price,
            "quantity": quantity,
            "contract_multiplier": contract_multiplier,
            "account_equity": account_equity,
            "max_risk_rmb": max_risk_rmb,
            "thesis": thesis,
            **checklist_values,
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
    st.metric("风险占账户权益", f"{preview_metrics.risk_pct * 100:.2f}%")
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

st.subheader("已实现PnL曲线与回撤")

pnl_curve = build_pnl_curve(st.session_state.trades)
realized_df = build_realized_df(st.session_state.trades)

if not pnl_curve.empty:
    chart_df = pnl_curve.set_index("close_no")[["cumulative_pnl", "drawdown"]]
    st.line_chart(chart_df, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前累计已实现PnL", f"¥{pnl_curve.iloc[-1]['cumulative_pnl']:,.0f}")
    c2.metric("最高累计已实现PnL", f"¥{pnl_curve['peak'].max():,.0f}")
    c3.metric("最大回撤", f"¥{pnl_curve['drawdown'].min():,.0f}")
    c4.metric("已实现胜率", f"{portfolio['win_rate'] * 100:.1f}%")

    with st.expander("LIFO 平仓匹配明细"):
        display_realized = realized_df.rename(
            columns={
                "close_date": "平仓日期",
                "symbol": "品种",
                "direction": "方向",
                "action": "动作",
                "lot_id": "Lot ID",
                "open_date": "开仓日期",
                "entry": "开仓价",
                "close_price": "平仓价",
                "qty": "匹配数量",
                "initial_risk": "匹配初始风险",
                "realized_pnl": "已实现PnL",
                "realized_r": "已实现R",
            }
        )
        st.dataframe(display_realized, use_container_width=True, hide_index=True)
else:
    st.info("暂无已实现平仓记录，减仓/平仓后会自动生成 PnL 曲线。")

st.divider()

st.subheader("交易记录")

if st.session_state.trades:
    rows = build_rows(st.session_state.trades)
    df = pd.DataFrame(rows)

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

    csv = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="下载当前筛选结果 CSV",
        data=csv,
        file_name="trade_journal.csv",
        mime="text/csv",
        use_container_width=True,
    )

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

st.caption(f"数据文件：{DATA_FILE.as_posix()}。部署到 Streamlit Cloud 时，本地文件存储适合轻量使用；长期稳定存储建议后续接 Supabase、Google Sheets 或数据库。")


# -----------------------------
# How to run locally
# -----------------------------
# 1. pip install streamlit pandas
# 2. save this file as app.py
# 3. streamlit run app.py
#
# For deployment:
# create requirements.txt with:
# streamlit
# pandas
