import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


# -----------------------------
# Calculation layer
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
SETUPS = ["趋势突破", "回调入场", "区间反转", "基本面事件", "价差/套利"]


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
    sign = -1 if direction == "空" else 1

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


def calc_portfolio(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {"total_risk": 0.0, "total_pnl": 0.0, "avg_r": 0.0, "win_rate": 0.0}

    metrics = [calc_trade(t) for t in trades]
    total_risk = sum(m.total_risk for m in metrics)
    total_pnl = sum(m.pnl for m in metrics)
    avg_r = sum(m.r_multiple for m in metrics) / len(metrics)
    win_rate = sum(1 for m in metrics if m.pnl > 0) / len(metrics)
    return {
        "total_risk": total_risk,
        "total_pnl": total_pnl,
        "avg_r": avg_r,
        "win_rate": win_rate,
    }


def build_rows(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trade in trades:
        m = calc_trade(trade)
        rows.append(
            {
                "日期": trade.get("date", ""),
                "市场": trade.get("market", ""),
                "品种": trade.get("symbol", ""),
                "方向": trade.get("direction", ""),
                "形态": trade.get("setup", ""),
                "入场": trade.get("entry", ""),
                "止损": trade.get("stop", ""),
                "目标1": trade.get("target1", ""),
                "目标2": trade.get("target2", ""),
                "退出价/最新价": trade.get("exit_price", ""),
                "数量": trade.get("quantity", ""),
                "合约乘数": trade.get("contract_multiplier", ""),
                "总风险": round(m.total_risk, 2),
                "盈亏": round(m.pnl, 2),
                "R倍数": round(m.r_multiple, 2),
                "风险占权益": f"{m.risk_pct * 100:.2f}%",
                "检查得分": f"{m.checklist_score}/5",
                "交易逻辑": trade.get("thesis", ""),
            }
        )
    return rows


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
    long_trade = {
        "direction": "多",
        "entry": "100",
        "stop": "95",
        "exit_price": "110",
        "target1": "110",
        "target2": "120",
        "quantity": "2",
        "contract_multiplier": "10",
        "max_risk_rmb": "5000",
        "account_equity": "100000",
        "趋势结构确认": True,
        "量仓配合": True,
        "盈亏比合格": True,
    }
    m = calc_trade(long_trade)
    assert_almost_equal(m.risk_per_unit, 50, "long risk_per_unit")
    assert_almost_equal(m.total_risk, 100, "long total_risk")
    assert_almost_equal(m.pnl, 200, "long pnl")
    assert_almost_equal(m.r_multiple, 2, "long r_multiple")
    assert_almost_equal(m.reward_risk_1, 2, "long reward_risk_1")
    assert_almost_equal(m.reward_risk_2, 4, "long reward_risk_2")
    assert_almost_equal(m.max_qty_by_risk, 100, "max_qty_by_risk")
    assert_almost_equal(m.risk_pct, 0.001, "risk_pct")
    assert_almost_equal(m.checklist_score, 3, "checklist_score")

    short_trade = {
        "direction": "空",
        "entry": "100",
        "stop": "105",
        "exit_price": "90",
        "quantity": "1",
        "contract_multiplier": "10",
        "max_risk_rmb": "5000",
    }
    m = calc_trade(short_trade)
    assert_almost_equal(m.total_risk, 50, "short total_risk")
    assert_almost_equal(m.pnl, 100, "short pnl")
    assert_almost_equal(m.r_multiple, 2, "short r_multiple")

    invalid_trade = {
        "direction": "多",
        "entry": "abc",
        "stop": None,
        "exit_price": "bad",
        "quantity": "not-a-number",
        "contract_multiplier": "bad",
    }
    m = calc_trade(invalid_trade)
    assert_almost_equal(m.total_risk, 0, "invalid total_risk")
    assert_almost_equal(m.pnl, 0, "invalid pnl")
    assert_almost_equal(m.r_multiple, 0, "invalid r_multiple")

    comma_trade = {
        "direction": "多",
        "entry": "5,000",
        "stop": "4,900",
        "exit_price": "5,100",
        "quantity": "1",
        "contract_multiplier": "5",
        "max_risk_rmb": "2,000",
    }
    m = calc_trade(comma_trade)
    assert_almost_equal(m.total_risk, 500, "comma total_risk")
    assert_almost_equal(m.pnl, 500, "comma pnl")
    assert_almost_equal(m.max_qty_by_risk, 4, "comma max_qty_by_risk")

    negative_qty_trade = {
        "direction": "多",
        "entry": "100",
        "stop": "90",
        "exit_price": "110",
        "quantity": "-3",
        "contract_multiplier": "10",
    }
    m = calc_trade(negative_qty_trade)
    assert_almost_equal(m.total_risk, 0, "negative quantity risk clamps to zero")
    assert_almost_equal(m.pnl, 0, "negative quantity pnl clamps to zero")

    portfolio = calc_portfolio([long_trade, short_trade, negative_qty_trade])
    assert_almost_equal(portfolio["total_risk"], 150, "portfolio total_risk")
    assert_almost_equal(portfolio["total_pnl"], 300, "portfolio total_pnl")
    assert_almost_equal(portfolio["avg_r"], (2 + 2 + 0) / 3, "portfolio avg_r")
    assert_almost_equal(portfolio["win_rate"], 2 / 3, "portfolio win_rate")

    rows = build_rows([long_trade])
    assert_equal(rows[0]["总风险"], 100, "export total_risk")
    assert_equal(rows[0]["盈亏"], 200, "export pnl")
    assert_equal(rows[0]["检查得分"], "3/5", "export checklist score")


run_tests()


# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(page_title="交易系统录入台", layout="wide")

st.title("交易系统录入台")
st.caption("纯记录工具：手动录入交易参数，自动计算风险、R倍数、盈亏和系统执行度。当前版本不接实时行情。")

if "trades" not in st.session_state:
    st.session_state.trades = []

if "draft_trade" not in st.session_state:
    st.session_state.draft_trade = {}

portfolio = calc_portfolio(st.session_state.trades)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("组合风险占用", f"¥{portfolio['total_risk']:,.0f}")
kpi2.metric("已记录盈亏", f"¥{portfolio['total_pnl']:,.0f}")
kpi3.metric("平均R", f"{portfolio['avg_r']:.2f}R")
kpi4.metric("胜率", f"{portfolio['win_rate'] * 100:.1f}%")

st.divider()

left, right = st.columns([2.15, 1])

with left:
    st.subheader("新增交易")
    with st.form("trade_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        trade_date = c1.date_input("日期", value=date.today())
        market = c2.selectbox("市场", MARKETS)
        symbol = c3.text_input("品种/代码", placeholder="如 CU2509 / 豆粕 / GEV")
        direction = c4.selectbox("方向", DIRECTIONS)

        c1, c2, c3, c4 = st.columns(4)
        entry = c1.text_input("入场价")
        stop = c2.text_input("止损价")
        target1 = c3.text_input("目标1")
        target2 = c4.text_input("目标2")

        c1, c2, c3, c4 = st.columns(4)
        exit_price = c1.text_input("退出价 / 最新价")
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
            else:
                st.session_state.trades.insert(0, trade)
                st.success("已保存交易。")
                st.rerun()

with right:
    st.subheader("自动计算")
    preview_metrics = calc_trade(trade if "trade" in locals() else {})
    st.metric("每手/每单位风险", f"¥{preview_metrics.risk_per_unit:,.0f}")
    st.metric("本笔总风险", f"¥{preview_metrics.total_risk:,.0f}")
    st.metric("按风险上限可开", f"{preview_metrics.max_qty_by_risk} 手/单位")
    st.metric("风险占账户权益", f"{preview_metrics.risk_pct * 100:.2f}%")
    st.metric("目标1盈亏比", f"{preview_metrics.reward_risk_1:.2f}R")
    st.metric("目标2盈亏比", f"{preview_metrics.reward_risk_2:.2f}R")
    st.metric("已实现/浮动盈亏", f"¥{preview_metrics.pnl:,.0f}")
    st.metric("本笔R倍数", f"{preview_metrics.r_multiple:.2f}R")
    st.info(f"系统检查得分：{preview_metrics.checklist_score}/5")

st.divider()

st.subheader("交易记录")

if st.session_state.trades:
    rows = build_rows(st.session_state.trades)
    df = pd.DataFrame(rows)

    c1, c2, c3 = st.columns([1, 1, 2])
    market_filter = c1.selectbox("筛选市场", ["全部"] + MARKETS)
    direction_filter = c2.selectbox("筛选方向", ["全部"] + DIRECTIONS)
    symbol_filter = c3.text_input("筛选品种/代码包含")

    filtered_df = df.copy()
    if market_filter != "全部":
        filtered_df = filtered_df[filtered_df["市场"] == market_filter]
    if direction_filter != "全部":
        filtered_df = filtered_df[filtered_df["方向"] == direction_filter]
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

    if st.button("清空全部交易", type="secondary"):
        st.session_state.trades = []
        st.rerun()
else:
    st.info("暂无交易记录。先录入一笔交易。")


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
