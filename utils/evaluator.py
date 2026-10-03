# -*- coding: utf-8 -*-
"""
抽取效果评估模块（闭环关键）
----------------------------
模拟数据自带 ground truth（label_amount / label_currency / label_style），
对 LLM 抽取结果做字段级自动评估：
  - amount_hit：相对误差 < 1% 视为命中（金额抽取允许尾差，不允许错数量级）
  - currency_hit：币种精确匹配
  - 按 label_style（5 种金额书写形态）分组统计，定位模型短板；
  - LLM 通道 vs 正则 baseline 同表对比，量化"为什么值得上 LLM"。

这套"带标注数据 → 抽取 → 字段级评估 → 短板分析"的闭环，等价于一个轻量
LLM 抽取评测集，是模型选型 / Prompt 迭代的回归测试。
"""
import pandas as pd

AMOUNT_TOL = 0.01  # 相对误差 1% 内算命中


def evaluate(df: pd.DataFrame, method_name: str = "llm") -> dict:
    d = df.dropna(subset=["label_amount"]).copy()
    if d.empty:
        return {"method": method_name, "n": 0}

    ext = pd.to_numeric(d["ext_amount"], errors="coerce")
    d["amount_hit"] = ((ext - d["label_amount"]).abs() / d["label_amount"] < AMOUNT_TOL)
    d["currency_hit"] = d["ext_currency"] == d["label_currency"]

    by_style = d.groupby("label_style").agg(
        n=("amount_hit", "size"),
        amount_acc=("amount_hit", "mean"),
        currency_acc=("currency_hit", "mean"),
    ).round(4)

    return {
        "method": method_name,
        "n": len(d),
        "extract_success_rate": round(ext.notna().mean(), 4),
        "amount_acc": round(d["amount_hit"].mean(), 4),
        "currency_acc": round(d["currency_hit"].mean(), 4),
        "by_style": by_style.reset_index().to_dict(orient="records"),
        "_df": d,
    }


def compare_table(results: list) -> pd.DataFrame:
    """多通道评估结果并排对比（总体指标）。"""
    rows = [{
        "抽取通道": r["method"],
        "样本数": r["n"],
        "抽取成功率": r.get("extract_success_rate"),
        "金额准确率": r.get("amount_acc"),
        "币种准确率": r.get("currency_acc"),
    } for r in results if r.get("n")]
    return pd.DataFrame(rows)


def style_table(results: list) -> pd.DataFrame:
    """按金额书写形态分组的准确率明细（定位短板）。"""
    frames = []
    for r in results:
        if not r.get("n"):
            continue
        t = pd.DataFrame(r["by_style"])
        t.insert(0, "抽取通道", r["method"])
        frames.append(t)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["label_style", "抽取通道"])
