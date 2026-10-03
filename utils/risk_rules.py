# -*- coding: utf-8 -*-
"""
异常资金识别规则引擎（可解释）
------------------------------
对"LLM 抽取 + 币种归一化"后的结构化流水做行级风险评分：
  R1 大额交易        单笔等值 >= 100万 CNY（监管大额交易报告演示口径）     +50
  R2 结构化拆分      同一付款人同日 >=3 笔、单笔压申报线下、累计超申报线   +60
                     （围绕等值 5 万 USD 跨境申报线的"化整为零"识别）
  R3 高风险地区      交易一方位于离岸金融中心名单                          +45
  R4 临界规避        单笔落在申报线 90%~100% 区间，疑似刻意压线           +30
  R5 夜间交易        0:00-5:00 非营业时段                                  +15
  R6 数据质量        LLM 抽取失败（降级/无金额），金额无法参与上述规则     +10

每条命中规则都会写入 risk_reasons，风险分 = 加权和封顶 100 —— 保证风控结论
可解释、可回溯（监管审计的基本要求），而不是黑盒打分。
"""
import numpy as np
import pandas as pd

import config as C

RULES_DOC = {
    "R1": f"大额交易(≥等值{C.LARGE_TX_CNY / 10000:.0f}万CNY)",
    "R2": "结构化拆分(同日多笔压申报线)",
    "R3": "高风险地区",
    "R4": "临界规避(申报线90%~100%)",
    "R5": "夜间交易",
    "R6": "数据质量(抽取失败)",
}


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """输入需已含 cny_amount / ts / payer / region_src / region_dst / extract_method。"""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["hour"] = df["ts"].dt.hour
    cny = df["cny_amount"]
    reasons = [[] for _ in range(len(df))]
    score = np.zeros(len(df), dtype=float)

    def hit(mask, rule, pts, note=""):
        idx = np.where(mask & (score < 100))[0]
        score[idx] += pts
        for i in idx:
            note_txt = note(i) if callable(note) else note
            reasons[i].append(f"{rule}{note_txt}")

    # R1 大额
    hit(cny >= C.LARGE_TX_CNY, "R1", 50,
        lambda i: f"(等值{df['cny_amount'].iloc[i] / 10000:.0f}万)")

    # R2 结构化拆分：仅交易流水参与（合同无付款时点语义）
    tx = df[df["record_type"] == "transaction"]
    if len(tx):
        g = tx.groupby([tx["payer"], tx["ts"].dt.date])["cny_amount"]
        cnt = g.transform("count")
        total = g.transform("sum")
        smurf_mask = pd.Series(False, index=df.index)
        smurf_mask.loc[tx.index] = (cnt >= C.SPLIT_MIN_COUNT) & \
                                   (tx["cny_amount"] < C.SPLIT_THRESHOLD_CNY) & \
                                   (total >= C.SPLIT_THRESHOLD_CNY)
        hit(smurf_mask.values, "R2", 60, lambda i: f"(同日{cnt.reindex(df.index).iloc[i]:.0f}笔累计{(total.reindex(df.index).iloc[i]) / 10000:.0f}万)")

    # R3 高风险地区
    hi_mask = df["region_src"].isin(C.HIGH_RISK_REGIONS) | df["region_dst"].isin(C.HIGH_RISK_REGIONS)
    hit(hi_mask.values, "R3", 45, lambda i: f"({df['region_dst'].iloc[i]})")

    # R4 临界规避
    dodge_mask = (cny >= C.SPLIT_THRESHOLD_CNY * C.DODGE_RATIO) & (cny < C.SPLIT_THRESHOLD_CNY)
    hit(dodge_mask.values, "R4", 30)

    # R5 夜间
    night_mask = df["hour"].between(*C.NIGHT_HOURS)
    hit(night_mask.values, "R5", 15)

    # R6 数据质量：抽取失败
    fail_mask = df["ext_amount"].isna() | df["extract_method"].str.contains("failed", na=False)
    hit(fail_mask.values, "R6", 10)

    df["risk_score"] = np.minimum(score, 100).astype(int)
    df["risk_reasons"] = ["；".join(r) for r in reasons]
    df["risk_level"] = df["risk_score"].map(
        lambda s: next(level for cut, level in C.RISK_LEVEL_CUTOFFS if s >= cut))
    return df


def risk_summary(df: pd.DataFrame) -> dict:
    """输出审计报告所需的统计口径。"""
    tx = df[df["record_type"] == "transaction"]
    suspicious_true = tx["label_suspicious"].ne("normal").sum()
    suspicious_found = tx["risk_level"].isin(["HIGH", "MEDIUM"]).sum()
    return {
        "total_records": len(df),
        "total_cny": round(df["cny_amount"].sum(skipna=True), 2),
        "large_count": int((df["cny_amount"] >= C.LARGE_TX_CNY).sum()),
        "level_counts": df["risk_level"].value_counts().to_dict(),
        "rule_hits": {r: int(df["risk_reasons"].str.contains(r, regex=False).sum())
                      for r in RULES_DOC},
        "label_suspicious": int(suspicious_true),
        "flagged_suspicious": int(suspicious_found),
    }
