# -*- coding: utf-8 -*-
"""
大湾区跨境交易金额智能提取与异常资金识别系统 —— 主程序
=======================================================
业务闭环：
  原始非结构化文本（交易流水附言 / SWIFT 摘要 / 合同条款）
    → LLM 金额抽取（Prompt + JSON Schema + 自愈重试，正则 baseline 降级）
    → 币种归一化（7 币种别名归一 + 汇率折算 CNY）
    → 异常资金识别（可解释规则引擎：大额 / 结构化拆分 / 高风险地区 / 临界规避 / 夜间）
    → 字段级自动评估（vs ground truth，含 LLM vs 正则 baseline 对比）
    → 可视化看板 + Markdown 审计报告 + 结构化结果 CSV

用法：
  python main.py                 # LLM 模式（需配置 API Key，见 .env.example）
  python main.py --mode mock     # 离线正则 baseline，无需 Key，全流程可跑
  python main.py --regen --n 200 # 重新生成模拟数据（200 笔正常交易）
"""
import argparse
import sys
from datetime import datetime

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
import config as C
from utils import data_generator, llm_extractor, normalizer, risk_rules, evaluator, visualizer


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def load_or_generate(regen: bool = False, n_normal: int = 140) -> pd.DataFrame:
    tx_path, ct_path = C.RAW_DIR / "transactions.csv", C.RAW_DIR / "contracts.jsonl"
    if regen or not tx_path.exists():
        data_generator.main()
    tx = pd.read_csv(tx_path, encoding="utf-8-sig")
    ct = pd.read_json(ct_path, lines=True)
    df = pd.concat([tx, ct], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# 归一化：币种别名归一 + 折算 CNY
# ---------------------------------------------------------------------------
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ext_currency_std"] = df["ext_currency"].map(
        lambda x: normalizer.normalize_currency(str(x)) if pd.notna(x) else None)
    cny, notes = [], []
    for _, r in df.iterrows():
        v, note = normalizer.to_cny(r["ext_amount"], r["ext_currency_std"])
        cny.append(v)
        notes.append(note)
    df["cny_amount"] = cny
    df["norm_note"] = notes
    return df


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------
def df_to_md(df: pd.DataFrame, float_fmt: callable = None) -> str:
    if df.empty:
        return "（无数据）"
    cols = list(df.columns)

    def cell(v):
        if pd.isna(v):
            return "-"
        return float_fmt(v) if float_fmt and isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in r) + " |")
    return "\n".join(lines)


def build_report(df, summary, eval_results, mode, elapsed) -> str:
    pct = lambda v: f"{v * 100:.1f}%"
    top10 = df.nlargest(10, "risk_score")[
        ["record_id", "payer", "region_dst", "ext_amount", "ext_currency",
         "cny_amount", "risk_score", "risk_level", "risk_reasons"]]
    top10 = top10.copy()
    top10["cny_amount"] = top10["cny_amount"].map(lambda v: f"{v:,.0f}")
    top10["ext_amount"] = top10["ext_amount"].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "-")
    top10.columns = ["记录ID", "付款方", "收款地区", "抽取金额", "币种", "等值CNY", "风险分", "等级", "命中规则"]

    cmp_df = evaluator.compare_table(eval_results).rename(columns={
        "extract_success_rate": "抽取成功率", "amount_acc": "金额准确率", "currency_acc": "币种准确率"})
    for col in ["抽取成功率", "金额准确率", "币种准确率"]:
        if col in cmp_df:
            cmp_df[col] = cmp_df[col].map(lambda v: pct(v) if pd.notna(v) else "-")

    style_df = evaluator.style_table(eval_results).rename(columns={
        "label_style": "金额书写形态", "n": "样本数", "amount_acc": "金额准确率", "currency_acc": "币种准确率"})
    for col in ["金额准确率", "币种准确率"]:
        style_df[col] = style_df[col].map(lambda v: pct(v) if pd.notna(v) else "-")

    lv = summary["level_counts"]
    return f"""# 跨境交易风控审计报告

> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S} ｜ 抽取通道：{mode} ｜ 模型：{C.LLM_MODEL} ｜ 耗时 {elapsed:.1f}s

## 一、总体统计

| 指标 | 数值 |
|---|---|
| 处理记录数 | {summary['total_records']}（含交易流水 + 合同文本） |
| 资金总量（等值CNY） | {summary['total_cny']:,.0f} 元 |
| 大额交易（≥等值{C.LARGE_TX_CNY / 10000:.0f}万） | {summary['large_count']} 笔 |
| 风险等级分布 | HIGH {lv.get('HIGH', 0)} ｜ MEDIUM {lv.get('MEDIUM', 0)} ｜ LOW {lv.get('LOW', 0)} |
| 模拟数据埋入可疑样本 | {summary['label_suspicious']} 笔（规则标记 MEDIUM 以上 {summary['flagged_suspicious']} 笔） |

## 二、命中风控规则统计

| 规则 | 说明 | 命中笔数 |
|---|---|---|
""" + "\n".join(
        f"| {r} | {name} | {summary['rule_hits'][r]} |"
        for r, name in risk_rules.RULES_DOC.items()) + f"""

## 三、TOP 10 高风险记录

{df_to_md(top10)}

## 四、抽取效果评估（字段级，vs 模拟数据 ground truth）

**通道对比：**

{df_to_md(cmp_df)}

**按金额书写形态分组（定位模型短板）：**

{df_to_md(style_df)}

## 五、说明

- 本报告由 `main.py` 自动生成，数据为模拟数据，阈值为演示口径（见 `config.py`）。
- 风险分与命中规则一一对应，支持逐笔回溯审计；HIGH 级建议推送人工复核队列。
"""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="大湾区跨境交易金额智能提取与异常资金识别")
    parser.add_argument("--mode", choices=["llm", "mock"], default="llm",
                        help="llm=大模型抽取（默认）；mock=正则 baseline 离线运行")
    parser.add_argument("--regen", action="store_true", help="重新生成模拟数据")
    parser.add_argument("--n", type=int, default=140, help="模拟数据中正常交易笔数")
    parser.add_argument("--no-viz", action="store_true", help="跳过可视化")
    args = parser.parse_args()

    t0 = datetime.now()
    print("=" * 62)
    print("大湾区跨境交易金额智能提取与异常资金识别系统")
    print("=" * 62)

    # 1) 数据
    print("\n[1/6] 加载模拟数据 ...")
    df = load_or_generate(args.regen, args.n)
    print(f"  共 {len(df)} 条记录（交易流水 {sum(df['record_type'] == 'transaction')}"
          f" + 合同 {sum(df['record_type'] == 'contract')}）")

    # 2) 抽取
    print(f"\n[2/6] 金额抽取（通道：{args.mode}）...")
    df = llm_extractor.extract_batch(df, mode=args.mode)
    ok = df["ext_amount"].notna().sum()
    print(f"  抽取成功 {ok}/{len(df)}（{ok / len(df):.1%}）")

    # 3) 币种归一化
    print("\n[3/6] 币种归一化 + 汇率折算 CNY ...")
    df = normalize(df)
    print(f"  已折算 {df['cny_amount'].notna().sum()} 条，币种未识别 {df['ext_currency_std'].isna().sum()} 条")

    # 4) 风控规则引擎
    print("\n[4/6] 异常资金识别（规则引擎）...")
    df = risk_rules.apply_rules(df)
    summary = risk_rules.risk_summary(df)
    lv = summary["level_counts"]
    print(f"  HIGH {lv.get('HIGH', 0)} ｜ MEDIUM {lv.get('MEDIUM', 0)} ｜ LOW {lv.get('LOW', 0)}")

    # 5) 评估闭环（LLM 模式下同时评估正则 baseline，形成对比）
    print("\n[5/6] 字段级抽取评估 ...")
    eval_results = [evaluator.evaluate(df, method_name="llm" if args.mode == "llm" else "regex-baseline")]
    if args.mode == "llm":
        base_df = llm_extractor.extract_batch(df[["text"]].copy(), mode="mock")
        base_df["label_amount"] = df["label_amount"].values
        base_df["label_currency"] = df["label_currency"].values
        base_df["label_style"] = df["label_style"].values
        eval_results.append(evaluator.evaluate(base_df, method_name="regex-baseline"))
    print(evaluator.compare_table(eval_results).to_string(index=False))

    # 6) 输出
    print("\n[6/6] 产出结果 ...")
    if not args.no_viz:
        visualizer.run_all(df)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = C.PROCESSED_DIR / f"extracted_{'llm' if args.mode == 'llm' else 'mock'}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  结构化结果 -> {out_csv}")

    elapsed = (datetime.now() - t0).total_seconds()
    report = build_report(df, summary, eval_results, args.mode, elapsed)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUTPUT_DIR / "risk_report.md").write_text(report, encoding="utf-8")
    print(f"  审计报告 -> {C.OUTPUT_DIR / 'risk_report.md'}")

    print(f"\n完成，总耗时 {elapsed:.1f}s。HIGH 风险记录建议进入人工复核队列。")


if __name__ == "__main__":
    main()
