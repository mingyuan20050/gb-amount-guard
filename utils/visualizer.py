# -*- coding: utf-8 -*-
"""
可视化模块（Matplotlib）
------------------------
产出 4 张业务图 + 1 张 2x2 汇总看板：
  1. 币种分布         跨境交易币种结构（港币/美元双主力是大湾区特色）
  2. 金额分布         等值 CNY 对数直方图 + 大额阈值线
  3. 风险画像         风险等级分布 + 命中规则 Top N
  4. 每日资金流向时序  按日等值 CNY 总额，标注高风险日
图表风格统一（中文字体 / 配色），直接可用于风控周报。
"""
import matplotlib
matplotlib.use("Agg")  # 服务器/CI 无显示环境也能出图
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import config as C

# 中文字体：Windows 优先微软雅黑，兜底黑体/macOS
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

C_MAIN, C_ACC, C_WARN, C_DANGER, C_MUTED = "#1f4e79", "#2e9bd6", "#e8a33d", "#c0392b", "#8a8f98"
LEVEL_COLORS = {"HIGH": C_DANGER, "MEDIUM": C_WARN, "LOW": "#7fb069"}


def _save(fig, name):
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {C.OUTPUT_DIR / name}")


def plot_currency_distribution(df):
    vc = df["ext_currency"].fillna("未识别").value_counts()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(vc.index, vc.values, color=C_MAIN, width=0.62)
    bars[int(vc.idxmax() == vc.index[0]) * 0].set_color(C_ACC)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vc.values) * 0.02,
                f"{v}", ha="center", fontsize=9)
    ax.set_title("跨境交易币种分布（抽取+归一化后）", fontsize=12, weight="bold")
    ax.set_xlabel("币种"); ax.set_ylabel("笔数")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "currency_distribution.png")


def plot_amount_distribution(df):
    s = df["cny_amount"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(s[s > 0], bins=40, color=C_ACC, edgecolor="white", log=True)
    ax.axvline(C.LARGE_TX_CNY, color=C_DANGER, ls="--", lw=1.4)
    ax.text(C.LARGE_TX_CNY, ax.get_ylim()[1] * 0.7,
            f" 大额阈值 = {C.LARGE_TX_CNY / 10000:.0f}万 CNY", color=C_DANGER, fontsize=9)
    ax.set_title("单笔交易金额分布（等值CNY，对数轴）", fontsize=12, weight="bold")
    ax.set_xlabel("等值人民币金额（元）"); ax.set_ylabel("笔数（对数）")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "amount_distribution.png")


def plot_risk_profile(df):
    from .risk_rules import RULES_DOC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    levels = ["HIGH", "MEDIUM", "LOW"]
    cnt = df["risk_level"].value_counts().reindex(levels, fill_value=0)
    ax1.bar(cnt.index, cnt.values, color=[LEVEL_COLORS[l] for l in levels], width=0.55)
    for i, v in enumerate(cnt.values):
        ax1.text(i, v + max(cnt.values) * 0.02, str(v), ha="center", fontsize=10)
    ax1.set_title("风险等级分布", fontsize=12, weight="bold")
    ax1.set_xlabel("风险等级"); ax1.set_ylabel("笔数")
    ax1.spines[["top", "right"]].set_visible(False)

    hits = {RULES_DOC[r]: int(df["risk_reasons"].str.contains(r, regex=False).sum())
            for r in RULES_DOC}
    hits = {k: v for k, v in sorted(hits.items(), key=lambda x: -x[1]) if v > 0}
    ax2.barh(list(hits.keys())[::-1], list(hits.values())[::-1], color=C_WARN, height=0.55)
    for i, v in enumerate(list(hits.values())[::-1]):
        ax2.text(v + max(hits.values()) * 0.02, i, str(v), va="center", fontsize=9)
    ax2.set_title("命中风控规则统计", fontsize=12, weight="bold")
    ax2.set_xlabel("命中笔数")
    ax2.tick_params(axis="y", labelsize=8.5)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "risk_profile.png")


def plot_daily_trend(df):
    tx = df[df["record_type"] == "transaction"].dropna(subset=["cny_amount"]).copy()
    tx["date"] = pd.to_datetime(tx["ts"]).dt.date
    daily = tx.groupby("date")["cny_amount"].sum() / 10000
    high_days = tx[tx["risk_level"] == "HIGH"].groupby("date")["cny_amount"].sum() / 10000
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(daily.index, daily.values, color=C_MAIN, lw=1.6, label="每日交易总额")
    if len(high_days):
        hv = high_days.reindex(daily.index).fillna(0)
        ax.scatter(list(daily.index), hv.values,
                   color=C_DANGER, s=26, zorder=3, label="含高风险交易的日期")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.set_title("每日跨境资金流（等值万元 CNY）", fontsize=12, weight="bold")
    ax.set_xlabel("日期"); ax.set_ylabel("金额（万元）")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    _save(fig, "daily_trend.png")


def plot_dashboard(df):
    """2x2 汇总看板：一张图进周报。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("大湾区跨境交易金额智能提取与异常资金识别 —— 风控看板", fontsize=14, weight="bold")

    vc = df["ext_currency"].fillna("未识别").value_counts()
    axes[0, 0].bar(vc.index, vc.values, color=C_MAIN, width=0.6)
    axes[0, 0].set_title("币种分布", fontsize=11, weight="bold")

    s = df["cny_amount"].dropna()
    axes[0, 1].hist(s[s > 0], bins=40, color=C_ACC, edgecolor="white", log=True)
    axes[0, 1].axvline(C.LARGE_TX_CNY, color=C_DANGER, ls="--", lw=1.4)
    axes[0, 1].set_title("金额分布（等值CNY，对数轴）", fontsize=11, weight="bold")

    cnt = df["risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
    axes[1, 0].bar(cnt.index, cnt.values, color=[LEVEL_COLORS[l] for l in cnt.index], width=0.55)
    axes[1, 0].set_title("风险等级分布", fontsize=11, weight="bold")

    top = df.nlargest(10, "risk_score")
    axes[1, 1].barh(range(len(top)), top["risk_score"], color=C_DANGER, height=0.6)
    axes[1, 1].set_yticks(range(len(top)))
    axes[1, 1].set_yticklabels([f"{r['record_id']} {r['payer'][:10]}" for _, r in top.iterrows()], fontsize=8)
    axes[1, 1].set_title("TOP10 风险评分", fontsize=11, weight="bold")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "dashboard.png")


def run_all(df):
    plot_currency_distribution(df)
    plot_amount_distribution(df)
    plot_risk_profile(df)
    plot_daily_trend(df)
    plot_dashboard(df)
