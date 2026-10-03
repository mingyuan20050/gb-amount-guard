# -*- coding: utf-8 -*-
"""
模拟数据生成器（大湾区跨境场景）
--------------------------------
生成两类"原始非结构化数据"并自带 ground truth 标注（label_* 列），
用于：1) 模拟真实业务的脏数据形态  2) 抽取效果的自动化评估闭环。

刻意覆盖 5 类金额书写形态（评估时按形态分组统计，看 LLM 到底强在哪）：
  comma    千分位数字：      支付货款 HK$1,234,567.89
  cn_upper 中文大写：        人民币贰拾伍万元整
  wan      万为单位：        500万港元
  symbol   符号缩写：        US$1.2M
  multi    多金额干扰：      定金10万港元，尾款90万港元（取总额为标签）

同时埋入 4 类可疑模式：结构化拆分(smurfing)、大额、高风险地区、临界规避+夜间。
"""
import csv
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

CN_DIGIT = "零壹贰叁肆伍陆柒捌玖"
CN_UNIT = {10: "拾", 100: "佰", 1000: "仟"}
CN_BIG_UNIT = [(10**8, "亿"), (10**4, "万")]

# 虚构主体池（避免与任何真实企业重名）
BAY_AREA_COMPANIES = [
    "深圳市鹏城芯半导体有限公司", "广州云山生物医药有限公司", "珠海伶仃海洋装备有限公司",
    "东莞松湖精密制造有限公司", "佛山顺德智能家电有限公司", "惠州大亚湾新材有限公司",
    "中山火炬光电有限公司", "江门五邑食品供应链有限公司", "肇庆新区铝业有限公司",
]
HK_COMPANIES = ["香港维港国际贸易有限公司", "香港中环资本管理有限公司", "香港科学园生物科技有限公司"]
MACAO_COMPANIES = ["澳门濠江会展服务有限公司", "澳门路环食品进出口有限公司"]
OFFSHORE_COMPANIES = ["开曼群岛环球控股有限公司", "巴拿马远洋航运有限公司",
                      "英属维尔京群岛珍珠投资有限公司", "塞舌尔蓝海投资有限公司"]
BAY_REGIONS = ["深圳", "广州", "珠海", "东莞", "佛山", "惠州", "中山", "香港", "澳门"]
CHANNELS = ["SWIFT跨境汇款", "跨境人民币结算CIPS", "港元RTGS", "银联跨境通道", "西联汇款"]
PURPOSES = ["货款", "服务费", "技术授权费", "仓储物流费", "咨询费", "设备采购款", "会展服务费"]
AMOUNT_STYLES = ["comma", "cn_upper", "wan", "symbol", "multi"]


# ---------------------------------------------------------------------------
# 工具：数字 <-> 中文大写
# ---------------------------------------------------------------------------
def _four_to_cn(x: int) -> str:
    """0~9999 整数转中文大写（含组内补零，如 1001 -> 壹仟零壹）。"""
    s, need_zero = "", False
    for u, name in [(1000, "仟"), (100, "佰"), (10, "拾")]:
        d, x = divmod(x, u)
        if d:
            s += CN_DIGIT[d] + name
            need_zero = False
        elif s:
            need_zero = True
    if x:
        s += ("零" if s and need_zero else "") + CN_DIGIT[x]
    return s


def to_cn_upper(amount: float) -> str:
    """数值转中文大写金额（生成器与 baseline 抽取共用，保证口径一致）。"""
    yuan = int(amount)
    cents = round((amount - yuan) * 100)
    parts = []
    yi, rest = divmod(yuan, 10**8)
    if yi:
        parts.append(_four_to_cn(yi) + "亿")
    wan, ge = divmod(rest, 10**4)
    if wan:
        parts.append(_four_to_cn(wan) + "万")
    if ge:
        parts.append(("零" if parts and ge < 1000 else "") + _four_to_cn(ge))
    s = ("".join(parts) or "零") + "元"
    if cents:
        jiao, fen = divmod(cents, 10)
        if jiao:
            s += CN_DIGIT[jiao] + "角"
        if fen:
            s += CN_DIGIT[fen] + "分"
    else:
        s += "整"
    return s


SYMBOL_MAP = {"HKD": "HK$", "USD": "US$", "EUR": "€", "GBP": "£", "JPY": "JP¥", "MOP": "MOP$", "CNY": "￥"}
CN_NAME_MAP = {"HKD": "港元", "USD": "美元", "EUR": "欧元", "GBP": "英镑", "JPY": "日元", "MOP": "澳门元", "CNY": "人民币"}


def fmt_comma(amount: float, currency: str) -> str:
    return f"{SYMBOL_MAP[currency]}{amount:,.2f}"


def fmt_wan(amount: float, currency: str) -> str:
    wan = amount / 10000
    txt = f"{wan:.1f}".rstrip("0").rstrip(".")
    return f"{txt}万{CN_NAME_MAP[currency]}"


def fmt_symbol(amount: float, currency: str) -> str:
    """缩写形态：US$1.2M / HK$3.5M。"""
    m = amount / 1e6
    txt = f"{m:.2f}".rstrip("0").rstrip(".")
    return f"{SYMBOL_MAP[currency]}{txt}M"


def render_amount(amount: float, currency: str, style: str, purpose: str) -> str:
    """按书写形态渲染金额片段（multi 形态返回含定金/尾款/合计的完整句）。"""
    if style == "comma":
        return f"{purpose}{fmt_comma(amount, currency)}"
    if style == "cn_upper":
        return f"{CN_NAME_MAP[currency]}（大写）{to_cn_upper(amount)}"
    if style == "wan":
        return f"{purpose}{fmt_wan(amount, currency)}"
    if style == "symbol":
        return f"{purpose}{fmt_symbol(amount, currency)}"
    # multi：定金 20% + 尾款 80%，干扰项
    deposit = round(amount * 0.2 / 100) * 100
    rest = amount - deposit
    return (f"合同总金额{fmt_comma(amount, currency)}，其中定金{fmt_wan(deposit, currency)}，"
            f"尾款{fmt_wan(rest, currency)}，本笔支付合计{fmt_comma(amount, currency)}")


TEXT_TEMPLATES = [
    "{date} {payer} 通过{channel} 向 {payee} 支付{amount_frag}，用途：{purpose}，附言：{note}。",
    "交易流水号 TXN{seq}：付款方{payer}（{region_src}），收款方{payee}（{region_dst}），金额{amount_frag}，渠道：{channel}。",
    "{payer} 提交跨境付款申请：收款人 {payee}，地区 {region_dst}，{amount_frag}，交易用途注明为{purpose}。备注 {note}。",
    "SWIFT MT103 报文摘要：ORD/{seq}，32A 金额 {amount_frag}，50K {payer}，59 {payee}，71A 费用共同承担。{note}",
]
NOTES = ["请尽快入账", "二期项目款项", "按合同第3.2条结算", "分批交割", "本月对账完毕", ""]


def make_tx(idx, ts, payer, payee, region_src, region_dst, channel,
            amount, currency, style, suspicious, note=""):
    """组装一条交易记录：text 为系统输入，label_* 为隐藏 ground truth。"""
    # M 缩写只有两位小数精度（0.01M = 1万）：symbol 形态先把金额量化到万，
    # 保证"文本可表达的金额"与 ground truth 一致（否则评估会把读对文本的
    # 模型误判为错——模拟数据的标签必须与文本自洽）
    if style == "symbol":
        amount = round(amount / 1e4) * 1e4
    purpose = random.choice(PURPOSES)
    amount_frag = render_amount(amount, currency, style, purpose)
    tpl = random.choice(TEXT_TEMPLATES)
    text = tpl.format(date=ts.strftime("%Y-%m-%d"), payer=payer, payee=payee,
                      region_src=region_src, region_dst=region_dst,
                      amount_frag=amount_frag, channel=channel,
                      purpose=purpose, note=note or random.choice(NOTES),
                      seq=f"{20260000 + idx}")
    return {
        "record_id": f"TX{idx:05d}",
        "record_type": "transaction",
        "ts": ts.isoformat(timespec="seconds"),
        "payer": payer, "payee": payee,
        "region_src": region_src, "region_dst": region_dst,
        "channel": channel,
        "text": text,
        # ---- ground truth（仅供评估，不参与抽取）----
        "label_amount": round(amount, 2),
        "label_currency": currency,
        "label_style": style,
        "label_suspicious": suspicious,
    }


def pick_currency(weights=None):
    currencies = ["CNY", "HKD", "HKD", "USD", "USD", "EUR", "MOP", "GBP", "JPY"]
    return random.choice(weights or currencies)


def generate_transactions(n_normal=140):
    """生成交易流水：正常 + 大额 + 结构化拆分 + 高风险地区 + 临界规避/夜间。"""
    random.seed(C.RANDOM_SEED)
    rows, idx = [], 0
    base = datetime(2026, 7, 1, 9, 0, 0)

    def rand_ts(days=90, hour=None):
        h = hour if hour is not None else random.randint(8, 20)
        return base + timedelta(days=random.randint(0, days),
                                hours=h - 9, minutes=random.randint(0, 59))

    # 1) 正常交易：等值 0.5万~80万 CNY
    for _ in range(n_normal):
        cur = pick_currency()
        cny = random.uniform(5_000, 800_000)
        rows.append(make_tx(idx, rand_ts(), random.choice(BAY_AREA_COMPANIES),
                            random.choice(HK_COMPANIES + MACAO_COMPANIES + BAY_AREA_COMPANIES),
                            random.choice(BAY_REGIONS), random.choice(BAY_REGIONS),
                            random.choice(CHANNELS), cny / C.FX_RATES[cur], cur,
                            random.choice(AMOUNT_STYLES), "normal"))
        idx += 1

    # 2) 大额交易：等值 100万~1800万 CNY
    for _ in range(22):
        cur = pick_currency(["CNY", "HKD", "USD", "USD", "EUR"])
        cny = random.uniform(C.LARGE_TX_CNY, 18_000_000)
        rows.append(make_tx(idx, rand_ts(), random.choice(BAY_AREA_COMPANIES),
                            random.choice(HK_COMPANIES + MACAO_COMPANIES),
                            random.choice(BAY_REGIONS), random.choice(["香港", "澳门", "深圳"]),
                            random.choice(CHANNELS), cny / C.FX_RATES[cur], cur,
                            random.choice(AMOUNT_STYLES), "large"))
        idx += 1

    # 3) 结构化拆分（smurfing）：同一付款人同日多笔、单笔压在申报线下、累计远超
    for g in range(3):
        payer = random.choice(BAY_AREA_COMPANIES)
        day = base + timedelta(days=random.randint(10, 80))
        for k in range(random.randint(4, 6)):
            cur = "HKD"
            cny = random.uniform(0.86, 0.97) * C.SPLIT_THRESHOLD_CNY
            rows.append(make_tx(idx, day + timedelta(hours=random.randint(1, 8) + k),
                                payer, random.choice(HK_COMPANIES),
                                random.choice(["深圳", "广州", "珠海"]), "香港",
                                "港元RTGS", cny / C.FX_RATES[cur], cur,
                                random.choice(["comma", "wan"]), "smurfing",
                                note="分批支付货款"))
            idx += 1

    # 4) 高风险离岸地区交易
    for _ in range(10):
        cur = pick_currency(["USD", "USD", "EUR", "HKD"])
        cny = random.uniform(80_000, 2_500_000)
        rows.append(make_tx(idx, rand_ts(), random.choice(BAY_AREA_COMPANIES),
                            random.choice(OFFSHORE_COMPANIES),
                            random.choice(BAY_REGIONS), random.choice(C.HIGH_RISK_REGIONS),
                            random.choice(["SWIFT跨境汇款", "西联汇款"]),
                            cny / C.FX_RATES[cur], cur,
                            random.choice(AMOUNT_STYLES),
                            "high_risk" + ("_large" if cny >= C.LARGE_TX_CNY else "")))
        idx += 1

    # 5) 临界规避：单笔落在申报线 90%~100%
    for _ in range(8):
        cur = random.choice(["HKD", "USD"])
        cny = random.uniform(C.DODGE_RATIO, 0.99) * C.SPLIT_THRESHOLD_CNY
        rows.append(make_tx(idx, rand_ts(hour=random.choice([2, 3, 4, 16])),
                            random.choice(BAY_AREA_COMPANIES), random.choice(HK_COMPANIES),
                            random.choice(BAY_REGIONS), "香港", random.choice(CHANNELS),
                            cny / C.FX_RATES[cur], cur,
                            random.choice(["comma", "cn_upper"]),
                            "threshold_dodge" if cur else "night"))
        idx += 1

    rows.sort(key=lambda r: r["ts"])
    for i, r in enumerate(rows):
        r["record_id"] = f"TX{i:05d}"
    return rows


CONTRACT_TEMPLATES = [
    ("《{a}与{b}货物购销合同》摘要：双方约定{a}向{b}采购{goods}，合同总金额{amount_frag}（含税）。"
     "付款方式：签约后 5 个工作日内支付定金{dep}，验收合格后 15 个工作日内支付尾款{rest}。"
     "交货地点：{region}；争议解决适用中华人民共和国法律。"),
    ("《{a}与{b}技术服务合同》第 3 条价款约定：本合同服务费总额为{amount_frag}。"
     "分两期支付：首期 {dep}，第二期 {rest}。服务期限 2026 年度，服务交付地：{region}。"),
    ("《{a}与{b}供应链合作协议》财务条款：合作期内累计结算金额不超过{amount_frag}，"
     "本期实际结算{amount_frag}，其中已付定金{dep}，剩余{rest}于月底前付清。"
     "结算币种为{cur_name}，跨境清算走{channel}。"),
]
GOODS = ["晶圆检测设备 2 台", "生物试剂耗材一批", "跨境冷链仓储服务", "LED 封装生产线 1 条", "会展搭建服务"]


def generate_contracts(n=12):
    """生成合同文本（多金额干扰重灾区：定金/尾款/上限，抽取目标=实际支付总额）。"""
    random.seed(C.RANDOM_SEED + 1)
    rows = []
    for i in range(n):
        a = random.choice(BAY_AREA_COMPANIES)
        b = random.choice(HK_COMPANIES + MACAO_COMPANIES)
        cur = pick_currency()
        cny = random.uniform(200_000, 15_000_000)
        amount = cny / C.FX_RATES[cur]
        style = random.choice(["comma", "cn_upper", "wan", "multi"])
        purpose = random.choice(PURPOSES)
        total_frag = render_amount(amount, cur, style, purpose)
        dep = fmt_wan(round(amount * 0.2 / 100) * 100, cur)
        rest = fmt_wan(amount - round(amount * 0.2 / 100) * 100, cur)
        tpl = CONTRACT_TEMPLATES[i % len(CONTRACT_TEMPLATES)]
        text = tpl.format(a=a, b=b, goods=random.choice(GOODS), amount_frag=total_frag,
                          dep=dep, rest=rest, region=random.choice(BAY_REGIONS),
                          cur_name=CN_NAME_MAP[cur], channel=random.choice(CHANNELS))
        rows.append({
            "record_id": f"CT{i + 1:03d}",
            "record_type": "contract",
            "ts": (datetime(2026, 7, 1) + timedelta(days=random.randint(0, 90))).isoformat(),
            "payer": a, "payee": b, "region_src": "", "region_dst": "", "channel": "",
            "text": text,
            "label_amount": round(amount, 2),
            "label_currency": cur,
            "label_style": style,
            "label_suspicious": "large" if cny >= C.LARGE_TX_CNY else "normal",
        })
    return rows


def save(rows, tx_path, ct_path):
    tx = [r for r in rows if r["record_type"] == "transaction"]
    ct = [r for r in rows if r["record_type"] == "contract"]
    tx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tx_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(tx[0].keys()))
        w.writeheader()
        w.writerows(tx)
    with open(ct_path, "w", encoding="utf-8") as f:
        for r in ct:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(tx), len(ct)


def main():
    rows = generate_transactions() + generate_contracts()
    n_tx, n_ct = save(rows, C.RAW_DIR / "transactions.csv", C.RAW_DIR / "contracts.jsonl")
    print(f"[data_generator] 生成完成：交易流水 {n_tx} 条 -> data/raw/transactions.csv")
    print(f"[data_generator] 生成完成：合同文本 {n_ct} 份 -> data/raw/contracts.jsonl")
    return rows


if __name__ == "__main__":
    main()
