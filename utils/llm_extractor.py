# -*- coding: utf-8 -*-
"""
LLM 金额信息抽取模块（核心）
----------------------------
两条抽取通道：
1. extract_llm   —— 大模型 Prompt 抽取（主力）：system prompt 定义输出 JSON Schema +
                   few-shot 示例 + 失败自愈重试（把解析错误回喂模型修正）；
2. extract_regex —— 正则 baseline（降级通道 & 对比实验）：处理千分位/万单位/符号缩写等
                   规则化形态，天然无法稳定处理中文大写与多金额歧义 —— 两条通道的字段级
                   评估对比（见 evaluator.py）恰好量化了"为什么这里值得上 LLM"。

模型接入走 OpenAI 兼容协议（默认 DashScope qwen-plus，环境变量可切任意兼容端点）。
"""
import json
import os
import re
import time

import config as C

# ---------------------------------------------------------------------------
# Prompt 资产
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是银行跨境结算业务的金额信息抽取引擎。任务：从一条交易/合同文本中，抽取【本笔实际支付金额】及其币种，严格输出 JSON。

输出 Schema（只能输出 JSON，不要任何其他文字）：
{
  "amount": 数值或null,          // 换算为最基本计数单位的数值：'500万'->5000000，'US$1.2M'->1200000，'贰拾伍万元整'->250000，千分位去掉逗号
  "currency": "CNY/HKD/USD/EUR/GBP/JPY/MOP" 或 null,
  "amount_text": "原文中的金额片段",
  "currency_basis": "symbol|name|context|default|null",  // 币种判定依据：显式符号(HK$/US$)/币种名(港元)/上下文推断/默认(仅'元'->CNY)
  "has_multiple_amounts": true/false,
  "extraction_notes": "一句话说明歧义处理，无则空字符串"
}

硬性规则：
1. 多金额（定金/尾款/合计/总额）时：取"本笔实际支付/合计/总额"，不得取定金或尾款单项；若只有定金+尾款，取两者之和并在 notes 说明。
2. 中文大写金额必须精确换算（壹贰叁肆伍陆柒捌玖 + 拾佰仟万亿 + 角分）。
3. "万/亿"是数量级不是货币单位：'500万港元'->amount=5000000, currency=HKD。
4. 币种判定优先级：显式符号 > 币种名 > 上下文（如'港元RTGS清算'）；仅写"元/万元"默认 CNY。
5. 文本确实没有可抽取金额时，amount=null、currency=null，不要编造。"""

FEW_SHOTS = [
    {"role": "user", "content": "2026-07-14 深圳市鹏城芯半导体有限公司 通过港元RTGS 向 香港维港国际贸易有限公司 支付货款人民币（大写）贰拾伍万陆仟元整，用途：货款。"},
    {"role": "assistant", "content": '{"amount": 256000.00, "currency": "CNY", "amount_text": "人民币（大写）贰拾伍万陆仟元整", "currency_basis": "name", "has_multiple_amounts": false, "extraction_notes": "中文大写换算：贰拾伍万陆仟=256000"}'},
    {"role": "user", "content": "《广州云山生物医药有限公司与香港中环资本管理有限公司货物购销合同》摘要：合同总金额HK$1,000,000.00（含税）。付款方式：定金20万港元，尾款80万港元，本笔支付合计HK$1,000,000.00。"},
    {"role": "assistant", "content": '{"amount": 1000000.00, "currency": "HKD", "amount_text": "本笔支付合计HK$1,000,000.00", "currency_basis": "symbol", "has_multiple_amounts": true, "extraction_notes": "定金20万+尾款80万为干扰项，取合计100万港元"}'},
    {"role": "user", "content": "珠海伶仃海洋装备有限公司 提交跨境付款申请：收款人 澳门路环食品进出口有限公司，金额85.6万，交易用途注明为仓储物流费。"},
    {"role": "assistant", "content": '{"amount": 856000.00, "currency": "CNY", "amount_text": "85.6万", "currency_basis": "default", "has_multiple_amounts": false, "extraction_notes": "无币种符号与币种名，按规则默认CNY"}'},
]


def build_messages(text: str):
    return [{"role": "system", "content": SYSTEM_PROMPT}, *FEW_SHOTS, {"role": "user", "content": text}]


# ---------------------------------------------------------------------------
# JSON 容错解析
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_llm_json(raw: str):
    """剥掉 markdown 代码围栏后解析 JSON；失败抛 ValueError 由上层触发自愈重试。"""
    m = _FENCE_RE.search(raw)
    payload = m.group(1).strip() if m else raw.strip()
    start, end = payload.find("{"), payload.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("回复中未找到 JSON 对象")
    return json.loads(payload[start:end + 1])


# ---------------------------------------------------------------------------
# 通道一：LLM 抽取（带自愈重试）
# ---------------------------------------------------------------------------
def make_client():
    from openai import OpenAI  # 延迟导入：mock 模式无需安装 openai
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 LLM_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY，"
                           "请配置 .env（参考 .env.example）或使用 --mode mock 离线运行")
    return OpenAI(base_url=os.environ.get("LLM_BASE_URL", C.LLM_BASE_URL), api_key=api_key)


def extract_llm(text: str, client=None, model: str = None):
    """返回 (抽取结果dict, extract_method)。失败自愈最多 C.LLM_MAX_RETRIES 次。"""
    client = client or make_client()
    model = model or os.environ.get("LLM_MODEL", C.LLM_MODEL)
    messages = build_messages(text)
    last_raw, last_err = "", None
    for attempt in range(1 + C.LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=C.LLM_TEMPERATURE, timeout=60)
            last_raw = resp.choices[0].message.content or ""
            result = parse_llm_json(last_raw)
            return {
                "amount": result.get("amount"),
                "currency": result.get("currency"),
                "amount_text": result.get("amount_text", ""),
                "currency_basis": result.get("currency_basis", ""),
                "extraction_notes": result.get("extraction_notes", ""),
            }, "llm"
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            # 自愈：把上一轮原始回复与解析错误回喂模型，要求只修正 JSON 格式
            last_err = f"{type(e).__name__}: {e}"
            messages = messages + [
                {"role": "assistant", "content": last_raw},
                {"role": "user", "content": f"你上一条回复不是合法 JSON（错误：{last_err}）。请严格只输出符合 Schema 的 JSON，不要任何解释。"},
            ]
        except Exception as e:  # 网络/限流等，简单退避重试
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    return {"amount": None, "currency": None, "amount_text": "", "currency_basis": "",
            "extraction_notes": f"llm_failed: {last_err}"}, "llm_failed"


# ---------------------------------------------------------------------------
# 通道二：正则 baseline（mock 模式 & 降级通道）
# ---------------------------------------------------------------------------
_NUM = r"(\d[\d,]*(?:\.\d+)?)"
PATTERNS = [
    # 1) 符号前缀（含 M 缩写）：HK$1,234,567.89 / US$1.2M
    (re.compile(rf"(HK\$|US\$|MOP\$|JP¥|￥|€|£)\s*{_NUM}\s*(M)?(?![\d.])"), "symbol"),
    # 2) 万单位带币种名：500万港元 / 85.6万
    (re.compile(rf"{_NUM}\s*万\s*(港元|港币|美元|欧元|英镑|日元|澳门元|人民币)?"), "wan"),
    # 3) 币种名+千分位数字：港元1,000,000 / 人民币500,000
    (re.compile(rf"(港元|港币|美元|欧元|英镑|日元|澳门元|人民币)\s*{_NUM}"), "name_num"),
    # 4) 兜底：普通数字（千分位）
    (re.compile(rf"{_NUM}\s*(?!万|M\b|%)"), "plain"),
]
SYM2CUR = {"HK$": "HKD", "US$": "USD", "MOP$": "MOP", "JP¥": "JPY", "￥": "CNY", "€": "EUR", "£": "GBP"}
NAME2CUR = {"港元": "HKD", "港币": "HKD", "美元": "USD", "欧元": "EUR", "英镑": "GBP",
            "日元": "JPY", "澳门元": "MOP", "人民币": "CNY"}
_M_UPPER = re.compile(r"[壹贰叁肆伍陆柒捌玖拾佰仟万亿][零壹贰叁肆伍陆柒捌玖拾佰仟万亿]{2,}元[整角分\d]*")


def extract_regex(text: str):
    """规则化形态优先；中文大写/多金额歧义场景是其能力边界（评估可见）。"""
    from .normalizer import parse_cn_upper

    # 多金额场景：优先取"合计/总金额/总额"引导的片段
    for lead in ("合计", "总金额", "总额", "累计结算金额", "本期实际结算"):
        m = re.search(rf"{lead}[^。；,，]*?([\d,.]+\s*万|[\d,]+(?:\.\d+)?)", text)
        if m:
            frag = m.group(1)
            cur = _guess_currency_context(text)
            return _pack(_to_number(frag), cur, frag, "regex_lead")

    for pat, kind in PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        groups = m.groups()
        if kind == "symbol":
            value = float(groups[1].replace(",", ""))
            if groups[2] == "M":
                value *= 1e6
            return _pack(value, SYM2CUR[groups[0]], m.group(0), "regex_symbol")
        if kind == "wan":
            value = float(groups[0].replace(",", "")) * 10000
            cur = NAME2CUR.get(groups[1] or "", None) or _guess_currency_context(text) or "CNY"
            return _pack(value, cur, m.group(0), "regex_wan")
        if kind == "name_num":
            return _pack(float(groups[1].replace(",", "")), NAME2CUR[groups[0]], m.group(0), "regex_name")
        if kind == "plain":
            return _pack(float(groups[0].replace(",", "")), _guess_currency_context(text) or "CNY",
                         m.group(0), "regex_plain")
    # 中文大写兜底（baseline 的弱项：复杂大写/亿级易解析失败）
    m = _M_UPPER.search(text)
    if m:
        val = parse_cn_upper(m.group(0))
        if val:
            cur = "CNY"
            for name, code in NAME2CUR.items():
                if name in text[:text.find(m.group(0))] and code != "CNY":
                    cur = code
                    break
            return _pack(val, cur, m.group(0), "regex_cn_upper")
    return _pack(None, None, "", "regex_failed")


def _guess_currency_context(text: str):
    for name, code in NAME2CUR.items():
        if name in text:
            return code
    return None


def _to_number(frag: str):
    frag = frag.strip().replace(",", "")
    return float(frag[:-1]) * 10000 if frag.endswith("万") else float(frag)


def _pack(amount, currency, frag, method):
    return {"amount": amount, "currency": currency, "amount_text": frag,
            "currency_basis": "", "extraction_notes": ""}, method


# ---------------------------------------------------------------------------
# 批量抽取入口
# ---------------------------------------------------------------------------
def extract_batch(df, mode: str = "llm"):
    """对 DataFrame 逐条抽取。mode=llm：LLM 主通道 + 失败降级正则；mode=mock：纯正则 baseline。"""
    client = make_client() if mode == "llm" else None
    results, methods = [], []
    total = len(df)
    for i, text in enumerate(df["text"], 1):
        if mode == "llm":
            out, method = extract_llm(text, client=client)
            if method == "llm_failed":
                out, fb = extract_regex(text)
                out["extraction_notes"] = (out["extraction_notes"] + " | fallback=regex").strip(" |")
                method = f"llm_failed→{fb}"
            time.sleep(C.LLM_SLEEP_SEC)
        else:
            out, method = extract_regex(text)
        results.append(out)
        methods.append(method)
        if i % 10 == 0 or i == total:
            print(f"  抽取进度 {i}/{total}")
    df = df.copy()
    df["ext_amount"] = [r["amount"] for r in results]
    df["ext_currency"] = [r["currency"] for r in results]
    df["ext_amount_text"] = [r["amount_text"] for r in results]
    df["ext_currency_basis"] = [r["currency_basis"] for r in results]
    df["ext_notes"] = [r["extraction_notes"] for r in results]
    df["extract_method"] = methods
    return df
