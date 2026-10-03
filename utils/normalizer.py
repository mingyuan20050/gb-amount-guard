# -*- coding: utf-8 -*-
"""
币种归一化模块
--------------
1. parse_cn_upper：中文大写金额 -> 数值（"贰拾伍万陆仟元整" -> 256000）
2. normalize_currency：币种别名归一（"港币"/"HK$" -> HKD）
3. to_cny：按演示汇率折算 CNY，异常币种返回 None 并记录原因（可解释）
"""
import config as C

_CN_DIGIT = {c: i for i, c in enumerate("零壹贰叁肆伍陆柒捌玖")}
_CN_SECTION = {c: v for c, v in zip("拾佰仟", (10, 100, 1000))}
_CN_BIG = {"万": 10**4, "亿": 10**8}


def parse_cn_upper(text: str):
    """解析中文大写金额，失败返回 None。支持 万/亿 分节 + 角/分。"""
    if not text or not any(c in _CN_DIGIT for c in text if c in "壹贰叁肆伍陆柒捌玖"):
        return None
    core = text.replace("人民币", "").replace("元整", "").replace("整", "")
    jiao = fen = 0
    for i, ch in enumerate(core):
        if ch == "角":
            jiao = _CN_DIGIT.get(core[i - 1], 0)
        if ch == "分":
            fen = _CN_DIGIT.get(core[i - 1], 0)
    if "角" in core or "分" in core:
        core = core.split("角")[0].split("分")[0] + "元"
    core = core.split("元")[0]

    total, section, current = 0, 0, 0
    met_digit = False
    for ch in core:
        if ch in _CN_DIGIT:
            current = _CN_DIGIT[ch]
            met_digit = True
        elif ch in _CN_SECTION:
            current = (current or 1) * _CN_SECTION[ch]
            section += current
            current = 0
            met_digit = False
        elif ch in _CN_BIG:
            if not met_digit and current == 0 and section == 0:
                section = 1
            section = (section + current) or 1
            total += section * _CN_BIG[ch]
            section, current, met_digit = 0, 0, False
    total += section + current
    if total == 0 and not met_digit:
        return None
    return round(total + jiao / 10 + fen / 100, 2)


def normalize_currency(token: str):
    """币种别名 -> 标准代码；无法识别返回 None。"""
    if not token:
        return None
    token = token.strip()
    low = token.lower()
    for alias, code in C.CURRENCY_ALIASES.items():
        if low == alias.lower():
            return code
    return None


def to_cny(amount, currency, fx=None):
    """折算 CNY。返回 (cny_value, note)；汇率缺失时 note 说明原因。"""
    fx = fx or C.FX_RATES
    if amount is None or currency is None:
        return None, "金额或币种缺失"
    if currency not in fx:
        return None, f"未配置汇率: {currency}"
    return round(float(amount) * fx[currency], 2), ""
