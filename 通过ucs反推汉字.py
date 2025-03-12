import re
import pandas as pd


def ucs_to_char(ucs):
    if pd.isna(ucs) or not re.match(r"[0-9A-F]{4,}", str(ucs)):
        return None
    try:
        return print(chr(int(ucs, 16)))  # 将十六进制 UCS 转换为字符
    except ValueError:
        return None


ucs = "2B7E6"
ucs_to_char(ucs)
