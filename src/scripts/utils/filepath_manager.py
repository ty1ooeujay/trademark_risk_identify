project_name = "trademask_risk"
from pathlib import Path

class FPM:
    """文件路径管理类"""
    PR = PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    CD = COMMON_DATA_PATH = PROJECT_ROOT / "data" / "common"
    # 海外音译商标
    FT = FOREIGN_TRADEMARK_PATH = COMMON_DATA_PATH / "trademarks_f.txt"
    # 中文商标
    CT =CHINESE_TRADEMARK_PATH = COMMON_DATA_PATH / "trademarks_c.txt"
    # 汉字字典
    HD = HANZI_DICT_PATH = CHINESE_TRADEMARK_PATH / "gsc.txt"
    # 相似字形对
    SZ = SIM_ZIXING_PATH = COMMON_DATA_PATH / "similar_chars_pairs.json"
    # 相似拼音对
    SP =SIM_PINYIN_PATH = COMMON_DATA_PATH / "similar_pinyin.txt"
    # 临时输出路径
    TO = TEMP_OUTPUT_PATH = PROJECT_ROOT / "data" / "temp"
    # 相似字形商标对
    SPTP =SIM_PINYIN_TRADEMARK_PAIRS_PATH = TEMP_OUTPUT_PATH / "trademark_sim_zixing_pairs.json"
    # 相似字音商标对
    SZTP = SIM_ZIXING_TRADEMARK_PAIRS_PATH = TEMP_OUTPUT_PATH / "trademark_sim_pinyin_pairs.json"
    # 相似语义商标对
    SMTP = SIM_MEAN_TRADEMARK_PAIRS_PATH = TEMP_OUTPUT_PATH / "trademark_sim_mean_pairs.json"
    # 非相似商标对
    SNTP = SIM_NOT_TRADEMARK_PAIRS_PATH = TEMP_OUTPUT_PATH / "trademark_sim_not_pairs.json"
    # 商标对
    TP = TRADEMARK_PAIRS_PATH = TEMP_OUTPUT_PATH / "trademarks_pairs.json"