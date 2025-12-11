import os
import json
import random
import pandas as pd
from pypinyin import lazy_pinyin, Style
from hanzi import Hanzi
hanzi = Hanzi()

from utils.filepath_manager import FPM

class SimapGenerator:
    """相似商标对生成器"""
    def __init__(self):
        pass

    def nosim_map(self, num:int=2000):
        input_filepath1 = os.path.join(self.common_data_path,"trademarks_c.txt")
        input_filepath = os.path.join(self.common_data_path, "trademarks_f.txt")
        output_filepath = os.path.join(self.common_data_path, "trademarks_nosim_pairs.json")

        # 读取中文商标和海外音译商标
        try:
            with open(FPM.CT,"r", encoding='utf-8') as file1:
                trademarks_c = file1.readlines()
            with open(FPM.FT,"r", encoding='utf-8') as file2:
                trademarks_f = file2.readlines()
        except FileNotFoundError:
            print("文件不存在")
            return None

        trademarks = trademarks_c + trademarks_f
        random.shuffle(trademarks)
        pairs = []
        for i in range(num):
            trademark1 = trademarks[random.randint(0, len(trademarks)-1)].strip()
            trademark2 = trademarks[random.randint(0, len(trademarks)-1)].strip()
            if trademark1 != trademark2:
                pair = {
                    "trademark1": trademark1,
                    "trademark2": trademark2,
                    "label": 2
                }
                pairs.append(pair)
        with open(output_filepath,"w", encoding='utf-8') as file:
            json.dump(FPM.SNTP, file, ensure_ascii=False, indent=2)

    def get_sentence(self, name):
        name_pinyin = '-'.join(lazy_pinyin(name, style=Style.TONE3, neutral_tone_with_five=True))
        name_stroke = []
        for char in name:
            char_strokes = hanzi.get_bihua(char)
            if char_strokes is None:
                char_strokes = "[UNK]"
            else:
                char_strokes = char_strokes[0]
            char_strokes = char_strokes.split(',')
            char_strokes = ''.join(char_strokes)
            name_stroke.append(char_strokes)
        name_stroke = '-'.join(name_stroke)
        return f"商标名称：{name}，拼音：{name_pinyin}，笔画：{name_stroke}"

    def pairs2dataset(self, output_filepath):
        with open(FPM.TP, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        dataset = []
        for pair in pairs:
            new_pair = {
                "sentence1": self.get_sentence(pair["trademark1"]),
                "sentence2": self.get_sentence(pair["trademark2"]),
                "label": pair["label"]
            }
            dataset.append(new_pair)
        df = pd.DataFrame(dataset)
        df.to_csv(output_filepath, index=False, encoding="utf-8")

    def getChinesetrademarks():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        input_file = os.path.join(project_root, 'data', "common", 'trademarks_yin.txt')
        output_file = os.path.join(project_root, 'data', "common", 'trademarks_yin.txt')
        with open(input_file, "r", encoding="utf-8") as f:
            trademarks = f.readlines()
        new_trademarks = []
        for trademark in trademarks:
            new_trademark = trademark.split("-")[-1]
            new_trademarks.append(new_trademark)
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(new_trademarks)

    def replace_trademarks_yin():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        inputfile_trademark = os.path.join(project_root, 'data', "common", 'trademarks_yin.txt')
        inputfile_pinyin = os.path.join(project_root, 'data', "common", 'similar_pinyin.txt')
        outputfile = os.path.join(project_root, 'data', "common", 'similar_trademark_pinyin_pairs.json')
        with open(inputfile_trademark, "r", encoding="utf-8") as f:
            trademarks = f.readlines()

        with open(inputfile_pinyin, "r", encoding="utf-8") as f:
            similar_pinyin = f.readlines()
        similar_pinyin_dict = {}
        for line in similar_pinyin:
            comb = line.split(" ")
            similar_pinyin_dict[comb[0]] = comb[-1][:5].strip()

        similar_pinyin_pairs = []
        for trademark in trademarks:
            trademark = trademark.strip()
            for index, char in enumerate(trademark):
                # 同音字替换
                try:
                    str_tongyin = similar_pinyin_dict[char]
                    for char_tongyin in str_tongyin:
                        trademark2 = list(trademark)
                        trademark2[index] = char_tongyin
                        pair = {
                            "trademark1": trademark,
                            "trademark2": "".join(trademark2),
                            "label":0
                        }
                        similar_pinyin_pairs.append(pair)
                except KeyError:
                    print(f"未找到{char}的同音字")
        similar_pinyin_pairs = similar_pinyin_pairs[::2]
        print(f"相似字符对数量: {len(similar_pinyin_pairs)}")

        with open(outputfile, "w", encoding="utf-8") as f:
            json.dump(similar_pinyin_pairs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    sg = SimapGenerator()
    sg.pairs2dataset("sim_trademark_pairs.csv")