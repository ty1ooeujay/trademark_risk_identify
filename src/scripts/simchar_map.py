import json
from skimage.metrics import structural_similarity as ssim
from itertools import combinations
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import math

class GB2312:
    """可迭代产生GB2312中字符"""
    level1 = range(0xB0, 0xD7+1), range(0xA1, 0xFE+1)  # 一级汉字区
    level2 = range(0xD8, 0xF7+1), range(0xA1, 0xFE+1)  # 二级汉字区
    allchar = range(0xB0, 0xF7+1), range(0xA1, 0xFE+1)  # 全部汉字区
    
    def __init__(self, charType=allchar):
        self.charType = charType
        self.badCode = 0

    def stfOfGB2312(self, h, l):
        """由GB2312区位编码得到字符"""
        return bytes([h, l]).decode('GB2312')
        
    def __iter__(self):
        """产生遍历charType的迭代器"""
        for hiCode in self.charType[0]:
            for loCode in self.charType[1]:
                try:
                    ch = self.stfOfGB2312(hiCode, loCode)
                except:
                    self.badCode += 1
                    continue
                else:
                    yield ch

class SimComputer:
    HISTOGRAM = 0


class SimCharMap:
    def __init__(self, char_type=None):
        """
        初始化汉字点位图生成器
        
        Args:
            char_filepath: 字符文件路径（可选，保留用于未来扩展）
            font_size: 字体大小，默认64
            image_size: 图像尺寸，默认为(font_size, font_size)
            font_path: 字体文件路径，如果为None则使用系统默认字体
        """
        char_filepath = "data/common/gsc.txt"
        if char_type is None:
            try:
                with open(char_filepath, 'r', encoding='utf-8') as file:
                    self.chars = file.readlines()
                self.chars = [char.strip() for char in self.chars]
            except Exception as e:
                print(f"读取字符文件失败: {e}")
                self.chars = []
        else:
            self.chars = list(GB2312(char_type))

        self.char_images = {}
        for char in self.chars:
            self.char_images[char] = self.get_char_pic(char)

    def get_char_pic(self, char, font_size=64):
        """
        获取汉字对应的点位图
        Args:
            char: 单个汉字字符,
            font_size: 字体大小,
        Returns:
            numpy.ndarray: 点位图
        """
        char = char[0]
        font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", font_size)
        img_pil = Image.new('RGB', (font_size, font_size), color='white')
        draw = ImageDraw.Draw(img_pil)
        draw.text((0, 0), char, font=font, fill=(0, 0, 0))
        bitmap = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return bitmap

    def compute_similarity(self, model_type=None, isblur=True):
        total = int(len(self.chars) * (len(self.chars) - 1) / 2)
        print(f"开始计算相似度，共有 {total} 个字符对")
        simchar_pairs = [] 
        for idx, (char1, char2) in enumerate(combinations(self.chars, 2)):
            # init
            is_sim = False
            similarity = 0.0

            # start
            img1 = self.char_images[char1]
            img2 = self.char_images[char2]
            # 使用高斯模糊处理
            if isblur:
                img1 = cv2.GaussianBlur(img1, (7,7), 0)
                img2 = cv2.GaussianBlur(img2, (7,7), 0)
            # 使用直方图相似度计算
            if model_type == SimComputer.HISTOGRAM:
                is_sim, similarity = self.histogram_comparison(img1, img2)

            # 如果相似，则添加到simchar_pairs和results中
            if is_sim:
                simchar_pair = {
                    "char1": char1,
                    "char2": char2,
                    "similarity": similarity,
                }
                simchar_pairs.append(simchar_pair)
            if idx % 1000 == 0:
                print(f"已计算 {idx} / {total} 个字符对")
        return simchar_pairs

    def histogram_comparison(self, img1, img2, threshold=0.3):
        # 直方图相似度
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        result = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        is_sim = result < threshold
        
        return is_sim, result

    def save(self, filepath, simchar_pairs):
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(simchar_pairs, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    scm = SimCharMap()
    simchar_pairs = scm.compute_similarity(model_type=SimComputer.HISTOGRAM)
    scm.save("simchar_pairs.json", simchar_pairs)