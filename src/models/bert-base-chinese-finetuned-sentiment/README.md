---
language: zh
license: mit
tags:
  - bert
  - sentiment-analysis
  - chinese
  - customer feedback
  - app reviews
datasets:
- custom
metrics:
  - accuracy
  - f1
pipeline_tag: text-classification
widget:
  - text: 商品搜尋體驗很好
  - text: 無法登入會員帳號
  - text: 結帳時系統出錯
base_model:
  - google-bert/bert-base-chinese
library_name: transformers
---

# BERT 中文情感分析模型

這是一個基於 BERT 的中文情感分析模型，可用於判斷文本的情感傾向（正面、負面或中性）。

## 模型描述

- 模型基於 bert-base-chinese 微調
- 適用於App中文評論的情感分析
- 輸出標籤：0（負面），1（正面），2（中性）
- 使用 Focal Loss 訓練，以處理類別不平衡問題

## 使用方法

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 載入模型和分詞器
model = AutoModelForSequenceClassification.from_pretrained("jackietung/bert-base-chinese-sentiment-finetuned")
tokenizer = AutoTokenizer.from_pretrained("jackietung/bert-base-chinese-sentiment-finetuned")

# 準備輸入
text = "這個App使用體驗很差！"
inputs = tokenizer(text, return_tensors="pt")

# 進行預測
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 獲取預測結果
    label_names = ["負面", "正面", "中性"]
    predicted_class = torch.argmax(predictions, dim=1).item()
    
    print(f"預測類別: {label_names[predicted_class]}")
    print(f"預測分數: {predictions[0][predicted_class].item():.4f}")
    
    # 顯示所有類別的分數
    for i, label in enumerate(label_names):
        print(f"{label} 分數: {predictions[0][i].item():.4f}")