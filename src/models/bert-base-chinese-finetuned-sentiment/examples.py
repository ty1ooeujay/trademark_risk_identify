
from transformers import pipeline

# 載入情感分析管道
classifier = pipeline(
    "sentiment-analysis",
    model="jackietung/bert-base-chinese-sentiment-finetuned",
    return_all_scores=True
)

# 測試文本
texts = [
    "這款 App 的界面設計非常直觀，使用起來很順暢！",
    "客服回應速度太慢，問題遲遲得不到解決，很失望。",
    "功能還算齊全，但偶爾會閃退，希望能改進。",
    "雖然有些小bug，但整體來說是個實用的工具App。",
    "完全不推薦下載，廣告太多而且耗電量驚人。"
]

# 進行預測
for text in texts:
    result = classifier(text)[0]
    print(f"文本: {text}")
    
    # 按分數排序
    sorted_scores = sorted(result, key=lambda x: x['score'], reverse=True)
    
    # 獲取最高分數的情感
    top_sentiment = sorted_scores[0]
    print(f"預測情感: {top_sentiment['label']} (分數: {top_sentiment['score']:.4f})")
    
    # 顯示所有情感分數
    print("所有情感分數:")
    for score_item in sorted_scores:
        print(f"  {score_item['label']}: {score_item['score']:.4f}")
    print("-" * 50)
