from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def main():
    # 載入模型和分詞器
    model = AutoModelForSequenceClassification.from_pretrained("weights/20251213_164526/checkpoint-10100")
    tokenizer = AutoTokenizer.from_pretrained("weights/20251213_164526/checkpoint-10100")

    # 準備輸入
    sentence1 = "商标名称：荣耀，拼音：rong2-yao4，笔画：122451234-24313554154132400000"
    sentence2 = "商标名称：茉耀，拼音：mo4-yao4，笔画：12211234-24313554154132400000"
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt")
    # 進行預測
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 獲取預測結果
        label_names = ["相似", "疑似", "不相似"]
        predicted_class = torch.argmax(predictions, dim=1).item()
        
        print(f"預測類別: {label_names[predicted_class]}")
        print(f"預測分數: {predictions[0][predicted_class].item():.4f}")
        
        # 顯示所有類別的分數
        for i, label in enumerate(label_names):
            print(f"{label} 分數: {predictions[0][i].item():.4f}")
            
if __name__ == "__main__":
    main()