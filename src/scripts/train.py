import time
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str, default="data/datasets/sim_trademark_pairs.csv")
    argparser.add_argument("--train_split", type=float, default=0.8)
    argparser.add_argument("--val_split", type=float, default=0.1)
    argparser.add_argument("--test_split", type=float, default=0.1)
    args = argparser.parse_args()
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained("jackietung/bert-base-chinese-finetuned-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("jackietung/bert-base-chinese-finetuned-sentiment")
    
    # 加载数据集
    fileclass = args.data_path.split(".")[-1]
    if fileclass == "csv":
        dataset = load_dataset("csv", data_files=args.data_path)
        dataset = dataset["train"].train_test_split(train_size=0.8, seed=42)
    else:
        raise ValueError(f"不支持的文件格式: {fileclass}")

    # 分词
    dataset = dataset.map(lambda examples: tokenizer(examples["sentence1"], examples["sentence2"]), batched=True)

    print("-----------------数据样例-----------------")
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print(f"时间戳: {timestamp}")
    print(f"数据集: {dataset}")
    print(dataset["train"][0]["sentence1"])
    print(dataset["train"][0]["sentence2"])
    print(f"标签: {dataset['train'][0]['label']}")
    print(f"解码输入: {tokenizer.decode(dataset['train'][0]['input_ids'])}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir= f"weights/{timestamp}",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=100,
        eval_strategy="epoch",
        warmup_steps=10,
        logging_steps=101,
        save_strategy="epoch",
        include_for_metrics=["loss"],
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()