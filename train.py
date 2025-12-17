import time
import math
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_dataset

def load_model(model_path: str):
    """加载模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, default="src/models/bert-base-chinese-finetuned-sentiment")
    argparser.add_argument("--data_path", type=str, default="data/datasets/sim_trademark_pairs.csv")
    argparser.add_argument("--train_split", type=float, default=0.8)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--learning_rate", type=float, default=2e-5)
    argparser.add_argument("--warmup_steps", type=int, default=10)
    args = argparser.parse_args()
    
    tokenizer, model = load_model(args.model_path)
    
    # 加载数据集
    fileclass = args.data_path.split(".")[-1]
    if fileclass == "csv":
        dataset = load_dataset("csv", data_files=args.data_path)
        dataset = dataset["train"].train_test_split(train_size=args.train_split, seed=args.seed)
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
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        warmup_steps=args.warmup_steps,
        logging_steps=int(math.ceil(dataset['train'].num_rows / args.batch_size)),
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