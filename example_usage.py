"""
商标风险数据集使用示例
演示如何使用datasets.py加载和使用商标相似度数据集
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.scripts.trademark_datasets import create_trademark_dataset, TrademarkRiskDataset
import torch
from torch.utils.data import DataLoader


def basic_usage_example():
    """基本使用示例"""
    print("=== 基本使用示例 ===")

    # 创建数据集
    dataset, info = create_trademark_dataset(
        tokenizer_name="bert-base-chinese",
        max_length=128
    )

    print(f"数据集信息: {info}")
    print(f"训练集大小: {len(dataset['train'])}")
    print(f"验证集大小: {len(dataset['validation'])}")
    print(f"测试集大小: {len(dataset['test'])}")

    # 查看一个样本
    sample = dataset['train'][0]
    print(f"样本keys: {sample.keys()}")
    print(f"输入形状: {sample['input_ids'].shape}")
    print(f"标签: {sample['labels']}")


def advanced_usage_example():
    """高级使用示例"""
    print("\n=== 高级使用示例 ===")

    # 创建数据集实例
    dataset_handler = TrademarkRiskDataset(
        tokenizer_name="bert-base-chinese",
        max_length=128,
        val_split=0.15,
        test_split=0.15
    )

    # 加载原始数据集
    raw_dataset = dataset_handler.load_data()

    # 获取类别权重
    class_weights = dataset_handler.get_class_weights()
    print(f"类别权重: {class_weights}")

    # 预处理数据集
    processed_dataset = dataset_handler.get_processed_dataset(batch_size=16)

    # 创建DataLoader
    train_dataloader = DataLoader(
        processed_dataset['train'],
        batch_size=8,
        shuffle=True
    )

    val_dataloader = DataLoader(
        processed_dataset['validation'],
        batch_size=8,
        shuffle=False
    )

    print("创建了DataLoader")
    print(f"训练批次数量: {len(train_dataloader)}")
    print(f"验证批次数量: {len(val_dataloader)}")

    # 查看一个批次
    batch = next(iter(train_dataloader))
    print(f"批次keys: {batch.keys()}")
    print(f"批次输入形状: {batch['input_ids'].shape}")
    print(f"批次标签形状: {batch['labels'].shape}")


def model_training_example():
    """模型训练示例"""
    print("\n=== 模型训练示例 ===")

    try:
        from transformers import AutoModelForSequenceClassification, AdamW
        from tqdm import tqdm

        # 创建数据集
        dataset, info = create_trademark_dataset(
            tokenizer_name="bert-base-chinese",
            max_length=128
        )

        # 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-chinese",
            num_labels=info['num_classes']
        )

        # 创建优化器
        optimizer = AdamW(model.parameters(), lr=2e-5)

        # 创建DataLoader
        train_dataloader = DataLoader(
            dataset['train'],
            batch_size=8,
            shuffle=True
        )

        # 训练循环示例
        model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print(f"使用设备: {device}")
        print("开始训练示例...")

        # 只训练一个epoch的前几个批次作为示例
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            if batch_idx >= 5:  # 只训练5个批次作为示例
                break

            # 将数据移到设备
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(".4f")

        print("训练示例完成")

    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保安装了必要的包: pip install transformers tqdm")


def data_analysis_example():
    """数据分析示例"""
    print("\n=== 数据分析示例 ===")

    dataset_handler = TrademarkRiskDataset()
    raw_dataset = dataset_handler.load_data()

    # 分析商标长度分布
    train_data = raw_dataset['train']

    trademark1_lengths = [len(text) for text in train_data['trademark1']]
    trademark2_lengths = [len(text) for text in train_data['trademark2']]

    print(f"商标1平均长度: {sum(trademark1_lengths) / len(trademark1_lengths):.2f}")
    print(f"商标2平均长度: {sum(trademark2_lengths) / len(trademark2_lengths):.2f}")
    print(f"商标1最大长度: {max(trademark1_lengths)}")
    print(f"商标2最大长度: {max(trademark2_lengths)}")

    # 分析标签分布
    labels = train_data['label']
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    print("标签分布:")
    for label, count in sorted(label_counts.items()):
        percentage = count / len(labels) * 100
        print(".1f")


if __name__ == "__main__":
    print("商标风险数据集使用示例")
    print("=" * 50)

    try:
        # 基本使用
        basic_usage_example()

        # 高级使用
        advanced_usage_example()

        # 数据分析
        data_analysis_example()

        # 模型训练示例（可选）
        # model_training_example()

        print("\n所有示例运行完成！")

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()