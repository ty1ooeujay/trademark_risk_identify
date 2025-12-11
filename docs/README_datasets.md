# 商标风险数据集加载模块

这个模块使用HuggingFace datasets库来加载和管理商标相似度数据集。

## 文件结构

```
src/scripts/trademark_datasets.py  # 主要数据集加载代码
example_usage.py                  # 使用示例
requirements.txt                  # 依赖包列表
data/datasets/trademarks_pairs.json  # 原始数据集
```

## 安装依赖

首先激活tmrisk conda环境，然后安装依赖包：

```bash
conda activate tmrisk
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```python
from src.scripts.trademark_datasets import create_trademark_dataset

# 创建数据集
dataset, info = create_trademark_dataset()

print(f"训练集大小: {len(dataset['train'])}")
print(f"验证集大小: {len(dataset['validation'])}")
print(f"测试集大小: {len(dataset['test'])}")

# 查看一个样本
sample = dataset['train'][0]
print(sample)
```

### 高级使用

```python
from src.scripts.trademark_datasets import TrademarkRiskDataset

# 创建数据集处理器
dataset_handler = TrademarkRiskDataset(
    tokenizer_name="bert-base-chinese",
    max_length=128,
    val_split=0.15,
    test_split=0.15
)

# 加载数据
raw_dataset = dataset_handler.load_data()

# 预处理数据
processed_dataset = dataset_handler.get_processed_dataset(batch_size=16)

# 获取类别权重（处理不平衡数据）
class_weights = dataset_handler.get_class_weights()
```

## 数据集格式

原始数据格式为JSON数组，每个样本包含：
- `trademark1`: 第一个商标文本
- `trademark2`: 第二个商标文本
- `label`: 相似度标签（0=相似，1=部分相似，2=不相似等）

预处理后的数据包含：
- `input_ids`: 分词后的token IDs
- `attention_mask`: 注意力掩码
- `labels`: 标签

## 使用示例运行

运行完整示例：

```bash
python example_usage.py
```

只运行数据集测试：

```bash
python src/scripts/trademark_datasets.py
```

## 配置参数

### TrademarkRiskDataset 参数

- `data_path`: 数据文件路径（默认自动查找）
- `tokenizer_name`: 分词器名称（默认"bert-base-chinese"）
- `max_length`: 最大序列长度（默认128）
- `val_split`: 验证集比例（默认0.1）
- `test_split`: 测试集比例（默认0.1）
- `seed`: 随机种子（默认42）

### create_trademark_dataset 参数

- 同上，并额外支持：
- `batch_size`: 批处理大小（默认32）

## 功能特性

1. **自动数据分割**: 支持训练/验证/测试集的自动分割
2. **数据预处理**: 自动进行分词和编码
3. **类别权重计算**: 自动计算类别权重以处理不平衡数据
4. **兼容PyTorch**: 预处理后的数据可直接用于PyTorch DataLoader
5. **灵活配置**: 支持自定义分词器、序列长度等参数

## 数据集信息

运行代码后会显示：
- 数据集大小分布
- 标签分布统计
- 类别权重
- 文本长度统计

## 故障排除

### 常见问题

1. **ModuleNotFoundError**: 确保安装了所有依赖包
   ```bash
   pip install -r requirements.txt
   ```

2. **文件未找到**: 检查数据文件路径是否正确

3. **CUDA错误**: 如果没有GPU，确保代码使用CPU
   ```python
   device = torch.device("cpu")
   ```

### 验证安装

运行以下代码验证安装：

```python
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())

# 测试分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
print("分词器加载成功")

# 测试数据集加载
dataset = load_dataset('json', data_files='data/datasets/trademarks_pairs.json', split='train[:5]')
print(f"数据集加载成功，样本数量: {len(dataset)}")
```
