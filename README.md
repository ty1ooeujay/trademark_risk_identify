# Nesim模型 - 基于Transformers的文本相似度二分类模型

## 项目简介

Nesim是一个基于Transformers库实现的文本相似度二分类模型，专门用于判断两个中文词语是否相似。该模型的特点是在处理拼音和笔顺编码时使用了GRU（门控循环单元）进行特征提取。

## 模型架构

### 核心组件

1. **NesimConfig**: 模型配置类，继承自BertConfig，添加了拼音和笔顺编码相关的参数
2. **NesimEmbeddings**: 嵌入层，使用GRU处理拼音和笔顺编码
3. **NesimModel**: 主模型，基于BERT架构
4. **NesimForSequenceClassification**: 二分类模型，用于相似度判断

### 关键特性

- **多模态输入**: 支持文本、拼音、笔顺编码三种输入
- **GRU特征提取**: 使用GRU处理拼音和笔顺编码序列
- **BERT架构**: 基于BERT的Transformer编码器
- **二分类任务**: 判断两个词语是否相似

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 简单使用示例

```python
from model.modeling_nesim import NesimConfig, NesimForSequenceClassification
from transformers import AutoTokenizer
import torch

# 创建配置
config = NesimConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=6,
    num_labels=2,  # 二分类
    pinyin_vocab_size=1000,
    stroke_vocab_size=1000,
    gru_hidden_size=256,
    gru_num_layers=2
)

# 创建模型
model = NesimForSequenceClassification(config)

# 创建tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

# 准备输入数据
text_a = "苹果"
text_b = "橙子"
pinyin_a = [1, 2, 3]  # 拼音编码
pinyin_b = [4, 5, 6]  # 拼音编码
stroke_a = [101, 102, 103]  # 笔顺编码
stroke_b = [104, 105, 106]  # 笔顺编码

# 数据预处理
combined_text = f"{text_a} [SEP] {text_b}"
encoding = tokenizer(combined_text, truncation=True, padding='max_length', 
                   max_length=128, return_tensors='pt')

# 处理拼音和笔顺编码
combined_pinyin = pinyin_a + pinyin_b + [0] * (20 - len(pinyin_a) - len(pinyin_b))
combined_stroke = stroke_a + stroke_b + [0] * (20 - len(stroke_a) - len(stroke_b))

# 模型推理
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        pinyin_ids=torch.tensor(combined_pinyin, dtype=torch.long).unsqueeze(0),
        stroke_ids=torch.tensor(combined_stroke, dtype=torch.long).unsqueeze(0)
    )
    
    prediction = torch.argmax(outputs.logits, dim=-1)
    print(f"预测结果: {'相似' if prediction.item() == 1 else '不相似'}")
```

### 2. 运行示例

```bash
# 运行简单示例
python simple_example.py

# 运行完整训练示例
python example_usage.py
```

## 模型参数说明

### NesimConfig参数

- `vocab_size`: 词汇表大小 (默认: 30522)
- `hidden_size`: 隐藏层大小 (默认: 768)
- `num_hidden_layers`: Transformer层数 (默认: 12)
- `num_attention_heads`: 注意力头数量 (默认: 12)
- `num_labels`: 分类标签数量 (默认: 2)
- `pinyin_vocab_size`: 拼音词汇表大小 (默认: 1000)
- `stroke_vocab_size`: 笔顺编码词汇表大小 (默认: 1000)
- `pinyin_embedding_dim`: 拼音嵌入维度 (默认: 128)
- `stroke_embedding_dim`: 笔顺编码嵌入维度 (默认: 128)
- `gru_hidden_size`: GRU隐藏层大小 (默认: 256)
- `gru_num_layers`: GRU层数 (默认: 2)

## 输入格式

### 文本输入
- 格式: `"词语A [SEP] 词语B"`
- 使用BERT tokenizer进行分词

### 拼音输入
- 格式: 数字序列，表示拼音的编码
- 长度: 固定长度20（不足部分用0填充）

### 笔顺输入
- 格式: 数字序列，表示笔顺的编码
- 长度: 固定长度20（不足部分用0填充）

## 输出格式

模型输出包含：
- `logits`: 原始预测分数 [batch_size, num_labels]
- `loss`: 损失值（训练时）
- `hidden_states`: 隐藏状态
- `attentions`: 注意力权重

## 训练建议

1. **数据准备**: 准备包含文本、拼音、笔顺编码和标签的训练数据
2. **超参数调优**: 根据数据规模调整学习率、批次大小等参数
3. **正则化**: 使用dropout和权重衰减防止过拟合
4. **评估指标**: 使用准确率、F1分数等指标评估模型性能

## 文件结构

```
TMRisk-Identi/
├── model/
│   ├── __init__.py
│   └── modeling_nesim.py    # 核心模型实现
├── main.py                  # 原始主文件
├── requirements.txt         # 依赖包
├── simple_example.py        # 简单使用示例
├── example_usage.py        # 完整训练示例
└── README.md               # 说明文档
```

## 注意事项

1. 确保安装了正确版本的transformers库
2. 拼音和笔顺编码需要预先转换为数字序列
3. 模型支持GPU训练，会自动检测可用设备
4. 建议使用预训练的BERT权重初始化模型

## 扩展功能

- 支持更多语言
- 添加更多特征（如字形特征）
- 实现多标签分类
- 添加注意力可视化功能
