from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
from transformers import BertConfig, BertModel, BertForSequenceClassification
from typing import List, Dict, Optional

class NesimConfig(PretrainedConfig):
    model_type = "nesim"

    def __init__(
        self,
        attention_probs_dropout_prob: float = 0.1,
        classifier_dropout: Optional[float] = None,
        directionality: str = "bidi",
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        hidden_size: int = 768,
        id2label: Optional[Dict[str, str]] = None,
        initializer_range: float = 0.02,
        intermediate_size: int = 3072,
        label2id: Optional[Dict[str, int]] = None,
        layer_norm_eps: float = 1e-12,
        max_position_embeddings: int = 512,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        pad_token_id: int = 0,
        pooler_fc_size: int = 768,
        pooler_num_attention_heads: int = 12,
        pooler_num_fc_layers: int = 3,
        pooler_size_per_head: int = 128,
        pooler_type: str = "first_token_transform",
        position_embedding_type: str = "absolute",
        torch_dtype: str = "float32",
        type_vocab_size: int = 2,
        use_cache: bool = True,
        vocab_size: int = 21128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 设置默认的 id2label 和 label2id
        if id2label is None:
            id2label = {"0": "相似", "1": "疑似", "2": "不相似"}
        if label2id is None:
            label2id = {"不相似": 2, "疑似": 1, "相似": 0}
        
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout = classifier_dropout
        self.directionality = directionality
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.id2label = id2label
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.label2id = label2id
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.pooler_fc_size = pooler_fc_size
        self.pooler_num_attention_heads = pooler_num_attention_heads
        self.pooler_num_fc_layers = pooler_num_fc_layers
        self.pooler_size_per_head = pooler_size_per_head
        self.pooler_type = pooler_type
        self.position_embedding_type = position_embedding_type
        self.dtype = torch_dtype
        self.type_vocab_size = type_vocab_size
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        
class NesimModel(PreTrainedModel):
    pass

def main():
    tm = TestModelConfig()
    print(tm)


if __name__ == "__main__":
    main()