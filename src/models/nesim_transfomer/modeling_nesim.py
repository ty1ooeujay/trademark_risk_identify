import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertModel, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class NesimConfig(BertConfig):
    """Nesim模型配置类"""
    model_type = "nesim"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        # 新增参数
        pinyin_vocab_size=1000,  # 拼音词汇表大小
        stroke_vocab_size=1000,  # 笔顺编码词汇表大小
        pinyin_embedding_dim=128,  # 拼音嵌入维度
        stroke_embedding_dim=128,  # 笔顺编码嵌入维度
        gru_hidden_size=256,  # GRU隐藏层大小
        gru_num_layers=2,  # GRU层数
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
            classifier_dropout=classifier_dropout,
            **kwargs
        )
        self.pinyin_vocab_size = pinyin_vocab_size
        self.stroke_vocab_size = stroke_vocab_size
        self.pinyin_embedding_dim = pinyin_embedding_dim
        self.stroke_embedding_dim = stroke_embedding_dim
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers


class NesimEmbeddings(nn.Module):
    """Nesim嵌入层，使用GRU处理拼音和笔顺编码"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # 位置嵌入
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # 类型嵌入
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # 拼音嵌入和GRU
        self.pinyin_embeddings = nn.Embedding(config.pinyin_vocab_size, config.pinyin_embedding_dim)
        self.pinyin_gru = nn.GRU(
            input_size=config.pinyin_embedding_dim,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            dropout=config.hidden_dropout_prob if config.gru_num_layers > 1 else 0
        )
        
        # 笔顺编码嵌入和GRU
        self.stroke_embeddings = nn.Embedding(config.stroke_vocab_size, config.stroke_embedding_dim)
        self.stroke_gru = nn.GRU(
            input_size=config.stroke_embedding_dim,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            dropout=config.hidden_dropout_prob if config.gru_num_layers > 1 else 0
        )
        
        # 投影层，将GRU输出投影到hidden_size
        self.pinyin_projection = nn.Linear(config.gru_hidden_size, config.hidden_size)
        self.stroke_projection = nn.Linear(config.gru_hidden_size, config.hidden_size)
        
        # LayerNorm和Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 位置ID
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids=None, pinyin_ids=None, stroke_ids=None, 
                token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 获取位置和类型嵌入
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # 处理拼音嵌入
        pinyin_features = None
        if pinyin_ids is not None:
            pinyin_embeds = self.pinyin_embeddings(pinyin_ids)
            pinyin_output, _ = self.pinyin_gru(pinyin_embeds)
            # 取最后一个时间步的输出
            pinyin_features = self.pinyin_projection(pinyin_output[:, -1, :])  # [batch_size, hidden_size]
        
        # 处理笔顺编码嵌入
        stroke_features = None
        if stroke_ids is not None:
            stroke_embeds = self.stroke_embeddings(stroke_ids)
            stroke_output, _ = self.stroke_gru(stroke_embeds)
            # 取最后一个时间步的输出
            stroke_features = self.stroke_projection(stroke_output[:, -1, :])  # [batch_size, hidden_size]
        
        # 组合所有嵌入
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        
        # 如果有拼音特征，将其添加到每个token位置
        if pinyin_features is not None:
            # 将拼音特征扩展到序列长度
            pinyin_features_expanded = pinyin_features.unsqueeze(1).expand(-1, seq_length, -1)
            embeddings = embeddings + pinyin_features_expanded
        
        # 如果有笔顺特征，将其添加到每个token位置
        if stroke_features is not None:
            # 将笔顺特征扩展到序列长度
            stroke_features_expanded = stroke_features.unsqueeze(1).expand(-1, seq_length, -1)
            embeddings = embeddings + stroke_features_expanded
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class NesimModel(BertModel):
    """Nesim主模型"""
    
    def __init__(self, config, add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.config = config
        
        self.embeddings = NesimEmbeddings(config)
        
        # 创建编码器和池化层
        from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        
        # 初始化权重
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        pinyin_ids=None,
        stroke_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # 过去键值长度
        if past_key_values is None:
            past_key_values_length = 0
        else:
            past_key_values_length = past_key_values[0][0].shape[2]
        
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # 扩展attention_mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
        # 准备head_mask
        if head_mask is not None:
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        # 获取嵌入
        embedding_output = self.embeddings(
            input_ids=input_ids,
            pinyin_ids=pinyin_ids,
            stroke_ids=stroke_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        # 编码器
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class NesimForSequenceClassification(BertForSequenceClassification):
    """Nesim二分类模型"""
    
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.nesim = NesimModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化权重
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        pinyin_ids=None,
        stroke_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.nesim(
            input_ids,
            pinyin_ids=pinyin_ids,
            stroke_ids=stroke_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )