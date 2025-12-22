import jittor as jt
import jittor.nn as nn
import numpy as np

class JittorConfig:
    def __init__(self, vocab_size=30522, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                 intermediate_size=4096, max_position_embeddings=512, type_vocab_size=2, layer_norm_eps=1e-12, pad_token_id=0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id

class JittorEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(0.1)
        self.padding_idx = getattr(config, 'pad_token_id', 0)

    def execute(self, input_ids, token_type_ids=None, position_ids=None):
        if token_type_ids is None:
            token_type_ids = jt.zeros_like(input_ids)

        seq_length = input_ids.shape[1]
        position_ids = jt.arange(seq_length, dtype='int32').unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        
        if self.padding_idx is not None:
            pad_mask = (input_ids == self.padding_idx)
            words_embeddings[pad_mask] = 0.0

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class JittorSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        # [batch, seq_len, heads, head_size]
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def execute(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        # Scaled Dot Product Attention
        attention_scores = jt.matmul(q, k.transpose(0, 1, 3, 2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Transpose to [batch, 1, 1, seq_len]
            extended_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
            attention_scores = attention_scores + extended_mask
            
        attention_probs = nn.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = jt.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).view(hidden_states.shape[0], hidden_states.shape[1], -1)
        return context_layer

class JittorSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(0.1)

    def execute(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class JittorAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = JittorSelfAttention(config)
        self.output = JittorSelfOutput(config)

    def execute(self, hidden_states, attention_mask):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

class JittorIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def execute(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class JittorOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(0.1)

    def execute(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class JittorLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = JittorAttention(config)
        self.intermediate = JittorIntermediate(config)
        self.output = JittorOutput(config)

    def execute(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class JittorEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([JittorLayer(config) for _ in range(config.num_hidden_layers)])

    def execute(self, hidden_states, attention_mask):
        all_hidden_states = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states

class JittorBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = JittorEmbeddings(config)
        self.encoder = JittorEncoder(config)
        self.config = config

    def execute(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = jt.ones_like(input_ids)
        
        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output, all_hidden_states = self.encoder(embedding_output, attention_mask)
        return sequence_output, all_hidden_states