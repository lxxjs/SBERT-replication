import jittor as jt
import jittor.nn as nn
import numpy as np
from models.modeling_jittor import JittorBertModel, JittorConfig
from transformers import AutoModel, AutoConfig

def load_hf_weights(jit_model, hf_model_name):
    print(f"Loading weights from {hf_model_name} to Jittor model...")
    
    # 加载 PyTorch 模型，用来提取 state_dict
    pt_model = AutoModel.from_pretrained(hf_model_name)
    pt_state = pt_model.state_dict()
    # 获取 Jittor 模型的参数字典
    jit_state = jit_model.state_dict()

    is_roberta = "roberta" in hf_model_name.lower()
    loaded_cnt = 0
    
    for key, param in jit_state.items():
        # 尝试 1: 直接匹配
        pt_key = key
        
        # 尝试 2: 加上 bert. 前缀
        if pt_key not in pt_state:
            pt_key = "bert." + key
            
        # 尝试 3: 加上 roberta. 前缀
        if pt_key not in pt_state:
            pt_key = "roberta." + key
            
        # 尝试 4 (特殊情况): 遍历 pt_state 寻找结尾匹配的 key
        if pt_key not in pt_state:
            for potential_key in pt_state.keys():
                if potential_key.endswith(key):
                    pt_key = potential_key
                    break
        
        if pt_key in pt_state:
            pt_np = pt_state[pt_key].cpu().detach().numpy()

            if is_roberta and "position_embeddings" in key:
                # Jittor 模型期望从 index 0 开始是位置 0
                # RoBERTa 权重从 index 2 开始才是位置 0 (index 0,1 是 pad/reserved)
                shift = 2 
                if pt_np.shape[0] > shift:
                    # pt_np (514, 768) -> valid_weights (512, 768)
                    valid_weights = pt_np[shift:, :]
                    
                    if param.shape[0] == valid_weights.shape[0]:
                        pt_np = valid_weights
                    elif param.shape[0] > valid_weights.shape[0]:
                        # 如果 Jittor param 还是 514，则把有效权重放在前面，后面补零
                        pad_len = param.shape[0] - valid_weights.shape[0]
                        zeros = np.zeros((pad_len, valid_weights.shape[1]), dtype=valid_weights.dtype)
                        pt_np = np.concatenate([valid_weights, zeros], axis=0)
                    else:
                        pt_np = valid_weights[:param.shape[0], :]

            # token_type_embeddings 对 RoBERTa 可能只有 1 行，而 BERT 有 2 行
            if "token_type_embeddings" in key:
                if param.shape[0] == 2 and pt_np.shape[0] == 1:
                    # 复制一行，使得 index 0 和 1 都有权重
                    pt_np = np.concatenate([pt_np, pt_np], axis=0)

            param.assign(pt_np)
            loaded_cnt += 1
        else:
            pass

    print(f"Successfully loaded {loaded_cnt} parameters layers.")

class SBERTModel(nn.Module):
    def __init__(self, model_path='models/bert-large-uncased', pooling='mean'):
        super(SBERTModel, self).__init__()
        
        # 根据 model_name 读取配置
        hf_config = AutoConfig.from_pretrained(model_path)
        
        # 将 HF 配置转为 Jittor 配置
        jt_config = JittorConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=hf_config.type_vocab_size,
            layer_norm_eps=hf_config.layer_norm_eps,
            pad_token_id = 1 if 'roberta' in model_path else 0
        )
        
        # 初始化 Jittor 版 BERT
        self.transformer = JittorBertModel(jt_config)
        
        # 加载权重
        load_hf_weights(self.transformer, model_path)
        
        self.pooling = pooling
        self.hidden_size = hf_config.hidden_size
        
    def execute(self, input_ids, attention_mask=None, token_type_ids=None):
        # transformer 返回 (last_hidden_state, all_hidden_states)
        outputs, _ = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        if self.pooling == 'cls':
            # CLS Pooling
            embeddings = outputs[:, 0, :]
        
        elif self.pooling == 'mean':
            # Mean Pooling
            # [Batch, Seq] -> [Batch, Seq, Hidden]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(outputs).float32()
            
            # 对 embeddings 进行加权求和（只保留非 Padding 部分）
            sum_embeddings = jt.sum(outputs * input_mask_expanded, dim=1)
            
            # 计算非 Padding 的 token 数量（加一个极小值 1e-9 防止除以 0）
            sum_mask = input_mask_expanded.sum(dim=1)
            sum_mask = jt.clamp(sum_mask, min_v=1e-9)
            embeddings = sum_embeddings / sum_mask
                
        else:
            # Max Pooling
            if attention_mask is not None:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(outputs).float32()
                outputs[input_mask_expanded == 0] = -1e9 # Mask 掉的部分设为极小值
            
            embeddings = jt.max(outputs, dim=1)
            
        return embeddings