import jittor as jt
import jittor.nn as nn
from models.modeling_jittor import JittorBertModel, JittorConfig
from transformers import AutoModel, AutoConfig

def load_hf_weights(jit_model, hf_model_name):
    print(f"Loading weights from {hf_model_name} to Jittor model...")
    
    # 使用 Transformers 下载并加载 PyTorch 模型，用来提取 state_dict
    pt_model = AutoModel.from_pretrained(hf_model_name)
    pt_state = pt_model.state_dict()
    
    # 获取 Jittor 模型的参数字典
    jit_state = jit_model.state_dict()
    
    # 建立映射并赋值
    prefix = ""
    if "roberta" in hf_model_name:
        prefix = "roberta."
    else:
        if any(k.startswith("bert.") for k in pt_state.keys()):
            prefix = "bert."
            
    loaded_cnt = 0
    for key, param in jit_state.items():
        # 构造 PyTorch 对应的 Key
        pt_key = prefix + key
        
        # 修正特殊情况
        if pt_key not in pt_state:
            if "LayerNorm.weight" in pt_key:
                alt_key = pt_key.replace("LayerNorm.weight", "LayerNorm.gamma")
                if alt_key in pt_state: pt_key = alt_key
            elif "LayerNorm.bias" in pt_key:
                alt_key = pt_key.replace("LayerNorm.bias", "LayerNorm.beta")
                if alt_key in pt_state: pt_key = alt_key
        
        if pt_key in pt_state:
            # PyTorch Tensor -> Numpy
            pt_np = pt_state[pt_key].cpu().detach().numpy()
            
            # 形状检查
            if pt_np.shape != param.shape:
                if pt_np.T.shape == param.shape:
                    pt_np = pt_np.T
                else:
                    print(f"Global Skip: {key} shape mismatch. Jittor: {param.shape}, PT: {pt_np.shape}")
                    continue
            
            param.assign(pt_np)
            loaded_cnt += 1
        else:
            pass

    print(f"Successfully loaded {loaded_cnt} parameters layers.")

class SBERTModel(nn.Module):
    def __init__(self, model_name='bert-large-uncased', pooling='mean'):
        super(SBERTModel, self).__init__()
        
        # 根据 model_name 读取配置
        hf_config = AutoConfig.from_pretrained(model_name)
        
        # 将 HF 配置转为 Jittor 配置
        jt_config = JittorConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=hf_config.type_vocab_size,
            layer_norm_eps=hf_config.layer_norm_eps
        )
        
        # 初始化 Jittor 版 BERT
        self.transformer = JittorBertModel(jt_config)
        
        # 加载权重
        load_hf_weights(self.transformer, model_name)
        
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
            if attention_mask is not None:
                # 扩展 mask: [batch, seq, 1] -> [batch, seq, hidden]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(outputs).float32()
                
                sum_embeddings = jt.sum(outputs * input_mask_expanded, dim=1)
                sum_mask = input_mask_expanded.sum(dim=1)
                sum_mask = jt.clamp(sum_mask, min_v=1e-9)
                
                embeddings = sum_embeddings / sum_mask
            else:
                embeddings = jt.mean(outputs, dim=1)
                
        else:
            # Max Pooling
            if attention_mask is not None:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(outputs).float32()
                outputs[input_mask_expanded == 0] = -1e9 # Mask 掉的部分设为极小值
            
            embeddings = jt.max(outputs, dim=1)
            
        return embeddings