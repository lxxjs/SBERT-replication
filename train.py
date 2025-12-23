import jittor as jt
from models.siamese_network import SiameseSBERT
from dataloader import NLIDataset
from transformers import AutoTokenizer
from data_utils import load_nli_data
from evaluate import evaluate_model

def clip_grad_norm_(parameters, optimizer, max_norm=1.0):
    if isinstance(parameters, jt.Var):
        parameters = [parameters]

    grads = []
    valid_params = []
    
    for p in parameters:
        g = p.opt_grad(optimizer)
        if g is not None:
            grads.append(g)
            valid_params.append(p)
    
    if len(grads) == 0:
        return 0.0

    max_norm = float(max_norm)

    # 计算全局范数
    total_norm_sq = jt.array(0.0)
    for g in grads:
        total_norm_sq += g.sqr().sum()
    
    total_norm = total_norm_sq.sqrt()
    
    # 只有当 total_norm > max_norm 时才需要缩放
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = jt.minimum(clip_coef, jt.array(1.0))
    
    for g in grads:
        g.assign(g * clip_coef)
    
    return total_norm.item()

def train():
    jt.flags.use_cuda = 1
    
    model_path = "models/bert-large-uncased"
    model = SiameseSBERT(model_path)
    optimizer = jt.optim.Adam(model.parameters(), lr=2e-5) # 设定 lr=2e-5
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    nli_data_paths = [
        'data/snli_1.0/snli_1.0_train.jsonl',
        'data/multinli_1.0/multinli_1.0_train.jsonl'
    ]
    train_samples = load_nli_data(nli_data_paths)
    total_samples = len(train_samples)
    train_dataset = NLIDataset(train_samples, tokenizer)
    train_loader = train_dataset.set_attrs(batch_size=16, shuffle=True, collate_batch=train_dataset.collate_batch)

    snli_dev_samples = load_nli_data(['data/snli_1.0/snli_1.0_dev.jsonl'])
    snli_dev_dataset = NLIDataset(snli_dev_samples, tokenizer)
    
    mnli_dev_samples = load_nli_data(['data/multinli_1.0/multinli_1.0_dev_matched.jsonl'])
    mnli_dev_dataset = NLIDataset(mnli_dev_samples, tokenizer)

    batch_size = 16
    total_steps = total_samples // batch_size
    warmup_proportion = 0.1  # 预热占比 10%
    warmup_steps = int(total_steps * warmup_proportion)  # 预热步数
    target_lr = 2e-5  # 目标学习率

    # 训练循环（1 Epoch）
    model.train()
    for batch_idx, (in_a, in_b, labels) in enumerate(train_loader):
        if batch_idx < warmup_steps:
            # 预热阶段
            current_lr = target_lr * (batch_idx + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        loss, logits = model(in_a, in_b, labels)
        optimizer.backward(loss) # 计算梯度

        # 在更新参数前裁剪梯度
        clip_grad_norm_(model.parameters(), optimizer, max_norm=1.0)
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch 1, Step {batch_idx}, Loss: {loss.item():.4f}")

        if batch_idx > 0 and batch_idx % 1000 == 0:
            print("-" * 30)
            acc_snli = evaluate_model(model, snli_dev_dataset, dataset_name="SNLI-Dev")
            acc_mnli = evaluate_model(model, mnli_dev_dataset, dataset_name="MNLI-Dev")
            print("-" * 30)

    model.save("data/sbert_jittor_roberta.pkl")

if __name__ == "__main__":
    train()