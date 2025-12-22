import jittor as jt
from models.siamese_network import SiameseSBERT
from dataloader import NLIDataset
from transformers import AutoTokenizer
from data_utils import load_nli_data
from evaluate import evaluate_model

def train():
    jt.flags.use_cuda = 0
    
    model_path = "models/roberta-base"
    model = SiameseSBERT(model_path)
    optimizer = jt.optim.Adam(model.parameters(), lr=2e-5) # 设定 lr=2e-5
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    nli_data_paths = [
        'data/snli_1.0/snli_1.0_train.jsonl',         # SNLI 训练集
        'data/multinli_1.0/multinli_1.0_train.jsonl'  # MultiNLI 训练集
    ]
    train_samples = load_nli_data(nli_data_paths, limit=5000)
    train_dataset = NLIDataset(train_samples, tokenizer)
    train_loader = train_dataset.set_attrs(batch_size=16, shuffle=True, collate_batch=train_dataset.collate_batch)

    snli_dev_samples = load_nli_data(['data/snli_1.0/snli_1.0_dev.jsonl'], limit=500)
    snli_dev_dataset = NLIDataset(snli_dev_samples, tokenizer)
    
    mnli_dev_samples = load_nli_data(['data/multinli_1.0/multinli_1.0_dev_matched.jsonl'], limit=500)
    mnli_dev_dataset = NLIDataset(mnli_dev_samples, tokenizer)

    # 训练循环（仅 1 Epoch）
    model.train()
    for batch_idx, (in_a, in_b, labels) in enumerate(train_loader):
        loss, logits = model(in_a, in_b, labels)
        
        optimizer.step(loss) # 更新梯度
        
        if batch_idx % 10 == 0:
            print(f"Epoch 1, Step {batch_idx}, Loss: {loss.item():.4f}")

        if batch_idx > 0 and batch_idx % 50 == 0:
            print("-" * 30)
            acc_snli = evaluate_model(model, snli_dev_dataset, dataset_name="SNLI-Dev")
            acc_mnli = evaluate_model(model, mnli_dev_dataset, dataset_name="MNLI-Dev")
            print("-" * 30)

    model.save("data/sbert_jittor_roberta.pkl")

if __name__ == "__main__":
    train()