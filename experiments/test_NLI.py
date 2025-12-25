import jittor as jt
import numpy as np
from models.siamese_network import SiameseSBERT
from dataloader import NLIDataset
from transformers import AutoTokenizer
from data_utils import load_nli_data

jt.flags.use_cuda = 0

def evaluate_accuracy(model, dataset, batch_size=16):
    data_loader = dataset.set_attrs(
        batch_size=batch_size, 
        shuffle=False, 
        collate_batch=dataset.collate_batch
    )
    
    model.eval()
    total_correct = 0
    total_samples = 0
    
    print(f"  > Starting evaluation on {len(dataset)} samples...")
    
    with jt.no_grad():
        for i, (in_a, in_b, labels) in enumerate(data_loader):
            _, logits = model(in_a, in_b, labels)
            
            predictions = np.argmax(logits.numpy(), axis=1)
            targets = labels.numpy()
            
            total_correct += np.sum(predictions == targets)
            total_samples += len(targets)
            
    accuracy = total_correct / total_samples
    return accuracy

def test_single_model(model_path, weights_path, test_configs):
    print("="*60)
    print(f"Testing Model: {model_path}")
    print(f"Weights Path: {weights_path}")
    print("="*60)
    
    # 初始化模型架构
    try:
        model = SiameseSBERT(model_path=model_path)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # 加载训练好的权重
    try:
        print("Loading weights...")
        model.load(weights_path)
    except Exception as e:
        print(f"Error loading weights from {weights_path}: {e}")
        print("Skipping this model.")
        return

    # 准备 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 在各个数据集上进行评估
    for dataset_name, file_path in test_configs.items():
        print(f"\nLoading {dataset_name} data from {file_path}...")
        try:
            samples = load_nli_data([file_path])
            dataset = NLIDataset(samples, tokenizer)
            
            acc = evaluate_accuracy(model, dataset)
            print(f"Result [{dataset_name}]: Accuracy = {acc*100:.2f}%")
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found. Skipping.")
        except Exception as e:
            print(f"Error evaluating on {dataset_name}: {e}")

def main():
    nli_test_sets = {
        "SNLI-Test": "data/snli_1.0/snli_1.0_test.jsonl",
        "MNLI-Matched (Dev)": "data/multinli_1.0/multinli_1.0_dev_matched.jsonl",
        "MNLI-Mismatched (Dev)": "data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl"
    }

    models_to_test = [
        ("models/roberta-base", "models/SRoBERTa-base.pkl"), 
        ("models/bert-large-uncased", "models/SBERT-large-uncased.pkl"),
    ]

    for hf_name, pkl_path in models_to_test:
        test_single_model(hf_name, pkl_path, nli_test_sets)

if __name__ == "__main__":
    main()