import jittor as jt

def evaluate_model(model, dataset, dataset_name="Dev Set", batch_size=16):
    print(f"\n--- Evaluating on {dataset_name} ({len(dataset)} samples) ---")
    model.eval()
    
    # 创建验证集的 DataLoader
    val_loader = dataset.set_attrs(
        batch_size=batch_size, 
        shuffle=False, 
        collate_batch=dataset.collate_batch
    )
    
    total_acc = 0
    total_count = 0
    
    with jt.no_grad():
        for in_a, in_b, labels in val_loader:
            # 前向传播，获取 logits
            _, logits = model(in_a, in_b, labels)
            
            # 计算准确率
            pred = jt.argmax(logits, dim=1)[0]
            acc = (pred == labels).float().sum()
            
            total_acc += acc.item()
            total_count += len(labels)
            
    accuracy = total_acc / total_count
    print(f"[{dataset_name}] Accuracy: {accuracy:.2%} ({int(total_acc)}/{total_count})")
    
    model.train()
    return accuracy