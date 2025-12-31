import jittor as jt
from scipy.stats import spearmanr
from STSdatasets_loader import load_sts_dataset
from model_loader import load_sbert_model

STS_DATASETS = {
    "STS12": "mteb/sts12-sts",
    "STS13": "mteb/sts13-sts",
    "STS14": "mteb/sts14-sts",
    "STS15": "mteb/sts15-sts",
    "STS16": "mteb/sts16-sts",
    "STSb":  "mteb/stsbenchmark-sts",
    "SICK-R": "mteb/sickr-sts",
}


def cosine_similarity(a, b):
    a = a / jt.norm(a, dim=1, keepdim=True)
    b = b / jt.norm(b, dim=1, keepdim=True)
    return (a * b).sum(dim=1)

def evaluate_sts(model, sentence_pairs, gold_scores, batch_size=32):
    model.eval()
    preds = []

    with jt.no_grad():
        for i in range(0, len(sentence_pairs), batch_size):
            batch = sentence_pairs[i:i+batch_size]
            s1 = [x[0] for x in batch]
            s2 = [x[1] for x in batch]

            emb1 = model.encode(s1)   # sentence embeddings
            emb2 = model.encode(s2)

            sim = cosine_similarity(emb1, emb2)
            preds.extend(sim.numpy())

    corr, _ = spearmanr(preds, gold_scores)
    return corr



def run_all_sts_datasets(model, datasets=STS_DATASETS, batch_size=32):
    """Load each dataset, run unsupervised STS eval, and return a dict of Spearman correlations."""
    results = {}
    for short_name, hf_name in datasets.items():
        sentence_pairs, gold_scores = load_sts_dataset(hf_name)
        corr = evaluate_sts(model, sentence_pairs, gold_scores, batch_size=batch_size)
        results[short_name] = float(corr)
    return results


def main():
    model = load_sbert_model(
        checkpoint_path="models/SRoBERTa-base.pkl",
        model_path="roberta-base",
        # checkpoint_path="models/SBERT-large-uncased.pkl",
        # model_path="bert-large-uncased",
        pooling="mean",
        strict=True
    )

    # quick sanity check
    emb = model.encode([
        "A man is playing guitar.",
        "A person plays music."
    ])
    print("Embedding shape:", emb.shape)

    results = run_all_sts_datasets(model)
    print(results)


if __name__ == "__main__":
    main()