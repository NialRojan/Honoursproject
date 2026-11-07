"""Text baselines evaluated under identical splits.

lr   -- Yada et al. 2022 bag-of-words logistic regression (unigram CountVectorizer,
        C=4.95, digits -> "number", punctuation stripped). Vectoriser is fit on the
        training fold only (Yada's released code fits it on all data).
bert -- bert-base-uncased fine-tune, the recipe of dp-detector's linguistic engine
        (3 epochs, batch 16, max_len 64, lr 2e-5). Trained per fold.

Each baseline returns a raw logit (positive-class margin) for the validation and
test rows, so calibration (temperature scaling) is applied outside, fit on the
validation rows only.
"""
import re

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def yada_preprocess(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"\d+", " number ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def lr_logits(train_texts, train_y, eval_sets, seed=42):
    vec = CountVectorizer(ngram_range=(1, 1))
    X = vec.fit_transform([yada_preprocess(t) for t in train_texts])
    clf = LogisticRegression(C=4.95, max_iter=2000, random_state=seed).fit(X, train_y)
    return [clf.decision_function(vec.transform([yada_preprocess(t) for t in ts]))
            for ts in eval_sets]


def bert_logits(train_texts, train_y, eval_sets, seed=42, epochs=3, lr=2e-5,
                batch_size=16, max_len=64, base="bert-base-uncased"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2).to(dev)
    enc = tok(list(train_texts), truncation=True, padding=True, max_length=max_len,
              return_tensors="pt")
    y = torch.tensor(np.asarray(train_y), dtype=torch.long)
    n = len(y)
    steps = epochs * ((n + batch_size - 1) // batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            batch = {k: v[b].to(dev) for k, v in enc.items()}
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = model(**batch, labels=y[b].to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    outs = []
    for ts in eval_sets:
        res = []
        for i in range(0, len(ts), 64):
            e = tok(list(ts[i:i + 64]), truncation=True, padding=True,
                    max_length=max_len, return_tensors="pt").to(dev)
            with torch.no_grad():
                lg = model(**e).logits.float()
            res.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
        outs.append(np.concatenate(res))
    del model
    torch.cuda.empty_cache()
    return outs


METHODS = {"lr": lr_logits, "bert": bert_logits}
