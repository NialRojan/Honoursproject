import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch


class _TsvTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def finetune_bert_classifier(dataset_tsv_path: str, output_dir: str, text_col: str, label_col: str,
                              base_model: str = "bert-base-uncased", epochs: int = 3) -> None:
    df = pd.read_csv(dataset_tsv_path, sep="\t")
    df = df.dropna(subset=[text_col, label_col])
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[label_col])

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    train_ds = _TsvTextDataset(train_df[text_col], train_df[label_col], tokenizer)
    val_ds = _TsvTextDataset(val_df[text_col], val_df[label_col], tokenizer)

    args = TrainingArguments(
        output_dir=output_dir + "_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
