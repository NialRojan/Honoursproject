from pathlib import Path

from training.bert_finetune import finetune_bert_classifier

DATASET_PATH = Path(__file__).parent / "data" / "behavior_dataset.tsv"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "behavior_bert"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found — run `python -m training.prepare_behavior_dataset` first (Task 9)."
        )
    finetune_bert_classifier(
        dataset_tsv_path=str(DATASET_PATH),
        output_dir=str(OUTPUT_DIR),
        text_col="text",
        label_col="label",
    )


if __name__ == "__main__":
    main()
