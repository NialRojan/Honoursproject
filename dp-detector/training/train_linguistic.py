import urllib.request
from pathlib import Path

from training.bert_finetune import finetune_bert_classifier

DATA_DIR = Path(__file__).parent / "data"
DATASET_URL = "https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/dataset/dataset.tsv"
DATASET_PATH = DATA_DIR / "linguistic_dataset.tsv"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "linguistic_bert"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    finetune_bert_classifier(
        dataset_tsv_path=str(DATASET_PATH),
        output_dir=str(OUTPUT_DIR),
        text_col="text",
        label_col="label",
    )


if __name__ == "__main__":
    main()
