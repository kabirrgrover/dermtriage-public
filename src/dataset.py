"""
Dataset classes for HAM10000 and PAD-UFES-20.

All paths are parameterized — no hardcoded Colab paths.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)

# Maps PAD-UFES-20 diagnostic labels to HAM10000 class names
PADUFES_TO_HAM = {"ACK": "akiec", "BCC": "bcc", "MEL": "mel", "NEV": "nv", "SEK": "bkl"}


class HAM10000Dataset(Dataset):
    """HAM10000 dermoscopic image dataset.

    Expects directory layout::

        data_dir/
          train/
            akiec/*.jpg
            bcc/*.jpg
            ...
          val/
            akiec/*.jpg
            ...
    """

    def __init__(self, data_dir, split="train", transform=None):
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.samples = []

        split_dir = Path(data_dir) / split
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append(
                        {"path": img_path, "label": self.class_to_idx[class_name], "class_name": class_name}
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample["label"]


class PADUFESDataset(Dataset):
    """PAD-UFES-20 clinical smartphone image dataset.

    Includes Fitzpatrick skin type metadata for fairness evaluation.
    Only keeps classes that map to HAM10000 via ``PADUFES_TO_HAM``.
    """

    def __init__(self, data_dir, split="train", transform=None, val_ratio=0.2):
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.data_dir = Path(data_dir)

        # Find metadata CSV
        metadata_path = None
        for p in self.data_dir.rglob("*.csv"):
            metadata_path = p
            break
        if metadata_path is None:
            raise FileNotFoundError(f"No CSV found in {data_dir}")

        df = pd.read_csv(metadata_path)
        diag_col = "diagnostic" if "diagnostic" in df.columns else "dx"
        df = df[df[diag_col].isin(PADUFES_TO_HAM.keys())].copy()
        df["ham_class"] = df[diag_col].map(PADUFES_TO_HAM)
        df["label"] = df["ham_class"].map(self.class_to_idx)

        train_df, val_df = train_test_split(df, test_size=val_ratio, stratify=df["ham_class"], random_state=42)
        self.df = train_df if split == "train" else val_df

        # Find image directories (those with >10 .png files)
        self.img_dirs = []
        for d in self.data_dir.rglob("*"):
            if d.is_dir() and len(list(d.glob("*.png"))) > 10:
                self.img_dirs.append(d)

        img_id_col = "img_id" if "img_id" in df.columns else "image_id"
        fitz_col = "fitspatrick" if "fitspatrick" in df.columns else None

        self.samples = []
        for _, row in self.df.iterrows():
            self.samples.append(
                {
                    "img_id": row[img_id_col],
                    "label": row["label"],
                    "class_name": row["ham_class"],
                    "fitzpatrick": row[fitz_col] if fitz_col else None,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample["img_id"]
        img_id_base = img_id.rsplit(".", 1)[0] if img_id.endswith((".png", ".jpg")) else img_id

        img_path = None
        for img_dir in self.img_dirs:
            for ext in [".png", ".PNG", ".jpg"]:
                candidate = img_dir / f"{img_id_base}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                candidate = img_dir / img_id
                if candidate.exists():
                    img_path = candidate
            if img_path:
                break

        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_id}")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample["label"]

    def get_fitzpatrick_groups(self):
        """Return sample indices grouped by Fitzpatrick skin type."""
        groups = {"I-II": [], "III-IV": [], "V-VI": [], "unknown": []}
        for idx, sample in enumerate(self.samples):
            fitz = sample["fitzpatrick"]
            if fitz in [1, 2]:
                groups["I-II"].append(idx)
            elif fitz in [3, 4]:
                groups["III-IV"].append(idx)
            elif fitz in [5, 6]:
                groups["V-VI"].append(idx)
            else:
                groups["unknown"].append(idx)
        return groups
