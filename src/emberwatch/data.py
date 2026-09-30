"""D-Fire download, conversion to a small local cache, and a torch dataset.

D-Fire (Venancio et al., 2022) is distributed by the authors via OneDrive, which cannot be
scripted. Its content is mirrored as parquet shards on Hugging Face (badsaarow/d-fire), which is
what `emberwatch fetch` uses. Each shard is downloaded, converted to 320x320 JPEGs plus
normalised boxes, and the raw shard is deleted, so peak extra disk use stays around 0.5 GB.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HF_BASE = "https://huggingface.co/datasets/badsaarow/d-fire/resolve/main/data"
SHARDS = {"train": 9, "test": 3}
IMG_SIZE = 320
# Raw D-Fire convention (verified on the data, see docs/results.md): 0 = smoke, 1 = fire.
RAW_TO_NAME = {0: "smoke", 1: "fire"}
CLASSES = ["fire", "smoke"]  # model label = index + 1 (0 is background)
DATA_DIR = Path("data")


def shard_url(split: str, i: int) -> str:
    return f"{HF_BASE}/{split}-{i:05d}-of-{SHARDS[split]:05d}.parquet"


def parse_yolo(text: str) -> list[tuple[str, list[float]]]:
    """YOLO txt (cls cx cy w h, normalised) -> [(name, [x1, y1, x2, y2] normalised)]."""
    out = []
    for line in text.strip().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        box = [min(max(v, 0.0), 1.0) for v in box]
        if box[2] - box[0] > 1e-3 and box[3] - box[1] > 1e-3:
            out.append((RAW_TO_NAME[c], box))
    return out


def is_val(filename: str) -> bool:
    """Deterministic ~5% validation carve-out from the official train split."""
    return int(hashlib.md5(filename.encode()).hexdigest(), 16) % 20 == 0


def fetch(data_dir: Path = DATA_DIR, splits: tuple[str, ...] = ("train", "test")) -> None:
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "labels.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    done = set(meta.get("_shards", []))
    meta.setdefault("items", {})
    for split in splits:
        for i in range(SHARDS[split]):
            key = f"{split}-{i}"
            if key in done:
                continue
            tmp = data_dir / f"_{key}.parquet"
            print(f"downloading {key} ...", flush=True)
            urllib.request.urlretrieve(shard_url(split, i), tmp)
            pf = pq.ParquetFile(tmp)
            for batch in pf.iter_batches(batch_size=64):
                for r in batch.to_pylist():
                    name = r["filename"]
                    im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                    im.save(img_dir / name, quality=90)
                    sp = "val" if split == "train" and is_val(name) else split
                    meta["items"][name] = {"split": sp, "boxes": parse_yolo(r["label"])}
            tmp.unlink()
            done.add(key)
            meta["_shards"] = sorted(done)
            meta_path.write_text(json.dumps(meta))
    print(f"{len(meta['items'])} images cached in {img_dir}")


def load_meta(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    return json.loads((data_dir / "labels.json").read_text())["items"]


def split_items(split: str, data_dir: Path = DATA_DIR) -> list[tuple[str, list]]:
    items = load_meta(data_dir)
    return sorted((k, v["boxes"]) for k, v in items.items() if v["split"] == split)


def load_image(name: str, data_dir: Path = DATA_DIR) -> np.ndarray:
    return np.asarray(Image.open(data_dir / "images" / name).convert("RGB"))


def to_target(boxes: list, size: int = IMG_SIZE):
    import torch

    if boxes:
        b = torch.tensor([bx for _, bx in boxes], dtype=torch.float32) * size
        lab = torch.tensor([CLASSES.index(n) + 1 for n, _ in boxes], dtype=torch.int64)
    else:
        b, lab = torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)
    return {"boxes": b, "labels": lab}


class DFireDataset:
    def __init__(self, items, data_dir: Path = DATA_DIR, train: bool = False):
        self.items, self.data_dir, self.train = items, data_dir, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch

        name, boxes = self.items[i]
        img = load_image(name, self.data_dir)
        boxes = [(n, list(b)) for n, b in boxes]
        if self.train and np.random.rand() < 0.5:  # horizontal flip
            img = img[:, ::-1].copy()
            boxes = [(n, [1 - b[2], b[1], 1 - b[0], b[3]]) for n, b in boxes]
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t, to_target(boxes)


def collate(batch):
    return tuple(zip(*batch, strict=True))
