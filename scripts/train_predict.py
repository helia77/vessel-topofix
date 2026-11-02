"""Train an nnU-Net on image/mask folders, then write mask, probability and GT per test image."""
import argparse
import os
import sys

import numpy as np
import skimage as ski

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topofix import nnunet

EXT = (".png", ".tif", ".tiff", ".jpg")
USAGE = """examples:
  python scripts/train_predict.py train   --images TRAIN_IMG --masks TRAIN_MASK --work WORK
  python scripts/train_predict.py predict --images TEST_IMG --work WORK --out OUT [--gt TEST_MASK]"""


def read(path):
    """First channel of an image file (the paper uses one channel)."""
    a = ski.io.imread(path)
    return a[..., 0] if a.ndim == 3 else a


def names(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXT))


def save_bin(path, M):
    ski.io.imsave(path, (np.asarray(M) > 0).astype(np.uint8) * 255, check_contrast=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("train", "predict"))
    ap.add_argument("--images", required=True, help="image folder")
    ap.add_argument("--masks", help="train: mask folder, same file names as --images")
    ap.add_argument("--gt", help="predict: optional GT mask folder, same file names, copied as gt_mask.png")
    ap.add_argument("--out", help="predict: output folder, one sub-folder per image")
    ap.add_argument("--work", required=True, help="nnU-Net work folder (raw, preprocessed, results)")
    ap.add_argument("--dataset-id", type=int, default=1)
    ap.add_argument("--fold", default="all", help='"all" (no validation split) or 0-4')
    ap.add_argument("--trainer", default=nnunet.PAPER_TRAINER)
    ap.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    model = nnunet.Model(a.work, dataset_id=a.dataset_id, fold=a.fold, trainer=a.trainer)
    files = names(a.images)
    if a.command == "train":
        if not a.masks:
            ap.error("train needs --masks")
        pairs = [(read(os.path.join(a.images, f)), read(os.path.join(a.masks, f))) for f in files]
        nnunet.train(pairs, model, a.device, a.retrain)
        print(f"trained on {len(pairs)} images -> {model.checkpoint}")
        return
    if not a.out:
        ap.error("predict needs --out")
    results = nnunet.predict([read(os.path.join(a.images, f)) for f in files], model, a.device)
    for f, (mask, prob) in zip(files, results):
        d = os.path.join(a.out, os.path.splitext(f)[0])
        os.makedirs(d, exist_ok=True)
        save_bin(os.path.join(d, "nnunet_mask.png"), mask)
        np.save(os.path.join(d, "prob_fg.npy"), prob)
        if a.gt:
            save_bin(os.path.join(d, "gt_mask.png"), read(os.path.join(a.gt, f)))
    print(f"wrote {len(files)} folders to {a.out}")


if __name__ == "__main__":
    main()
