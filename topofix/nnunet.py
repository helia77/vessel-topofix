"""Thin nnU-Net v2 wrappers: train on (image, mask) pairs, predict a binary mask and foreground probability.

nnU-Net is driven through its command-line tools; all of its files live under one work directory."""
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import skimage as ski

PAPER_TRAINER = "nnUNetTrainer_50epochs"  # trainer used in the paper


def default_device():
    """'cuda' if available, then 'mps', else 'cpu'."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _env(work_dir):
    """Environment with nnU-Net's three folder variables set, and this interpreter's bin on PATH."""
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    for key in ("raw", "preprocessed", "results"):
        env[f"nnUNet_{key}"] = os.path.join(work_dir, f"nnUNet_{key}")
    return env


def _save(arr, path):
    """Save a 2D or RGB array as an 8-bit PNG (floats in [0, 1] are scaled to 255)."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = (arr * 255).round() if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1 else arr
        arr = arr.astype(np.uint8)
    ski.io.imsave(path, arr, check_contrast=False)


@dataclass
class Model:
    """Which trained nnU-Net to use; the weights stay on disk under work_dir."""
    work_dir: str
    dataset_id: int = 1
    dataset_name: str = "Vessels"
    configuration: str = "2d"
    fold: int | str = "all"
    trainer: str = PAPER_TRAINER

    @property
    def folder(self):
        return f"Dataset{self.dataset_id:03d}_{self.dataset_name}"

    @property
    def checkpoint(self):
        return os.path.join(self.work_dir, "nnUNet_results", self.folder,
                            f"{self.trainer}__nnUNetPlans__{self.configuration}", f"fold_{self.fold}",
                            "checkpoint_final.pth")

    def channels(self):
        """Number of input channels, from the dataset.json saved next to the weights."""
        with open(os.path.join(os.path.dirname(os.path.dirname(self.checkpoint)), "dataset.json")) as fh:
            return len(json.load(fh)["channel_names"])


def train(pairs, model, device=None, retrain=False):
    """Train model on (image, mask) array pairs; mask > 0 is vessel. Skipped if a checkpoint exists.

    fold "all" trains on every pair without a validation split; a numeric fold needs at least 5 pairs."""
    model.work_dir = os.path.abspath(model.work_dir)
    env = _env(model.work_dir)
    ds = os.path.join(env["nnUNet_raw"], model.folder)
    for sub in ("imagesTr", "labelsTr"):
        shutil.rmtree(os.path.join(ds, sub), ignore_errors=True)
        os.makedirs(os.path.join(ds, sub))
    for i, (image, mask) in enumerate(pairs):
        _save(image, os.path.join(ds, "imagesTr", f"case_{i:04d}_0000.png"))
        _save((np.asarray(mask) > 0).astype(np.uint8), os.path.join(ds, "labelsTr", f"case_{i:04d}.png"))
    rgb = np.asarray(pairs[0][0]).ndim == 3
    with open(os.path.join(ds, "dataset.json"), "w") as fh:
        json.dump(dict(channel_names={"0": "R", "1": "G", "2": "B"} if rgb else {"0": "grayscale"},
                       labels={"background": 0, "vessel": 1}, numTraining=len(pairs), file_ending=".png"), fh, indent=2)
    subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", str(model.dataset_id), "--verify_dataset_integrity",
                    "-c", model.configuration], env=env, check=True)
    if retrain or not os.path.exists(model.checkpoint):
        subprocess.run(["nnUNetv2_train", str(model.dataset_id), model.configuration, str(model.fold),
                        "-tr", model.trainer, "-device", device or default_device()], env=env, check=True)
    return model


def predict(images, model, device=None):
    """Segment a list of images in one nnU-Net call; returns [(binary mask, foreground probability)]."""
    model.work_dir = os.path.abspath(model.work_dir)
    if not os.path.exists(model.checkpoint):
        raise FileNotFoundError(f"no finished checkpoint at {model.checkpoint}")
    n_ch = model.channels()
    env = _env(model.work_dir)
    src, dst = (os.path.join(model.work_dir, d) for d in ("predict_in", "predict_out"))
    for d in (src, dst):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    for i, image in enumerate(images):
        image = np.asarray(image)
        if (image.shape[2] if image.ndim == 3 else 1) != n_ch:
            raise ValueError(f"image {i} has shape {image.shape}, the model expects {n_ch} channel(s)")
        _save(image, os.path.join(src, f"case_{i:04d}_0000.png"))
    subprocess.run(["nnUNetv2_predict", "-i", src, "-o", dst, "-d", str(model.dataset_id), "-c", model.configuration,
                    "-f", str(model.fold), "-tr", model.trainer, "-device", device or default_device(),
                    "--save_probabilities"], env=env, check=True)
    out = []
    for i in range(len(images)):
        lab = ski.io.imread(os.path.join(dst, f"case_{i:04d}.png"))
        prob = np.load(os.path.join(dst, f"case_{i:04d}.npz"))["probabilities"][1, 0]  # (classes, 1, H, W)
        out.append(((lab[..., 0] if lab.ndim == 3 else lab) > 0, prob.astype(np.float32)))
    return out
