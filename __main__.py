# coding: utf-8
import os
import sys

# Must be set before any TensorBoard/TensorFlow import (breaks cuDNN if loaded late).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def _set_cuda_visible_devices():
    """Parse --gpu_id before torch import; CUDA_VISIBLE_DEVICES must be set early."""
    # Prefer already-set shell binding (multi-GPU launch scripts).
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu_id" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
        if arg.startswith("--gpu_id="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return


_set_cuda_visible_devices()

import argparse
from training import train, test


def main():
    ap = argparse.ArgumentParser(
        "Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production"
    )
    ap.add_argument("mode", choices=["train", "test"], help="train a model or test")
    ap.add_argument(
        "config_path",
        default="./Configs/Sign-IDD.yaml",
        type=str,
        help="path to YAML config file",
    )
    ap.add_argument("--ckpt", type=str, help="path to model checkpoint")
    ap.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")
    args = ap.parse_args()

    # Keep consistent with FGDM; if shell already bound a device, do not override.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.mode == "train":
        train(cfg_file=args.config_path, ckpt=args.ckpt)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
