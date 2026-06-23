from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AV_v2 import train_cremad, train_ks, train_ave
from AV_v2.datasets import load_ks_classes


def build_dataset_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train audio-video baselines on CREMA-D, KineticSound, or AVE.")
    parser.add_argument("--dataset", choices=["cremad", "ks", "ave"], required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--modality", choices=["av", "audio", "visual"], default="av")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--audio-duration", type=float, default=None)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--fgm", dest="fgm", action="store_true", help="Enable CMI-FGM gradient modulation for AV training.")
    parser.add_argument("--no-fgm", dest="fgm", action="store_false", help="Disable CMI-FGM gradient modulation.")
    parser.set_defaults(fgm=True)
    parser.add_argument("--fgm-lambda", type=float, default=0.5)
    parser.add_argument("--fgm-tau", type=float, default=1.0)
    parser.add_argument("--fgm-momentum", type=float, default=0.9)
    parser.add_argument("--fgm-warmup-steps", type=int, default=0)

    parser.add_argument("--split-csv-root", type=str, default="ICCV2025-GDL-main/dataset/data/CREMAD")
    parser.add_argument("--use-video-frames", type=int, default=None)
    parser.add_argument("--class-file", type=str, default="ICCV2025-GDL-main/dataset/data/KineticSound/class.txt")

    args = parser.parse_args(argv)

    if args.dataset == "cremad":
        args.data_root = args.data_root or "dataset/CREMA-D"
        args.output_dir = args.output_dir or "runs/video_cremad"
        args.num_classes = 6
        args.audio_duration = 3.0 if args.audio_duration is None else args.audio_duration
        args.n_fft = 512 if args.n_fft is None else args.n_fft
        args.hop_length = 160 if args.hop_length is None else args.hop_length
        args.win_length = 400 if args.win_length is None else args.win_length
        args.use_video_frames = 1 if args.use_video_frames is None else args.use_video_frames
        args.fps = args.use_video_frames
    elif args.dataset == "ks":
        args.data_root = args.data_root or "dataset/kinect_sound"
        args.output_dir = args.output_dir or "runs/video_ks"
        args.num_classes = len(load_ks_classes(args.class_file))
        args.audio_duration = 5.0 if args.audio_duration is None else args.audio_duration
        args.n_fft = 256 if args.n_fft is None else args.n_fft
        args.hop_length = 128 if args.hop_length is None else args.hop_length
        args.win_length = 256 if args.win_length is None else args.win_length
        args.use_video_frames = 3 if args.use_video_frames is None else args.use_video_frames
    else:  # ave
        args.data_root = args.data_root or "dataset/AVE"
        args.output_dir = args.output_dir or "runs/video_ave"
        args.num_classes = 28
        args.use_video_frames = 10 if args.use_video_frames is None else args.use_video_frames

    return args


def main() -> None:
    args = build_dataset_args()
    if args.dataset == "cremad":
        train_cremad.run_training(args)
    elif args.dataset == "ks":
        train_ks.run_training(args)
    else:
        train_ave.run_training(args)


if __name__ == "__main__":
    main()
