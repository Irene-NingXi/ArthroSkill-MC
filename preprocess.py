"""
Quality-screened video preprocessing entry point
Usage: python preprocess.py --input ./videos --output ./processed
"""
import argparse
from utils import VideoPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess arthroscopic surgery videos")
    parser.add_argument("--input", "-i", required=True, help="Raw video directory")
    parser.add_argument("--output", "-o", default="./processed", help="Output directory")
    parser.add_argument("--clip-len", type=int, default=16, help="Clip length")
    parser.add_argument("--fps", type=int, default=25, help="Expected frame rate")
    parser.add_argument("--brightness", type=int, default=30, help="Brightness threshold")
    parser.add_argument("--blur", type=int, default=100, help="Blur threshold")
    args = parser.parse_args()

    preprocessor = VideoPreprocessor(
        clip_len=args.clip_len,
        fps=args.fps,
        brightness_thresh=args.brightness,
        blur_thresh=args.blur
    )

    total = preprocessor.process_directory(args.input, args.output)
    print(f"\nPreprocessing complete: {total} clips")


if __name__ == "__main__":
    main()
