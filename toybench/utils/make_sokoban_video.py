import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime


def find_latest_sokoban_attempt(results_dir: str) -> str | None:
    if not os.path.isdir(results_dir):
        return None
    # Find sokoban run directories
    runs = [
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.lower().startswith("sokoban_")
    ]
    if not runs:
        return None
    runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    # Pick the newest run and newest attempt_* within it
    for run_dir in runs:
        attempts = [
            os.path.join(run_dir, a)
            for a in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, a)) and a.lower().startswith("attempt_")
        ]
        if attempts:
            attempts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return attempts[0]
    return None


def collect_frames(attempt_dir: str) -> list[str]:
    # Match Sokoban screenshot pattern and extract iteration index for sorting
    pattern = os.path.join(attempt_dir, "sokoban_screenshot_iteration_*.png")
    files = glob.glob(pattern)
    def key_fn(p: str) -> tuple[int, str]:
        m = re.search(r"iteration_(\d+)\.png$", p)
        idx = int(m.group(1)) if m else -1
        return (idx, p)
    files.sort(key=key_fn)
    return files


def ensure_ffmpeg() -> str | None:
    # 1) Try system ffmpeg
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if out.returncode == 0:
            return "ffmpeg"
    except FileNotFoundError:
        pass

    # 2) Try imageio-ffmpeg if available
    try:
        import imageio_ffmpeg  # type: ignore
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    return None


def write_concat_list(frames: list[str], list_path: str):
    # Use concat demuxer, one line per file with proper quoting
    with open(list_path, 'w', encoding='utf-8') as f:
        for frame in frames:
            # Use forward slashes for portability; ffmpeg on Windows accepts them
            rel = os.path.relpath(frame, os.path.dirname(list_path)).replace('\\', '/')
            f.write(f"file '{rel}'\n")


def build_video(ffmpeg_bin: str, attempt_dir: str, frames: list[str], fps: int, out_path: str) -> tuple[bool, str]:
    list_file = os.path.join(attempt_dir, "frames.txt")
    write_concat_list(frames, list_file)
    vf = f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-vf", vf,
        "-vsync", "vfr",
        "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = (proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0)
    logs = proc.stderr or proc.stdout
    return ok, logs


def main():
    parser = argparse.ArgumentParser(description="Create a Sokoban timelapse video from attempt screenshots.")
    parser.add_argument("--attempt_dir", default=None, help="Path to an attempt directory containing Sokoban screenshots.")
    parser.add_argument("--results_dir", default="results", help="Base results directory (used if --attempt_dir not provided).")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for the video.")
    parser.add_argument("--output", default=None, help="Output video path (default: attempt_dir/sokoban_timelapse.mp4)")
    args = parser.parse_args()

    attempt_dir = args.attempt_dir or find_latest_sokoban_attempt(args.results_dir)
    if not attempt_dir:
        print("Error: Could not locate an attempt directory with Sokoban results.")
        sys.exit(1)

    frames = collect_frames(attempt_dir)
    if not frames:
        print(f"Error: No Sokoban screenshots found in {attempt_dir}.")
        sys.exit(1)

    out_path = args.output or os.path.join(attempt_dir, "sokoban_timelapse.mp4")

    ffmpeg_bin = ensure_ffmpeg()
    if not ffmpeg_bin:
        print("Error: ffmpeg not available. Install system ffmpeg or 'pip install imageio-ffmpeg'.")
        sys.exit(2)

    ok, logs = build_video(ffmpeg_bin, attempt_dir, frames, args.fps, out_path)
    if not ok:
        print("Error: ffmpeg failed. Logs:\n" + logs)
        sys.exit(3)

    print(json.dumps({
        "attempt_dir": attempt_dir,
        "output": out_path,
        "frames": len(frames),
        "fps": args.fps
    }))


if __name__ == "__main__":
    main()
