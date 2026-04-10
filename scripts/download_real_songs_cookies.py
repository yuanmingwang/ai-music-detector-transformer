#!/usr/bin/env python3
"""
download_real_songs.py

Download real-song MP3 files using YouTube IDs from dataset/real_songs.csv.

This version is designed for large downloads and WSL use:
1. Saves MP3 files into dataset/real_songs/
2. Uses a persistent yt-dlp archive so completed songs are skipped
3. Uses per-run JSONL success/failure logs
4. Skips files already present on disk
5. Handles Ctrl+C cleanly so you can resume later
6. Supports either:
   - --cookies /path/to/youtube_cookies.txt   (recommended for WSL)
   - --cookies-from-browser chrome            (works only if browser exists in Linux env)

Recommended for your setup:
    python scripts/download_real_songs.py --cookies scripts/youtube_cookies.txt

Small test:
    python scripts/download_real_songs.py --limit 5 --cookies scripts/youtube_cookies.txt

Resume from row 113:
    python scripts/download_real_songs.py --start-index 113 --cookies scripts/youtube_cookies.txt

Resume from row 113 with a short delay between downloads:
    python scripts/download_real_songs.py --start-index 113 --sleep 1.5
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def now_iso() -> str:
    """Return a readable UTC timestamp for logs."""
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_mkdir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Dict) -> None:
    """Append one JSON object as one line to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read metadata CSV into a list of dictionaries."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_youtube_url(youtube_id: str) -> str:
    """Build full YouTube URL from a YouTube ID."""
    return f"https://www.youtube.com/watch?v={youtube_id}"


def expected_output_path(row: Dict[str, str], out_dir: Path) -> Path:
    """
    Decide the final output path for one song.

    Prefer the CSV 'filename' field, because that is usually what the dataset expects.
    If missing, fall back to id.mp3, then youtube_id.mp3.
    """
    filename = (row.get("filename") or "").strip()
    if filename:
        return out_dir / filename

    song_id = (row.get("id") or "").strip()
    if song_id:
        return out_dir / f"{song_id}.mp3"

    youtube_id = (row.get("youtube_id") or "").strip()
    return out_dir / f"{youtube_id}.mp3"


def file_looks_complete(path: Path, min_bytes: int = 16 * 1024) -> bool:
    """
    Heuristic check for a finished MP3 file.

    This avoids redownloading obviously valid files while ignoring tiny broken leftovers.
    """
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def resolve_default_cookies_path(user_value: Optional[Path]) -> Optional[Path]:
    """
    Resolve cookies file path.

    Priority:
    1. user-supplied --cookies path
    2. scripts/youtube_cookies.txt next to this script
    3. scripts/cookies.txt next to this script

    Returns None if no file is found.
    """
    if user_value is not None:
        return user_value

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "youtube_cookies.txt",
        script_dir / "cookies.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def build_yt_dlp_command(
    *,
    youtube_url: str,
    output_template: str,
    archive_file: Path,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[Path] = None,
    retries: int = 10,
    fragment_retries: int = 10,
    socket_timeout: int = 30,
) -> List[str]:
    """
    Build yt-dlp command.

    Important choices:
    - use 'python -m yt_dlp' so the current .venv is used
    - use '--download-archive' for resume behavior
    - extract audio as MP3
    - use '--continue' and '--no-overwrites'
    """
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--continue",
        "--no-overwrites",
        "--download-archive", str(archive_file),
        "--retries", str(retries),
        "--fragment-retries", str(fragment_retries),
        "--socket-timeout", str(socket_timeout),
        "--newline",
        "--output", output_template,
        youtube_url,
    ]

    if cookies_file is not None:
        cmd.extend(["--cookies", str(cookies_file)])

    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])

    return cmd


def download_one_song(
    *,
    row: Dict[str, str],
    out_dir: Path,
    archive_file: Path,
    success_log: Path,
    failure_log: Path,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[Path] = None,
    dry_run: bool = False,
) -> bool:
    """
    Download one song.

    Returns:
        True  -> song exists after this function
        False -> failed
    """
    song_id = (row.get("id") or "").strip()
    youtube_id = (row.get("youtube_id") or "").strip()
    title = (row.get("title") or "").strip()
    artist = (row.get("artist") or "").strip()

    if not youtube_id:
        append_jsonl(
            failure_log,
            {
                "time": now_iso(),
                "id": song_id,
                "youtube_id": youtube_id,
                "title": title,
                "artist": artist,
                "reason": "missing_youtube_id",
            },
        )
        return False

    final_path = expected_output_path(row, out_dir)

    # Skip existing valid file
    if file_looks_complete(final_path):
        append_jsonl(
            success_log,
            {
                "time": now_iso(),
                "id": song_id,
                "youtube_id": youtube_id,
                "title": title,
                "artist": artist,
                "status": "already_exists",
                "path": str(final_path),
            },
        )
        print(f"[SKIP] already exists: {final_path.name}")
        return True

    youtube_url = build_youtube_url(youtube_id)
    output_template = str(final_path.with_suffix("")) + ".%(ext)s"

    cmd = build_yt_dlp_command(
        youtube_url=youtube_url,
        output_template=output_template,
        archive_file=archive_file,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )

    print(f"[DL  ] {title} - {artist} ({youtube_id})")
    print(f"       -> {final_path}")

    if dry_run:
        print("       dry-run command:", " ".join(cmd))
        return True

    try:
        subprocess.run(cmd, check=True)

        if file_looks_complete(final_path):
            append_jsonl(
                success_log,
                {
                    "time": now_iso(),
                    "id": song_id,
                    "youtube_id": youtube_id,
                    "title": title,
                    "artist": artist,
                    "status": "downloaded",
                    "path": str(final_path),
                },
            )
            return True

        append_jsonl(
            failure_log,
            {
                "time": now_iso(),
                "id": song_id,
                "youtube_id": youtube_id,
                "title": title,
                "artist": artist,
                "reason": "missing_output_after_success",
                "path": str(final_path),
            },
        )
        return False

    except subprocess.CalledProcessError as e:
        append_jsonl(
            failure_log,
            {
                "time": now_iso(),
                "id": song_id,
                "youtube_id": youtube_id,
                "title": title,
                "artist": artist,
                "reason": "yt_dlp_failed",
                "returncode": e.returncode,
                "url": youtube_url,
                "path": str(final_path),
            },
        )
        print(f"[FAIL] yt-dlp failed for {youtube_id} (return code {e.returncode})")
        return False

    except Exception as e:
        append_jsonl(
            failure_log,
            {
                "time": now_iso(),
                "id": song_id,
                "youtube_id": youtube_id,
                "title": title,
                "artist": artist,
                "reason": "unexpected_exception",
                "error": repr(e),
                "url": youtube_url,
                "path": str(final_path),
            },
        )
        print(f"[FAIL] unexpected error for {youtube_id}: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download real songs from YouTube using youtube_id in real_songs.csv"
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset") / "real_songs.csv",
        help="Path to real_songs.csv (default: dataset/real_songs.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset") / "real_songs",
        help="Directory to store MP3 files (default: dataset/real_songs)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("dataset") / ".download_state",
        help="Directory for archive/log files (default: dataset/.download_state)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows after filtering",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this zero-based CSV row index",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep this many seconds between downloads",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Path to Netscape-format cookies.txt file. Recommended for WSL.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="Browser name for yt-dlp cookies, e.g. chrome, firefox, edge, brave",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    csv_path: Path = args.csv
    out_dir: Path = args.out_dir
    state_dir: Path = args.state_dir

    archive_file = state_dir / "yt_dlp_archive.txt"

    # Correct log file names
    run_stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    success_log = state_dir / f"success_{run_stamp}.jsonl"
    failure_log = state_dir / f"failures_{run_stamp}.jsonl"

    safe_mkdir(out_dir)
    safe_mkdir(state_dir)

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    ffmpeg_exists = shutil_which("ffmpeg") is not None
    if not ffmpeg_exists:
        print(
            "WARNING: ffmpeg was not found on PATH. "
            "yt-dlp usually needs ffmpeg to convert audio to mp3.\n"
            "Install it with: sudo apt install -y ffmpeg",
            file=sys.stderr,
        )

    cookies_file = resolve_default_cookies_path(args.cookies)
    if cookies_file is not None and not cookies_file.exists():
        print(f"ERROR: cookies file not found: {cookies_file}", file=sys.stderr)
        return 1

    rows = read_csv_rows(csv_path)

    total_rows = len(rows)
    if total_rows == 0:
        print("ERROR: CSV contains no rows.", file=sys.stderr)
        return 1

    if args.start_index < 0 or args.start_index >= total_rows:
        print(
            f"ERROR: --start-index {args.start_index} is outside valid range 0..{max(total_rows - 1, 0)}",
            file=sys.stderr,
        )
        return 1

    rows = rows[args.start_index:]

    if args.limit is not None:
        rows = rows[:args.limit]

    print("============================================================")
    print("real-song downloader")
    print("============================================================")
    print(f"CSV file:               {csv_path}")
    print(f"Output folder:          {out_dir}")
    print(f"State folder:           {state_dir}")
    print(f"Archive file:           {archive_file}")
    print(f"Success log:            {success_log}")
    print(f"Failure log:            {failure_log}")
    print(f"Rows in CSV:            {total_rows}")
    print(f"Rows to process:        {len(rows)}")
    print(f"Start index:            {args.start_index}")
    print(f"Cookies file:           {cookies_file}")
    print(f"Cookies from browser:   {args.cookies_from_browser}")
    print(f"Dry run:                {args.dry_run}")
    print("============================================================")

    ok_count = 0
    fail_count = 0

    try:
        for i, row in enumerate(rows, start=args.start_index):
            print(f"\n[{i + 1}/{total_rows}] Processing row index {i}")

            success = download_one_song(
                row=row,
                out_dir=out_dir,
                archive_file=archive_file,
                success_log=success_log,
                failure_log=failure_log,
                cookies_from_browser=args.cookies_from_browser,
                cookies_file=cookies_file,
                dry_run=args.dry_run,
            )

            if success:
                ok_count += 1
            else:
                fail_count += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
        print("Progress is preserved.")
        print("You can rerun the same command later to resume.")
        print(f"Partial summary: ok={ok_count}, failed={fail_count}")
        return 130

    print("\nDone.")
    print(f"Summary: ok={ok_count}, failed={fail_count}")
    return 0


def shutil_which(executable_name: str) -> Optional[str]:
    """Small wrapper around shutil.which."""
    import shutil
    return shutil.which(executable_name)


if __name__ == "__main__":
    raise SystemExit(main())