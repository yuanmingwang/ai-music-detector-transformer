#!/usr/bin/env python3
"""
download_real_songs.py

Download all real-song MP3 files for the dataset using the YouTube IDs
stored in dataset/real_songs.csv.

This script is designed to be robust for very large downloads:
1. It saves files into dataset/real_songs/
2. It uses a persistent yt-dlp download archive, so already completed songs
   are not downloaded again.
3. It keeps a JSONL log of successes and failures.
4. It skips files that already exist on disk.
5. It handles Ctrl+C cleanly, so you can stop the script and later resume it.
6. It uses "python -m yt_dlp" via sys.executable, which works nicely inside
   a project .venv and follows the same pattern used by the reference
   get_msd.py file.

Recommended usage:
    source .venv/bin/activate
    python -m pip install yt-dlp
    sudo apt install -y ffmpeg
    python download_real_songs.py

Optional usage:
    python download_real_songs.py --limit 100
    python download_real_songs.py --start-index 5000
    python download_real_songs.py --sleep 2.0
    python download_real_songs.py --cookies-from-browser chrome

Resume from row 40082 with a short delay between downloads:
    python scripts/download_real_songs.py --start-index 40082 --sleep 1.5

Notes:
- ffmpeg is required by yt-dlp for audio extraction/conversion to mp3.
- If some videos are unavailable, blocked, removed, or age-restricted,
  the script logs them and continues.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def now_iso() -> str:
    """Return a readable UTC timestamp for logs."""
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_mkdir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Dict) -> None:
    """
    Append one JSON object as a single line to a .jsonl file.

    JSONL is useful for large jobs because:
    - it is append-friendly,
    - it is readable,
    - it survives interruptions well,
    - it avoids rewriting the entire file every time.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    Read all rows from the metadata CSV into a list of dictionaries.

    We use DictReader so we can access columns by name, such as:
    - id
    - filename
    - youtube_id
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_youtube_url(youtube_id: str) -> str:
    """Create the full YouTube watch URL from a YouTube video ID."""
    return f"https://www.youtube.com/watch?v={youtube_id}"


def expected_output_path(row: Dict[str, str], out_dir: Path) -> Path:
    """
    Decide the final target path for one song.

    The metadata contains a 'filename' column for the real song file name.
    If that value exists, we use it directly. Otherwise we fall back to '<id>.mp3'.

    This is safer than always naming by YouTube ID because the dataset metadata
    is what the rest of the pipeline is likely to expect.
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
    Heuristic check for whether an existing file is likely a valid finished MP3.

    We do not fully validate audio integrity here. We only avoid obvious bad cases:
    - file missing
    - zero-byte or tiny file

    A small threshold helps ignore broken leftovers from interrupted runs.
    """
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def build_yt_dlp_command(
    *,
    youtube_url: str,
    output_template: str,
    archive_file: Path,
    cookies_from_browser: Optional[str] = None,
    retries: int = 10,
    fragment_retries: int = 10,
    socket_timeout: int = 30,
) -> List[str]:
    """
    Build the yt-dlp command.

    Important design choices:
    - use 'python -m yt_dlp' instead of 'yt-dlp' so the .venv interpreter is used
    - use '--download-archive' so completed YouTube IDs are remembered
    - extract audio and convert to mp3
    - use '--continue' and '--no-overwrites' for safer resume behavior
    - use '--newline' for cleaner progress output in terminals/logs
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

    if cookies_from_browser:
        # Examples: chrome, firefox, edge, brave
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
    dry_run: bool = False,
) -> bool:
    """
    Download exactly one song.

    Returns:
        True  -> the song is present after this call (downloaded or already existed)
        False -> the song is still unavailable / failed
    """
    song_id = (row.get("id") or "").strip()
    youtube_id = (row.get("youtube_id") or "").strip()
    title = (row.get("title") or "").strip()
    artist = (row.get("artist") or "").strip()

    # Basic validation before we attempt any network call.
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

    # Fast skip: if the file already exists and looks non-trivial, do not redownload.
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

    # yt-dlp output template:
    # We want an exact final target name from the CSV metadata.
    # yt-dlp expects something like "/path/to/name.%(ext)s"
    output_template = str(final_path.with_suffix("")) + ".%(ext)s"

    cmd = build_yt_dlp_command(
        youtube_url=youtube_url,
        output_template=output_template,
        archive_file=archive_file,
        cookies_from_browser=cookies_from_browser,
    )

    print(f"[DL  ] {title} - {artist} ({youtube_id})")
    print(f"       -> {final_path}")

    if dry_run:
        print("       dry-run command:", " ".join(cmd))
        return True

    try:
        # We do not capture stdout/stderr here because yt-dlp progress is useful live.
        # On failure, CalledProcessError gives us the return code.
        subprocess.run(cmd, check=True)

        # Verify final file exists after yt-dlp returns successfully.
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

        # Rare case: yt-dlp reported success but the expected file is not present.
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
    """
    Parse command-line arguments.

    Defaults are chosen to match the repo layout described in the README:
    - metadata CSV at dataset/real_songs.csv
    - output directory at dataset/real_songs/
    """
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
        help="Directory to store downloaded MP3 files (default: dataset/real_songs)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("dataset") / ".download_state",
        help="Directory to store resume/archive/log files (default: dataset/.download_state)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows after filtering (useful for testing)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this row index in the CSV (default: 0)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep this many seconds between downloads (default: 0.0)",
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
        help="Print what would be downloaded without actually downloading",
    )
    return parser.parse_args()


def main() -> int:
    """
    Main program flow.

    Resume behavior comes from three layers:
    1. Existing MP3 files are skipped immediately.
    2. yt-dlp uses a persistent download archive file to skip completed YouTube IDs.
    3. Success/failure logs record what happened for later inspection.
    """
    args = parse_args()

    csv_path: Path = args.csv
    out_dir: Path = args.out_dir
    state_dir: Path = args.state_dir

    archive_file = state_dir / "yt_dlp_archive.txt"
    # success_log = state_dir / "success.jsonl"
    # failure_log = state_dir / "failures.jsonl"
    run_stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    success_log = state_dir / f"failures_{run_stamp}.jsonl"
    failure_log = state_dir / f"success_{run_stamp}.jsonl"

    safe_mkdir(out_dir)
    safe_mkdir(state_dir)

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    # Helpful early check so users know why MP3 conversion might fail.
    # We do not hard-fail here because some yt-dlp setups can still work depending
    # on the source format, but for reliable mp3 conversion ffmpeg is strongly advised.
    ffmpeg_exists = shutil_which("ffmpeg") is not None
    if not ffmpeg_exists:
        print(
            "WARNING: ffmpeg was not found on PATH. "
            "yt-dlp usually needs ffmpeg to convert audio to mp3.\n"
            "Install it with: sudo apt install -y ffmpeg",
            file=sys.stderr,
        )

    rows = read_csv_rows(csv_path)

    total_rows = len(rows)
    if args.start_index < 0 or args.start_index >= total_rows:
        print(
            f"ERROR: --start-index {args.start_index} is outside valid range 0..{max(total_rows - 1, 0)}",
            file=sys.stderr,
        )
        return 1

    rows = rows[args.start_index:]

    if args.limit is not None:
        rows = rows[: args.limit]

    print("============================================================")
    print("real-song downloader")
    print("============================================================")
    print(f"CSV file:         {csv_path}")
    print(f"Output folder:    {out_dir}")
    print(f"State folder:     {state_dir}")
    print(f"Archive file:     {archive_file}")
    print(f"Success log:      {success_log}")
    print(f"Failure log:      {failure_log}")
    print(f"Rows in CSV:      {total_rows}")
    print(f"Rows to process:  {len(rows)}")
    print(f"Start index:      {args.start_index}")
    print(f"Dry run:          {args.dry_run}")
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
    """
    Small wrapper around shutil.which without importing shutil at top level only for this use.
    """
    import shutil
    return shutil.which(executable_name)


if __name__ == "__main__":
    raise SystemExit(main())