"""Automated download, verification, and extraction of the PTB-XL ECG dataset."""

import os
import shutil
import zipfile
import urllib.request
import concurrent.futures
from pathlib import Path
from typing import Optional

from ptbxl_ecg.config import (
    PHYSIONET_PTBXL_URL,
    DEFAULT_DATA_DIR,
    DEFAULT_ZIP_NAME,
)


def download_ptbxl_archive(
    url: str = PHYSIONET_PTBXL_URL,
    destination_zip: Path = Path(DEFAULT_ZIP_NAME),
    num_threads: int = 4,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download PTB-XL zip archive using parallel HTTP Range chunk requests.

    Parameters:
        url: Remote endpoint URL.
        destination_zip: Output zip file path.
        num_threads: Concurrency worker thread count.
        chunk_size: Stream buffer size in bytes.

    Returns:
        Path to downloaded zip file.
    """
    dest_path = Path(destination_zip)
    if dest_path.exists() and dest_path.stat().st_size > 1000000:
        return dest_path

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    if total_size == 0:
        # Fallback to standard streaming download if server omits Content-Length
        urllib.request.urlretrieve(url, str(dest_path))
        return dest_path

    # Pre-allocate destination file
    with open(dest_path, "wb") as f:
        f.seek(total_size - 1)
        f.write(b"\0")

    part_size = total_size // num_threads
    ranges = []
    for i in range(num_threads):
        start = i * part_size
        end = (start + part_size - 1) if i < num_threads - 1 else total_size - 1
        ranges.append((start, end))

    def download_range(start_byte: int, end_byte: int):
        range_req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Range": f"bytes={start_byte}-{end_byte}",
            },
        )
        with urllib.request.urlopen(range_req, timeout=45) as resp, open(dest_path, "r+b") as fp:
            fp.seek(start_byte)
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                fp.write(buf)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(download_range, s, e) for s, e in ranges]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    return dest_path


def extract_ptbxl_archive(
    zip_path: Path,
    target_dir: Path = DEFAULT_DATA_DIR,
    remove_zip_after: bool = False,
) -> Path:
    """Extract and normalize directory hierarchy for PTB-XL records.

    Parameters:
        zip_path: Path to local zip archive.
        target_dir: Destination extraction folder.
        remove_zip_after: If True, deletes zip file after successful extraction.

    Returns:
        Path to normalized dataset directory.
    """
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
        zip_ref.extractall(str(out_dir))

    # Normalize nested folder if present (e.g. ptb-xl-1.0.3)
    nested_candidates = list(out_dir.glob("ptb-xl*"))
    for cand in nested_candidates:
        if cand.is_dir() and (cand / "ptbxl_database.csv").exists():
            for item in cand.iterdir():
                dest = out_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(out_dir))
            try:
                cand.rmdir()
            except OSError:
                pass

    if remove_zip_after and Path(zip_path).exists():
        Path(zip_path).unlink()

    return out_dir


def verify_ptbxl_integrity(data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """Verify presence of essential metadata CSVs and signal waveform folders.

    Parameters:
        data_dir: Root dataset folder.

    Returns:
        True if all verification checks pass.
    """
    root = Path(data_dir)
    has_db = (root / "ptbxl_database.csv").exists()
    has_scp = (root / "scp_statements.csv").exists()
    has_rec100 = (root / "records100").exists() or (root / "records500").exists()

    return bool(has_db and has_scp and has_rec100)
