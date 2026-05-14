"""Download the UCI Individual Household Electric Power Consumption dataset.

Usage:
    python scripts/download_data.py
    make data

Behavior:
    1. Read the source URL and (optional) expected SHA-256 from ``conf/base.yaml``.
    2. Download the ZIP archive to ``data/raw/`` with a progress bar.
    3. Extract the data file from the zip.
    4. Compute SHA-256 of the extracted file:
       - If the config has no hash yet (trust-on-first-use), pin it back to YAML.
       - If the config has a hash, refuse to proceed on mismatch (catches
         silent dataset changes).
    5. Skip work entirely if the file already exists with the right hash.

Exit codes:
    0  Success or already present
    1  Network error, hash mismatch, or unexpected zip structure
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import yaml
from tqdm import tqdm

from energy_forecasting.config import DEFAULT_CONFIG_PATH, ForecastConfig
from energy_forecasting.utils import compute_sha256, configure_logging, ensure_dir, get_logger
from energy_forecasting.utils.io import atomic_write_yaml

_logger = get_logger(__name__)


def main() -> int:
    """Entry point. Returns POSIX exit code."""
    configure_logging(level="INFO")
    cfg = ForecastConfig.from_yaml(DEFAULT_CONFIG_PATH)

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = ensure_dir(project_root / "data" / "raw")
    target_file = raw_dir / cfg.data.raw_filename
    zip_path = raw_dir / "_household_power.zip"

    # ── Fast path: file already present ────────────────────────
    if target_file.exists():
        existing_hash = compute_sha256(target_file)
        _logger.info(f"Raw data already present: {target_file}")
        _logger.info(f"SHA-256: {existing_hash}")
        if cfg.data.expected_sha256 and existing_hash != cfg.data.expected_sha256:
            _logger.error(
                f"Existing file hash {existing_hash[:16]}… does not match "
                f"expected {cfg.data.expected_sha256[:16]}… — refusing to overwrite. "
                "Delete the file manually if you want to re-download."
            )
            return 1
        if cfg.data.expected_sha256 is None:
            _logger.info("Pinning existing file's hash to conf/base.yaml")
            _update_config_hash(DEFAULT_CONFIG_PATH, existing_hash)
        return 0

    # ── Download ───────────────────────────────────────────────
    _logger.info(f"Downloading from {cfg.data.source_url}")
    try:
        _stream_download(cfg.data.source_url, zip_path)
    except URLError as e:
        _logger.error(f"Network error downloading {cfg.data.source_url}: {e}")
        return 1

    # ── Extract ────────────────────────────────────────────────
    _logger.info(f"Extracting to {raw_dir}")
    try:
        data_member = _extract_data_file(zip_path, target_file)
    except (zipfile.BadZipFile, KeyError) as e:
        _logger.error(f"Failed to extract zip: {e}")
        return 1

    _logger.info(f"Extracted {data_member} → {target_file}")
    zip_path.unlink(missing_ok=True)

    # ── Verify / pin hash ──────────────────────────────────────
    actual_hash = compute_sha256(target_file)
    _logger.info(f"SHA-256: {actual_hash}")

    if cfg.data.expected_sha256 is None:
        _logger.info("First download — pinning hash to conf/base.yaml")
        _update_config_hash(DEFAULT_CONFIG_PATH, actual_hash)
    elif actual_hash != cfg.data.expected_sha256:
        _logger.error(
            f"Downloaded file hash {actual_hash} does not match "
            f"expected {cfg.data.expected_sha256}. "
            "Possible dataset change or corruption — file NOT trusted."
        )
        target_file.unlink(missing_ok=True)
        return 1

    _logger.info(f"Dataset ready at {target_file}")
    return 0


def _stream_download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a progress bar."""
    with urlopen(url) as response:  # nosec B310 - URL is from validated config
        total = int(response.headers.get("Content-Length", "0"))
        with dest.open("wb") as fh, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as pbar:
            for chunk in iter(lambda: response.read(8192), b""):
                fh.write(chunk)
                pbar.update(len(chunk))


def _extract_data_file(zip_path: Path, target_file: Path) -> str:
    """Find the data file inside the zip and extract it to ``target_file``.

    Returns the name of the member that was extracted.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # UCI sometimes nests the file or uses different naming. Be lenient.
        data_member = None
        for name in names:
            lower = name.lower()
            if lower.endswith(".txt") and ("household" in lower or "power" in lower):
                data_member = name
                break
        if data_member is None and len(names) == 1 and names[0].endswith(".txt"):
            data_member = names[0]
        if data_member is None:
            raise KeyError(f"Could not locate data .txt in zip members: {names}")

        with zf.open(data_member) as source, target_file.open("wb") as target:
            target.write(source.read())
        return data_member


def _update_config_hash(config_path: Path, hash_value: str) -> None:
    """Atomically pin a SHA-256 into ``data.expected_sha256`` in the YAML config."""
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data["data"]["expected_sha256"] = hash_value
    atomic_write_yaml(config_path, data)


if __name__ == "__main__":
    sys.exit(main())
