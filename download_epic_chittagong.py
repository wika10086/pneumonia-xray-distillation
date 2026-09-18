from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


DATASET_API_URL = "https://data.mendeley.com/public-api/datasets/wndbd5r26y"
DATASET_PAGE_URL = "https://data.mendeley.com/datasets/wndbd5r26y"
SOURCE_NAME = "epic_chittagong"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {"normal", "pneumonia"}
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/json,application/octet-stream,*/*",
}


@dataclass
class KnownImage:
    path: str
    class_name: str
    sha256: str
    dhash: str


@dataclass
class ImportRow:
    source_path: str
    target_path: str
    class_name: str
    status: str
    reason: str
    sha256: str
    dhash: str
    duplicate_path: str = ""
    hamming_distance: int | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dhash(path: Path) -> int:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8), resampling)
        if hasattr(image, "get_flattened_data"):
            pixels = list(image.get_flattened_data())
        else:
            pixels = list(image.getdata())

    value = 0
    for row in range(8):
        row_start = row * 9
        for col in range(8):
            value <<= 1
            if pixels[row_start + col] > pixels[row_start + col + 1]:
                value |= 1
    return value


def iter_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def get_dataset_metadata() -> dict:
    request = urllib.request.Request(DATASET_API_URL, headers=HTTP_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def select_zip_file(metadata: dict) -> dict:
    zip_files = [
        file_info
        for file_info in metadata.get("files", [])
        if file_info.get("filename", "").lower().endswith(".zip")
    ]
    if not zip_files:
        raise RuntimeError("No zip file was found in the dataset metadata.")
    return max(zip_files, key=lambda item: item.get("size", 0))


def download_file(url: str, target_path: Path, expected_sha256: str, expected_size: int) -> bool:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        current_hash = sha256_file(target_path)
        if current_hash == expected_sha256:
            print(f"Download skipped, existing zip passed SHA256 check: {target_path}")
            return False
        print("Existing zip hash did not match; downloading a fresh copy.")

    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    downloaded = 0
    last_reported_mb = -1
    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
            downloaded += len(chunk)
            downloaded_mb = downloaded // (50 * 1024 * 1024)
            if downloaded_mb != last_reported_mb:
                last_reported_mb = downloaded_mb
                print(f"Downloaded {downloaded / (1024 * 1024):.1f} MB")

    if expected_size and temp_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Downloaded file size mismatch: got {temp_path.stat().st_size}, expected {expected_size}"
        )

    actual_sha256 = sha256_file(temp_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Downloaded file SHA256 mismatch: {actual_sha256} != {expected_sha256}")

    if target_path.exists():
        target_path.unlink()
    temp_path.replace(target_path)
    print(f"Download completed and verified: {target_path}")
    return True


def extract_zip(zip_path: Path, extract_dir: Path, expected_sha256: str) -> bool:
    marker_path = extract_dir / ".source_sha256"
    if marker_path.exists() and marker_path.read_text(encoding="utf-8").strip() == expected_sha256:
        print(f"Extraction skipped, existing folder matches this zip: {extract_dir}")
        return False

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    marker_path.write_text(expected_sha256 + "\n", encoding="utf-8")
    print(f"Extraction completed: {extract_dir}")
    return True


def detect_class(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    for part in reversed(relative.parts[:-1]):
        normalized = part.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in CLASS_NAMES:
            return normalized
    return None


def build_known_index(data_dir: Path) -> tuple[dict[str, KnownImage], list[KnownImage], list[str]]:
    sha_index: dict[str, KnownImage] = {}
    hash_index: list[KnownImage] = []
    errors: list[str] = []

    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        for image_path in iter_images(split_dir):
            class_name = image_path.parent.name.lower()
            try:
                file_sha = sha256_file(image_path)
                file_dhash = image_dhash(image_path)
            except (OSError, UnidentifiedImageError) as exc:
                errors.append(f"{image_path}: {exc}")
                continue

            known = KnownImage(
                path=str(image_path),
                class_name=class_name,
                sha256=file_sha,
                dhash=f"{file_dhash:016x}",
            )
            sha_index[file_sha] = known
            hash_index.append(known)

    return sha_index, hash_index, errors


def find_near_duplicate(
    dhash_value: int,
    known_hashes: list[KnownImage],
    threshold: int,
) -> tuple[KnownImage | None, int | None]:
    best_match: KnownImage | None = None
    best_distance: int | None = None

    for known in known_hashes:
        distance = (dhash_value ^ int(known.dhash, 16)).bit_count()
        if distance <= threshold and (best_distance is None or distance < best_distance):
            best_match = known
            best_distance = distance
            if distance == 0:
                break

    return best_match, best_distance


def copy_unique_images(
    source_root: Path,
    data_dir: Path,
    import_split: str,
    duplicate_threshold: int,
    dry_run: bool,
) -> tuple[list[ImportRow], list[str]]:
    sha_index, hash_index, errors = build_known_index(data_dir)
    rows: list[ImportRow] = []
    imported_counter = {class_name: 0 for class_name in CLASS_NAMES}

    for source_path in sorted(iter_images(source_root)):
        class_name = detect_class(source_path, source_root)
        if class_name not in CLASS_NAMES:
            rows.append(
                ImportRow(
                    source_path=str(source_path),
                    target_path="",
                    class_name="",
                    status="skipped",
                    reason="class_not_detected",
                    sha256="",
                    dhash="",
                )
            )
            continue

        try:
            file_sha = sha256_file(source_path)
            file_dhash = image_dhash(source_path)
        except (OSError, UnidentifiedImageError) as exc:
            rows.append(
                ImportRow(
                    source_path=str(source_path),
                    target_path="",
                    class_name=class_name,
                    status="skipped",
                    reason=f"cannot_read_image: {exc}",
                    sha256="",
                    dhash="",
                )
            )
            continue

        exact_duplicate = sha_index.get(file_sha)
        if exact_duplicate is not None:
            rows.append(
                ImportRow(
                    source_path=str(source_path),
                    target_path="",
                    class_name=class_name,
                    status="skipped",
                    reason="exact_duplicate",
                    sha256=file_sha,
                    dhash=f"{file_dhash:016x}",
                    duplicate_path=exact_duplicate.path,
                    hamming_distance=0,
                )
            )
            continue

        near_duplicate, distance = find_near_duplicate(file_dhash, hash_index, duplicate_threshold)
        if near_duplicate is not None:
            rows.append(
                ImportRow(
                    source_path=str(source_path),
                    target_path="",
                    class_name=class_name,
                    status="skipped",
                    reason="near_duplicate",
                    sha256=file_sha,
                    dhash=f"{file_dhash:016x}",
                    duplicate_path=near_duplicate.path,
                    hamming_distance=distance,
                )
            )
            continue

        imported_counter[class_name] += 1
        extension = source_path.suffix.lower()
        target_name = f"{SOURCE_NAME}_{class_name}_{imported_counter[class_name]:05d}_{file_sha[:10]}{extension}"
        target_path = data_dir / import_split / class_name / target_name

        if not dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

        known = KnownImage(
            path=str(target_path),
            class_name=class_name,
            sha256=file_sha,
            dhash=f"{file_dhash:016x}",
        )
        sha_index[file_sha] = known
        hash_index.append(known)

        rows.append(
            ImportRow(
                source_path=str(source_path),
                target_path=str(target_path),
                class_name=class_name,
                status="imported" if not dry_run else "would_import",
                reason="unique",
                sha256=file_sha,
                dhash=f"{file_dhash:016x}",
            )
        )

    return rows, errors


def save_report(
    rows: list[ImportRow],
    errors: list[str],
    output_dir: Path,
    metadata: dict,
    zip_path: Path,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_stem = "epic_chittagong_dry_run" if args.dry_run else "epic_chittagong_import"
    report_path = output_dir / f"{report_stem}_report.json"
    csv_path = output_dir / f"{report_stem}_rows.csv"

    summary: dict[str, int] = {}
    class_summary: dict[str, dict[str, int]] = {}
    for row in rows:
        summary[row.status] = summary.get(row.status, 0) + 1
        if row.class_name:
            class_summary.setdefault(row.class_name, {})
            class_summary[row.class_name][row.status] = class_summary[row.class_name].get(row.status, 0) + 1

    report = {
        "source": {
            "name": metadata.get("name"),
            "doi": metadata.get("doi", {}).get("id"),
            "page_url": DATASET_PAGE_URL,
            "api_url": DATASET_API_URL,
            "zip_path": str(zip_path),
        },
        "settings": {
            "data_dir": str(args.data_dir),
            "import_split": args.import_split,
            "duplicate_threshold": args.duplicate_threshold,
            "dry_run": args.dry_run,
        },
        "summary": summary,
        "class_summary": class_summary,
        "index_errors": errors,
        "rows": [asdict(row) for row in rows],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()) if rows else list(ImportRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Epic Chittagong chest X-ray dataset and import only non-duplicate images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--import-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--duplicate-threshold", type=int, default=2)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir or (args.data_dir / "raw" / SOURCE_NAME)
    extract_dir = raw_dir / "extracted"

    print("Fetching dataset metadata...")
    metadata = get_dataset_metadata()
    zip_info = select_zip_file(metadata)
    content_details = zip_info["content_details"]
    zip_path = raw_dir / zip_info["filename"]

    download_file(
        url=content_details["download_url"],
        target_path=zip_path,
        expected_sha256=content_details["sha256_hash"],
        expected_size=int(content_details["size"]),
    )
    extract_zip(zip_path, extract_dir, content_details["sha256_hash"])

    if args.download_only:
        print("Download-only mode finished.")
        return 0

    print("Building duplicate index and importing unique images...")
    rows, errors = copy_unique_images(
        source_root=extract_dir,
        data_dir=args.data_dir,
        import_split=args.import_split,
        duplicate_threshold=args.duplicate_threshold,
        dry_run=args.dry_run,
    )
    report_path = save_report(rows, errors, args.output_dir, metadata, zip_path, args)

    imported = sum(1 for row in rows if row.status == "imported")
    would_import = sum(1 for row in rows if row.status == "would_import")
    skipped = sum(1 for row in rows if row.status == "skipped")
    print(f"Imported: {imported}")
    print(f"Would import: {would_import}")
    print(f"Skipped: {skipped}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
