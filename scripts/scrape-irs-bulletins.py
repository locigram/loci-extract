#!/usr/bin/env python3
"""Scrape IRS Internal Revenue Bulletins for the current year.

Downloads all PDF bulletins from https://www.irs.gov/internal-revenue-bulletins
for the specified year into a local directory.

Usage:
    python scripts/scrape-irs-bulletins.py                  # current year
    python scripts/scrape-irs-bulletins.py --year 2025      # specific year
    python scripts/scrape-irs-bulletins.py --output ./pdfs   # custom output dir
    python scripts/scrape-irs-bulletins.py --extract         # download + extract via loci-extract
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup


IRS_BULLETINS_URL = "https://www.irs.gov/internal-revenue-bulletins"
IRS_BASE_URL = "https://www.irs.gov"
DEFAULT_OUTPUT_DIR = "irs-bulletins"


def fetch_bulletin_links(year: int, *, max_pages: int = 70) -> list[dict[str, str]]:
    """Scrape all bulletin PDF links for a given year from the IRS site."""
    bulletins: list[dict[str, str]] = []
    year_short = str(year)[-2:]
    page = 0

    print(f"Scanning IRS bulletins for year {year} (irb{year_short}-XX.pdf)...")

    with httpx.Client(
        timeout=30,
        follow_redirects=True,
        headers={"User-Agent": "loci-extract-scraper/1.0 (document research)"},
    ) as client:
        while page < max_pages:
            url = f"{IRS_BULLETINS_URL}?page={page}" if page > 0 else IRS_BULLETINS_URL
            try:
                response = client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  Error fetching page {page}: {exc}")
                break

            soup = BeautifulSoup(response.text, "html.parser")

            # Find all links to PDF files
            found_on_page = 0
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if f"irb{year_short}-" in href and href.endswith(".pdf"):
                    full_url = urljoin(IRS_BASE_URL, href)
                    name = link.get_text(strip=True) or Path(href).name
                    filename = Path(href).name

                    if not any(b["url"] == full_url for b in bulletins):
                        bulletins.append({
                            "name": name,
                            "filename": filename,
                            "url": full_url,
                        })
                        found_on_page += 1

            # Check if we've gone past the target year (bulletins are newest-first)
            # If no bulletins found on this page, we might be past our year
            all_links_on_page = [
                link["href"] for link in soup.find_all("a", href=True)
                if "irb" in link["href"] and link["href"].endswith(".pdf")
            ]

            # Check if any links on this page are for years before our target
            older_year_found = False
            for href in all_links_on_page:
                match = re.search(r"irb(\d{2})-", href)
                if match:
                    link_year = int(match.group(1))
                    target_year = int(year_short)
                    if link_year < target_year:
                        older_year_found = True
                        break

            if older_year_found and found_on_page == 0:
                # We've gone past our target year
                break

            # Check for next page
            next_link = soup.find("a", {"rel": "next"}) or soup.find("a", string=re.compile(r"Next|›"))
            if not next_link:
                break

            page += 1
            time.sleep(0.5)  # Be polite

    # Sort by bulletin number
    bulletins.sort(key=lambda b: b["filename"])
    return bulletins


def download_bulletins(
    bulletins: list[dict[str, str]],
    output_dir: Path,
    *,
    skip_existing: bool = True,
) -> list[Path]:
    """Download bulletin PDFs to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    print(f"\nDownloading {len(bulletins)} bulletins to {output_dir}/")

    with httpx.Client(
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": "loci-extract-scraper/1.0 (document research)"},
    ) as client:
        for i, bulletin in enumerate(bulletins, 1):
            filepath = output_dir / bulletin["filename"]

            if skip_existing and filepath.exists():
                print(f"  [{i}/{len(bulletins)}] {bulletin['filename']} (exists, skipping)")
                downloaded.append(filepath)
                continue

            try:
                print(f"  [{i}/{len(bulletins)}] Downloading {bulletin['filename']}...", end="", flush=True)
                response = client.get(bulletin["url"])
                response.raise_for_status()
                filepath.write_bytes(response.content)
                size_mb = len(response.content) / (1024 * 1024)
                print(f" {size_mb:.1f} MB")
                downloaded.append(filepath)
                time.sleep(0.3)  # Be polite
            except httpx.HTTPError as exc:
                print(f" ERROR: {exc}")

    return downloaded


def extract_bulletins(
    files: list[Path],
    *,
    extract_url: str = "http://127.0.0.1:8000",
    profile: str = "general",
    output_dir: Path | None = None,
) -> None:
    """Send downloaded PDFs to loci-extract for processing."""
    output_dir = output_dir or files[0].parent / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {len(files)} bulletins via {extract_url} (profile={profile})...")

    with httpx.Client(timeout=300) as client:
        for i, filepath in enumerate(files, 1):
            json_path = output_dir / filepath.with_suffix(".json").name

            if json_path.exists():
                print(f"  [{i}/{len(files)}] {filepath.name} (extracted, skipping)")
                continue

            try:
                print(f"  [{i}/{len(files)}] Extracting {filepath.name}...", end="", flush=True)
                with filepath.open("rb") as f:
                    response = client.post(
                        f"{extract_url}/extract/structured",
                        files={"file": (filepath.name, f, "application/pdf")},
                        data={"extraction_profile": profile},
                    )
                response.raise_for_status()
                result = response.json()
                json_path.write_text(json.dumps(result, indent=2))
                doc_type = result.get("classification", {}).get("doc_type", "?")
                text_len = len(result.get("raw_extraction", result).get("raw_text", ""))
                print(f" {doc_type} ({text_len} chars)")
            except Exception as exc:
                print(f" ERROR: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape IRS Internal Revenue Bulletins for a given year",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now().year,
        help="Year to scrape (default: current year)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Also extract via loci-extract after downloading",
    )
    parser.add_argument(
        "--extract-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="loci-extract API URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="general",
        help="Extraction profile to use (default: general)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list bulletins, don't download",
    )

    args = parser.parse_args()
    output_dir = Path(args.output) / str(args.year)

    # Find bulletins
    bulletins = fetch_bulletin_links(args.year)

    if not bulletins:
        print(f"No bulletins found for year {args.year}")
        sys.exit(1)

    print(f"\nFound {len(bulletins)} bulletins for {args.year}:")
    for b in bulletins:
        print(f"  {b['filename']:20s}  {b['name']}")

    if args.list_only:
        return

    # Download
    files = download_bulletins(bulletins, output_dir)
    print(f"\n{len(files)} files in {output_dir}/")

    # Extract
    if args.extract:
        extract_bulletins(
            files,
            extract_url=args.extract_url,
            profile=args.profile,
            output_dir=output_dir / "extracted",
        )


if __name__ == "__main__":
    main()
