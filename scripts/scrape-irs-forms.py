#!/usr/bin/env python3
"""Scrape IRS Forms, Instructions, and Publications.

Downloads PDFs from https://www.irs.gov/forms-instructions-and-publications
with filtering by form type, year, and search term.

Usage:
    python scripts/scrape-irs-forms.py --list-only                      # list all 3000+ forms
    python scripts/scrape-irs-forms.py --search "W-2" --list-only       # find W-2 related forms
    python scripts/scrape-irs-forms.py --search "1040"                  # download 1040 forms
    python scripts/scrape-irs-forms.py --search "1099"                  # download 1099 forms
    python scripts/scrape-irs-forms.py --current-year-only              # only current revision year
    python scripts/scrape-irs-forms.py --max-forms 50                   # limit downloads
    python scripts/scrape-irs-forms.py --extract --extract-url http://10.10.100.20:8000
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


IRS_FORMS_URL = "https://www.irs.gov/forms-instructions-and-publications"
IRS_BASE_URL = "https://www.irs.gov"
DEFAULT_OUTPUT_DIR = "irs-forms"


def fetch_form_links(
    *,
    search: str | None = None,
    current_year_only: bool = False,
    max_pages: int = 130,
    per_page: int = 200,
) -> list[dict[str, str]]:
    """Scrape all form/publication PDF links from the IRS site."""
    forms: list[dict[str, str]] = []
    current_year = str(datetime.now().year)
    page = 0

    print("Scanning IRS forms and publications...")
    if search:
        print(f"  Search filter: '{search}'")
    if current_year_only:
        print(f"  Year filter: {current_year} only")

    with httpx.Client(
        timeout=30,
        follow_redirects=True,
        headers={"User-Agent": "loci-extract-scraper/1.0 (document research)"},
    ) as client:
        while page < max_pages:
            params: dict[str, str] = {
                "items_per_page": str(per_page),
                "page": str(page),
            }
            if search:
                params["find"] = search

            try:
                response = client.get(IRS_FORMS_URL, params=params)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  Error fetching page {page}: {exc}")
                break

            soup = BeautifulSoup(response.text, "html.parser")

            # Find the results table
            rows_found = 0
            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                # Extract form info from table cells
                link = cells[0].find("a", href=True)
                if not link:
                    continue

                href = link["href"]
                if not href.endswith(".pdf"):
                    continue

                product_number = cells[0].get_text(strip=True)
                title = cells[1].get_text(strip=True)
                revision_date = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                posted_date = cells[3].get_text(strip=True) if len(cells) > 3 else ""

                # Year filter
                if current_year_only:
                    if current_year not in revision_date and current_year not in posted_date:
                        continue

                full_url = urljoin(IRS_BASE_URL, href)
                filename = Path(href).name

                if not any(f["url"] == full_url for f in forms):
                    forms.append({
                        "product_number": product_number,
                        "title": title,
                        "revision_date": revision_date,
                        "posted_date": posted_date,
                        "filename": filename,
                        "url": full_url,
                    })
                    rows_found += 1

            if rows_found == 0:
                break

            # Check for next page
            next_link = soup.find("a", {"rel": "next"})
            if not next_link:
                # Also check for pagination links
                pager = soup.find("nav", {"class": "pager"}) or soup.find("ul", {"class": "pagination"})
                if not pager:
                    break
                next_a = pager.find("a", string=re.compile(r"Next|›|>>"))
                if not next_a:
                    break

            page += 1
            if page % 10 == 0:
                print(f"  Scanned {page} pages, found {len(forms)} forms so far...")
            time.sleep(0.3)  # Be polite

    forms.sort(key=lambda f: f["product_number"])
    return forms


def download_forms(
    forms: list[dict[str, str]],
    output_dir: Path,
    *,
    skip_existing: bool = True,
    max_forms: int | None = None,
) -> list[Path]:
    """Download form PDFs to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    to_download = forms[:max_forms] if max_forms else forms

    print(f"\nDownloading {len(to_download)} forms to {output_dir}/")

    with httpx.Client(
        timeout=60,
        follow_redirects=True,
        headers={"User-Agent": "loci-extract-scraper/1.0 (document research)"},
    ) as client:
        for i, form in enumerate(to_download, 1):
            filepath = output_dir / form["filename"]

            if skip_existing and filepath.exists():
                print(f"  [{i}/{len(to_download)}] {form['filename']} (exists, skipping)")
                downloaded.append(filepath)
                continue

            try:
                print(f"  [{i}/{len(to_download)}] {form['product_number']:20s} {form['filename']}...", end="", flush=True)
                response = client.get(form["url"])
                response.raise_for_status()
                filepath.write_bytes(response.content)
                size_mb = len(response.content) / (1024 * 1024)
                print(f" {size_mb:.1f} MB")
                downloaded.append(filepath)
                time.sleep(0.2)
            except httpx.HTTPError as exc:
                print(f" ERROR: {exc}")

    return downloaded


def extract_forms(
    files: list[Path],
    *,
    extract_url: str = "http://127.0.0.1:8000",
    profile: str = "tax",
    output_dir: Path | None = None,
) -> None:
    """Send downloaded PDFs to loci-extract for processing."""
    output_dir = output_dir or files[0].parent / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {len(files)} forms via {extract_url} (profile={profile})...")

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


def save_manifest(forms: list[dict[str, str]], output_dir: Path) -> None:
    """Save a JSON manifest of all scraped forms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(forms, indent=2))
    print(f"\nManifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape IRS Forms, Instructions, and Publications",
    )
    parser.add_argument("--search", type=str, help="Search/filter term (e.g. 'W-2', '1040', '1099')")
    parser.add_argument("--current-year-only", action="store_true", help="Only include current year revisions")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--max-forms", type=int, help="Maximum number of forms to download")
    parser.add_argument("--extract", action="store_true", help="Also extract via loci-extract after downloading")
    parser.add_argument("--extract-url", type=str, default="http://127.0.0.1:8000", help="loci-extract API URL")
    parser.add_argument("--profile", type=str, default="tax", help="Extraction profile (default: tax)")
    parser.add_argument("--list-only", action="store_true", help="Only list forms, don't download")

    args = parser.parse_args()
    output_dir = Path(args.output)
    if args.search:
        output_dir = output_dir / args.search.lower().replace(" ", "-")

    # Find forms
    forms = fetch_form_links(
        search=args.search,
        current_year_only=args.current_year_only,
    )

    if not forms:
        print("No forms found matching criteria")
        sys.exit(1)

    print(f"\nFound {len(forms)} forms/publications:")
    for f in forms[:30]:
        print(f"  {f['product_number']:20s} {f['revision_date']:12s} {f['title'][:60]}")
    if len(forms) > 30:
        print(f"  ... and {len(forms) - 30} more")

    if args.list_only:
        save_manifest(forms, output_dir)
        return

    # Download
    files = download_forms(forms, output_dir / "pdfs", max_forms=args.max_forms)
    save_manifest(forms, output_dir)
    print(f"\n{len(files)} files in {output_dir}/pdfs/")

    # Extract
    if args.extract:
        extract_forms(
            files,
            extract_url=args.extract_url,
            profile=args.profile,
            output_dir=output_dir / "extracted",
        )


if __name__ == "__main__":
    main()
