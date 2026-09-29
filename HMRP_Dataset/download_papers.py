"""
================================================
HMRP Dataset Harvester & PDF Downloader
================================================
Harvests candidate research papers across 15 multidisciplinary domains from arXiv.
Enforces:
  - Exact 50 valid papers per domain (Total: 750 valid papers)
  - Duplicate detection (arXiv ID, normalized title, cross-domain prevention)
  - Strict PDF validation (PyMuPDF / fitz, valid stream, page count > 0)
  - Text validation (word count >= MIN_FULLTEXT_WORDS, abstract words >= MIN_ABSTRACT_WORDS)
  - Resumable state persistence
  - Rate-limit handling & exponential backoff
  - Structured reporting & CSV exports
"""

import os
import sys
import time
import re
import csv
import json
import argparse
import unicodedata
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from urllib.parse import quote

import requests
import feedparser

# Optional PyMuPDF
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HMRP_Dataset.config import (
    DOMAINS,
    TARGET_PER_DOMAIN,
    MIN_TITLE_WORDS,
    MIN_ABSTRACT_WORDS,
    MIN_FULLTEXT_WORDS,
    DOMAIN_DIR_NAMES,
)

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "PDFs"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [PDF_DIR, DATA_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logger Setup ──────────────────────────────────────────────────────────────
log_file = LOGS_DIR / "dataset_harvesting.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("HMRP_Harvester")

HEADERS = {
    "User-Agent": "HMRP-Academic-Research/2.0 (mailto:academic-research@university.edu)"
}


# ── Title Normalization & Deduplication ──────────────────────────────────────
def normalize_title(title: str) -> str:
    """
    Normalizes a paper title for deterministic duplicate detection.
    Performs Unicode normalization (NFKD), replaces hyphens and slashes with spaces,
    strips non-alphanumerics, and collapses whitespace.
    """
    if not title:
        return ""
    title = unicodedata.normalize("NFKD", title)
    title = re.sub(r"[-_/]+", " ", title)
    title = re.sub(r"[^a-zA-Z0-9\s]", "", title).lower()
    return " ".join(title.split())


# ── PDF Validation ────────────────────────────────────────────────────────────
def validate_pdf_file(filepath: Path) -> Tuple[bool, str, int, int]:
    """
    Validates PDF file integrity and extracted text.
    Returns:
        (is_valid, reason, page_count, word_count)
    """
    if not filepath.exists():
        return False, "file_not_found", 0, 0

    file_size = filepath.stat().st_size
    if file_size < 1024:
        return False, "file_too_small", 0, 0

    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
            if header != b"%PDF":
                return False, "invalid_pdf_header", 0, 0
    except Exception as e:
        return False, f"file_read_error_{e}", 0, 0

    page_count = 0
    full_text = ""

    if HAS_FITZ:
        try:
            doc = fitz.open(str(filepath))
            page_count = len(doc)
            if page_count == 0:
                doc.close()
                return False, "empty_pdf_no_pages", 0, 0

            for page in doc:
                full_text += page.get_text("text") + " "
            doc.close()
        except Exception as e:
            return False, f"fitz_open_error_{e}", 0, 0
    else:
        # Fallback reading
        try:
            import pdfplumber
            with pdfplumber.open(str(filepath)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    full_text += (page.extract_text() or "") + " "
        except Exception as e:
            return False, f"pdfplumber_error_{e}", 0, 0

    words = full_text.split()
    word_count = len(words)

    if word_count < MIN_FULLTEXT_WORDS:
        return False, f"insufficient_text_{word_count}_words", page_count, word_count

    return True, "valid", page_count, word_count


# ── Harvester Pipeline Class ──────────────────────────────────────────────────
class HMRPDatasetHarvester:
    """
    Manages the multi-domain paper harvesting, deduplication, downloading,
    verification, and reporting pipeline for HMRP.
    """

    def __init__(self, target_per_domain: int = TARGET_PER_DOMAIN):
        self.target = target_per_domain
        self.papers_json_path = DATA_DIR / "papers.json"
        self.papers_csv_path = DATA_DIR / "papers.csv"
        self.duplicates_csv_path = REPORTS_DIR / "DUPLICATES.csv"
        self.rejected_csv_path = REPORTS_DIR / "REJECTED_PAPERS.csv"
        self.domain_stats_csv_path = REPORTS_DIR / "DOMAIN_STATISTICS.csv"
        self.report_md_path = REPORTS_DIR / "DATASET_REPORT.md"

        self.accepted_papers: List[Dict[str, Any]] = []
        self.global_seen_ids: Set[str] = set()
        self.global_seen_titles: Set[str] = set()
        self.duplicates_log: List[Dict[str, str]] = []
        self.rejected_log: List[Dict[str, str]] = []

        self._load_existing_state()

    def _load_existing_state(self):
        """Loads existing accepted papers to support seamless resume."""
        if self.papers_json_path.exists():
            try:
                with open(self.papers_json_path, "r", encoding="utf-8") as f:
                    self.accepted_papers = json.load(f)
                logger.info(f"Loaded {len(self.accepted_papers)} existing accepted papers from state.")
            except Exception as e:
                logger.warning(f"Could not load papers.json: {e}")
                self.accepted_papers = []

        # Populate deduplication sets
        for p in self.accepted_papers:
            if p.get("arxiv_id"):
                self.global_seen_ids.add(p["arxiv_id"])
            if p.get("title"):
                norm = normalize_title(p["title"])
                if norm:
                    self.global_seen_titles.add(norm)

    def _get_domain_folder(self, domain: str) -> Path:
        folder_name = DOMAIN_DIR_NAMES.get(domain, domain.replace(" ", "_").replace("&", "and"))
        folder_path = PDF_DIR / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def _save_state_and_reports(self):
        """Atomically saves papers.json, papers.csv, and summary logs."""
        # 1. Save papers.json
        with open(self.papers_json_path, "w", encoding="utf-8") as f:
            json.dump(self.accepted_papers, f, indent=2, ensure_ascii=False)

        # 2. Save papers.csv
        if self.accepted_papers:
            fields = [
                "paper_id", "domain", "arxiv_id", "title", "authors",
                "year", "page_count", "word_count", "pdf_path", "categories"
            ]
            with open(self.papers_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for p in self.accepted_papers:
                    row = dict(p)
                    if isinstance(row.get("authors"), list):
                        row["authors"] = ", ".join(row["authors"])
                    if isinstance(row.get("categories"), list):
                        row["categories"] = ", ".join(row["categories"])
                    writer.writerow(row)

        # 3. Save DUPLICATES.csv
        if self.duplicates_log:
            with open(self.duplicates_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["arxiv_id", "title", "domain", "reason", "timestamp"])
                writer.writeheader()
                writer.writerows(self.duplicates_log)

        # 4. Save REJECTED_PAPERS.csv
        if self.rejected_log:
            with open(self.rejected_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["arxiv_id", "title", "domain", "reason", "timestamp"])
                writer.writeheader()
                writer.writerows(self.rejected_log)

    def scan_and_revalidate_disk(self):
        """Scans disk PDF folders, removes corrupt files, and registers valid ones."""
        logger.info("Scanning existing PDF directory for valid papers...")
        valid_by_domain: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DOMAINS}
        
        # Build existing index
        existing_lookup = {p["arxiv_id"]: p for p in self.accepted_papers if "arxiv_id" in p}

        for domain in DOMAINS:
            folder = self._get_domain_folder(domain)
            for pdf_file in sorted(folder.glob("*.pdf")):
                arxiv_id = pdf_file.stem
                is_valid, reason, pages, words = validate_pdf_file(pdf_file)

                if not is_valid:
                    logger.warning(f"Removing invalid PDF on disk: {pdf_file.name} (Reason: {reason})")
                    try:
                        pdf_file.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete {pdf_file.name}: {e}")
                    continue

                if arxiv_id in existing_lookup:
                    paper_data = existing_lookup[arxiv_id]
                    paper_data["domain"] = domain
                    paper_data["pdf_path"] = str(pdf_file)
                    paper_data["page_count"] = pages
                    paper_data["word_count"] = words
                    valid_by_domain[domain].append(paper_data)
                else:
                    # Generic record from disk
                    paper_data = {
                        "paper_id": f"arxiv_{arxiv_id.replace('.', '_')}",
                        "title": arxiv_id,
                        "authors": [],
                        "abstract": "",
                        "domain": domain,
                        "arxiv_id": arxiv_id,
                        "categories": DOMAINS[domain],
                        "published_date": "",
                        "updated_date": "",
                        "pdf_path": str(pdf_file),
                        "text_path": "",
                        "page_count": pages,
                        "word_count": words,
                        "year": "",
                    }
                    valid_by_domain[domain].append(paper_data)

        # Merge unique validated papers
        new_accepted = []
        self.global_seen_ids.clear()
        self.global_seen_titles.clear()

        for domain, plist in valid_by_domain.items():
            for p in plist:
                aid = p["arxiv_id"]
                norm = normalize_title(p["title"])
                if aid not in self.global_seen_ids:
                    self.global_seen_ids.add(aid)
                    if norm and norm != normalize_title(aid):
                        self.global_seen_titles.add(norm)
                    new_accepted.append(p)

        self.accepted_papers = new_accepted
        self._save_state_and_reports()
        logger.info(f"Revalidation complete. Total valid papers on disk: {len(self.accepted_papers)}")

    def fetch_arxiv_batch(self, categories: List[str], start: int, max_results: int = 50) -> Optional[List[Any]]:
        """Queries arXiv API with retry and rate-limiting exponential backoff."""
        cat_query = " OR ".join([f"cat:{c}" for c in categories])
        if len(categories) > 1:
            query_str = f"({cat_query})"
        else:
            query_str = cat_query

        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={quote(query_str)}"
            f"&start={start}"
            f"&max_results={max_results}"
            "&sortBy=submittedDate"
            "&sortOrder=descending"
        )

        for attempt in range(1, 6):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=45)
                if resp.status_code == 429:
                    wait_time = 15 * attempt
                    logger.warning(f"arXiv Rate Limit (429). Backing off {wait_time}s (Attempt {attempt}/5)...")
                    time.sleep(wait_time)
                    continue

                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
                return feed.entries
            except Exception as e:
                wait_time = 10 * attempt
                logger.warning(f"arXiv API error: {e}. Retrying in {wait_time}s (Attempt {attempt}/5)...")
                time.sleep(wait_time)

        logger.error(f"Failed to fetch batch from arXiv after 5 attempts: start={start}")
        return None

    def download_pdf_with_backoff(self, arxiv_id: str, candidate_urls: List[str], target_path: Path) -> bool:
        """Downloads PDF from candidate URLs with timeout and validation."""
        for url in candidate_urls:
            for attempt in range(1, 4):
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
                    if resp.status_code == 429:
                        time.sleep(15 * attempt)
                        continue

                    if resp.status_code == 200 and resp.content.startswith(b"%PDF"):
                        with open(target_path, "wb") as f:
                            f.write(resp.content)
                        return True
                except Exception as ex:
                    time.sleep(3)
        return False

    def harvest_domain(self, domain: str) -> Dict[str, Any]:
        """Harvests papers for a specific domain until TARGET_PER_DOMAIN is reached."""
        categories = DOMAINS[domain]
        folder = self._get_domain_folder(domain)

        # Count current valid papers in this domain
        domain_papers = [p for p in self.accepted_papers if p.get("domain") == domain]
        valid_count = len(domain_papers)

        logger.info("=" * 70)
        logger.info(f"DOMAIN: {domain}")
        logger.info(f"Categories: {categories}")
        logger.info(f"Target: {self.target} | Existing Valid: {valid_count} | Needed: {max(0, self.target - valid_count)}")
        logger.info(f"Folder: {folder}")
        logger.info("=" * 70)

        if valid_count >= self.target:
            logger.info(f"Domain '{domain}' already has {valid_count}/{self.target} valid papers. Skipping.")
            return {"domain": domain, "valid": valid_count, "downloaded": 0, "rejected": 0}

        start_offset = 0
        batch_size = 50
        downloaded_count = 0
        rejected_count = 0
        candidates_seen = 0

        while valid_count < self.target and start_offset < 3000:
            entries = self.fetch_arxiv_batch(categories, start=start_offset, max_results=batch_size)
            if entries is None or len(entries) == 0:
                logger.warning(f"No entries returned at offset {start_offset}. Advancing offset.")
                start_offset += batch_size
                time.sleep(5)
                continue

            for entry in entries:
                if valid_count >= self.target:
                    break

                candidates_seen += 1
                raw_id = entry.id.split("/")[-1]
                arxiv_id = raw_id.split("v")[0] if "v" in raw_id else raw_id
                arxiv_id = arxiv_id.replace("/", "_")

                title = " ".join(entry.title.split())
                norm_title = normalize_title(title)
                abstract = " ".join(entry.summary.split()) if hasattr(entry, "summary") else ""
                authors = [a.name for a in entry.authors] if hasattr(entry, "authors") else []
                entry_cats = [t.get("term") for t in getattr(entry, "tags", []) if t.get("term")]

                # 1. Primary Duplicate Check (arXiv ID)
                if arxiv_id in self.global_seen_ids:
                    self.duplicates_log.append({
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "domain": domain,
                        "reason": "duplicate_arxiv_id",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    continue

                # 2. Title Normalization Duplicate Check
                if norm_title and norm_title in self.global_seen_titles:
                    self.duplicates_log.append({
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "domain": domain,
                        "reason": "duplicate_title",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    continue

                # 3. Metadata Validation
                title_words = len(title.split())
                abstract_words = len(abstract.split())

                if title_words < MIN_TITLE_WORDS:
                    self.rejected_log.append({
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "domain": domain,
                        "reason": f"title_too_short_{title_words}_words",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    rejected_count += 1
                    continue

                if abstract_words < MIN_ABSTRACT_WORDS:
                    self.rejected_log.append({
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "domain": domain,
                        "reason": f"abstract_too_short_{abstract_words}_words",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    rejected_count += 1
                    continue

                # 4. Download PDF
                pdf_filename = f"{arxiv_id}.pdf"
                pdf_filepath = folder / pdf_filename

                candidate_urls = [
                    f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    f"https://export.arxiv.org/pdf/{arxiv_id}.pdf",
                ]
                for link in getattr(entry, "links", []):
                    if link.get("type") == "application/pdf" and link.get("href"):
                        candidate_urls.insert(0, link["href"])

                logger.info(f"[{valid_count + 1}/{self.target}] Downloading {arxiv_id} for '{domain}': {title[:65]}...")
                download_success = self.download_pdf_with_backoff(arxiv_id, candidate_urls, pdf_filepath)

                if not download_success:
                    logger.warning(f"  ✗ Download failed for {arxiv_id}")
                    self.rejected_log.append({
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "domain": domain,
                        "reason": "download_failed",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    rejected_count += 1
                    time.sleep(3)
                    continue

                # 5. Strict PDF Validation & Full-text Extraction
                is_valid, val_reason, pages, words = validate_pdf_file(pdf_filepath)

                if not is_valid:
                    logger.warning(f"  ✗ PDF validation failed for {arxiv_id}: {val_reason}")
                    try:
                        pdf_filepath.unlink()
                    except Exception:
                        pass
                    self.rejected_log.append({
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "domain": domain,
                        "reason": f"pdf_invalid_{val_reason}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    rejected_count += 1
                    time.sleep(3)
                    continue

                # 6. Accept Paper
                self.global_seen_ids.add(arxiv_id)
                if norm_title:
                    self.global_seen_titles.add(norm_title)

                paper_record = {
                    "paper_id": f"arxiv_{arxiv_id.replace('.', '_')}",
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "domain": domain,
                    "arxiv_id": arxiv_id,
                    "categories": entry_cats if entry_cats else categories,
                    "published_date": getattr(entry, "published", ""),
                    "updated_date": getattr(entry, "updated", ""),
                    "pdf_path": str(pdf_filepath),
                    "text_path": "",
                    "page_count": pages,
                    "word_count": words,
                    "year": entry.published[:4] if hasattr(entry, "published") and len(entry.published) >= 4 else "",
                }

                self.accepted_papers.append(paper_record)
                valid_count += 1
                downloaded_count += 1
                logger.info(f"  ✓ Accepted {arxiv_id} ({pages} pages, {words} words). Total: {valid_count}/{self.target}")

                # Save intermediate progress
                if valid_count % 5 == 0 or valid_count == self.target:
                    self._save_state_and_reports()

                # Polite rate limiting between PDF downloads
                time.sleep(3.0)

            start_offset += batch_size
            time.sleep(4.0)

        self._save_state_and_reports()
        return {
            "domain": domain,
            "valid": valid_count,
            "downloaded": downloaded_count,
            "rejected": rejected_count,
            "candidates": candidates_seen,
        }

    def run_harvesting(self, specific_domain: Optional[str] = None) -> None:
        """Executes full harvesting for all or a specified domain."""
        logger.info(f"Starting HMRP Multidisciplinary Dataset Harvesting. Target: {self.target} papers/domain.")
        domains_to_run = [specific_domain] if specific_domain and specific_domain in DOMAINS else list(DOMAINS.keys())

        for domain in domains_to_run:
            self.harvest_domain(domain)
            time.sleep(3)

        self.generate_reports()

    def generate_reports(self) -> Dict[str, Any]:
        """Generates domain distribution table, statistics, and DATASET_REPORT.md."""
        domain_counts = {d: 0 for d in DOMAINS}
        total_pages = 0
        total_words = 0
        abstract_word_counts = []
        page_counts = []
        word_counts = []

        for p in self.accepted_papers:
            d = p.get("domain", "Unknown")
            if d in domain_counts:
                domain_counts[d] += 1
            pg = p.get("page_count", 0)
            wd = p.get("word_count", 0)
            ab_words = len(p.get("abstract", "").split())

            total_pages += pg
            total_words += wd
            page_counts.append(pg)
            word_counts.append(wd)
            abstract_word_counts.append(ab_words)

        total_valid = len(self.accepted_papers)
        target_total = len(DOMAINS) * self.target
        is_complete = total_valid >= target_total and all(cnt >= self.target for cnt in domain_counts.values())
        status_str = "COMPLETE" if is_complete else "INCOMPLETE"

        avg_pages = (total_pages / total_valid) if total_valid > 0 else 0.0
        avg_words = (total_words / total_valid) if total_valid > 0 else 0.0
        min_words = min(word_counts) if word_counts else 0
        max_words = max(word_counts) if word_counts else 0
        avg_abs_len = (sum(abstract_word_counts) / len(abstract_word_counts)) if abstract_word_counts else 0.0

        # Save DOMAIN_STATISTICS.csv
        with open(self.domain_stats_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Domain", "Target", "Valid_Papers", "ArXiv_Categories", "Status"])
            for d in DOMAINS:
                cnt = domain_counts.get(d, 0)
                d_status = "REACHED" if cnt >= self.target else "PENDING"
                writer.writerow([d, self.target, cnt, ", ".join(DOMAINS[d]), d_status])

        # Generate markdown report
        report_content = f"""# 📊 HMRP Multidisciplinary Dataset Report

**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Status:** `{status_str}`

---

## 1. Dataset Summary

| Metric | Target | Actual Value |
| :--- | :--- | :--- |
| **Total Research Domains** | {len(DOMAINS)} | {len(DOMAINS)} |
| **Target Papers per Domain** | {self.target} | {self.target} |
| **Total Expected Papers** | **{target_total}** | **{total_valid}** |
| **Corpus Completeness** | 100.0% | **{(total_valid / target_total * 100):.1f}%** |
| **Total Rejection / Error Events** | - | **{len(self.rejected_log)}** |
| **Duplicates Detected & Prevented** | - | **{len(self.duplicates_log)}** |

---

## 2. Domain Distribution

| Domain | arXiv Category | Target | Valid Count | Completion |
| :--- | :--- | :--- | :--- | :--- |
"""
        for d in DOMAINS:
            cats = ", ".join(DOMAINS[d])
            cnt = domain_counts.get(d, 0)
            pct = (cnt / self.target) * 100
            report_content += f"| **{d}** | `{cats}` | {self.target} | **{cnt}** | {pct:.1f}% |\n"

        report_content += f"""| **TOTAL** | - | **{target_total}** | **{total_valid}** | **{(total_valid / target_total * 100):.1f}%** |

---

## 3. Corpus & Text Statistics

| Metric | Value |
| :--- | :--- |
| **Total Pages Across Corpus** | **{total_pages:,} pages** |
| **Average Pages per Paper** | **{avg_pages:.2f} pages** |
| **Total Word Count** | **{total_words:,} words** |
| **Average Words per Paper** | **{avg_words:.2f} words** |
| **Word Count Range** | {min_words:,} – {max_words:,} words |
| **Average Abstract Length** | **{avg_abs_len:.1f} words** |

---

## 4. Rejection Statistics & Deduplication

- **Duplicates Avoided:** `{len(self.duplicates_log)}` cross-domain / title duplicate attempts filtered.
- **Rejected Candidates:** `{len(self.rejected_log)}` candidate papers rejected due to short abstracts, corrupted PDFs, or insufficient text length.
- Detailed audit logs saved to [`DUPLICATES.csv`](file://{self.duplicates_csv_path}) and [`REJECTED_PAPERS.csv`](file://{self.rejected_log}).

---

## 5. Dataset Status

```text
========================================
DATASET STATUS: {status_str} ({total_valid}/{target_total} VALID PAPERS)
========================================
```
"""
        with open(self.report_md_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # Print summary table to CLI
        print("\n" + "=" * 75)
        print(" HMRP MULTIDISCIPLINARY DATASET VERIFICATION")
        print("=" * 75)
        print(f"{'Domain':<42} | {'Target':<7} | {'Valid':<7} | {'Status'}")
        print("-" * 75)
        for d in DOMAINS:
            cnt = domain_counts.get(d, 0)
            st = "✓ READY" if cnt >= self.target else f"{cnt}/{self.target}"
            print(f"{d:<42} | {self.target:<7} | {cnt:<7} | {st}")
        print("-" * 75)
        print(f"{'TOTAL':<42} | {target_total:<7} | {total_valid:<7} | {status_str}")
        print("=" * 75 + "\n")

        return {
            "status": status_str,
            "total_valid": total_valid,
            "target_total": target_total,
            "domain_counts": domain_counts
        }


# ── CLI Interface ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HMRP Multidisciplinary Dataset Harvester")
    parser.add_argument("--domain", type=str, default=None, help="Harvest a specific domain")
    parser.add_argument("--resume", action="store_true", help="Scan and resume dataset harvesting")
    parser.add_argument("--verify", action="store_true", help="Verify and print dataset status without downloading")
    parser.add_argument("--report", action="store_true", help="Regenerate dataset report and statistics")
    parser.add_argument("--target", type=int, default=TARGET_PER_DOMAIN, help="Target papers per domain (default: 50)")

    args = parser.parse_args()

    harvester = HMRPDatasetHarvester(target_per_domain=args.target)

    if args.verify or args.report:
        harvester.scan_and_revalidate_disk()
        harvester.generate_reports()
        return

    if args.resume:
        harvester.scan_and_revalidate_disk()

    harvester.run_harvesting(specific_domain=args.domain)


if __name__ == "__main__":
    main()
