import os
import sys
import time
import requests
import feedparser
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "PDFs")

DOMAINS = {
    "NLP": "cs.CL",
    "IR": "cs.IR",
    "AI": "cs.AI",
    "ML": "cs.LG",
    "CV": "cs.CV",
}

TARGET_PER_DOMAIN = 100

HEADERS = {
    "User-Agent": "HMRP-Academic-Research/1.0 (mailto:academic-research@university.edu)"
}

def validate_pdf(filepath):
    """Check if file exists, is non-empty, and starts with %PDF marker."""
    if not os.path.exists(filepath):
        return False
    try:
        if os.path.getsize(filepath) < 1024:  # Under 1KB is almost certainly corrupt/HTML error
            return False
        with open(filepath, "rb") as f:
            header = f.read(4)
            return header == b"%PDF"
    except Exception:
        return False

def count_valid_pdfs(output_dir):
    """Count valid PDFs and remove corrupt/empty ones."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return 0, []

    valid_count = 0
    existing_ids = []
    
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".pdf"):
            continue
        fpath = os.path.join(output_dir, fname)
        if validate_pdf(fpath):
            valid_count += 1
            paper_id = fname[:-4]
            existing_ids.append(paper_id)
        else:
            print(f"  [CLEANUP] Removing invalid/corrupt PDF: {fname}")
            try:
                os.remove(fpath)
            except Exception as e:
                print(f"  [ERROR] Could not remove {fname}: {e}")
                
    return valid_count, existing_ids

def download_domain(domain, category, target=TARGET_PER_DOMAIN):
    output_dir = os.path.join(PDF_DIR, domain)
    os.makedirs(output_dir, exist_ok=True)
    
    valid_count, existing_ids = count_valid_pdfs(output_dir)
    print("\n" + "=" * 60)
    print(f"DOMAIN: {domain} (arXiv category: {category})")
    print(f"Target: {target} PDFs | Existing valid: {valid_count} | Needed: {max(0, target - valid_count)}")
    print(f"Output folder: {output_dir}")
    print("=" * 60)
    
    if valid_count >= target:
        print(f"Domain {domain} already has {valid_count} valid PDFs. Target reached!")
        return {
            "domain": domain,
            "target": target,
            "valid_count": valid_count,
            "downloaded": 0,
            "skipped": len(existing_ids),
            "failed": 0,
        }

    seen_ids = set(existing_ids)
    downloaded_this_run = 0
    failed_count = 0
    skipped_count = len(existing_ids)
    
    start = 0
    query = f"cat:{category}"
    max_batch = 50
    consecutive_empty = 0

    while valid_count < target and start < 2000:
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={quote(query)}"
            f"&start={start}"
            f"&max_results={max_batch}"
            "&sortBy=submittedDate"
            "&sortOrder=descending"
        )
        
        print(f"\n[API Query] {domain} start={start}, batch_size={max_batch}...")
        
        response = None
        retries = 0
        while retries < 5:
            try:
                response = requests.get(url, headers=HEADERS, timeout=45)
                if response.status_code == 429:
                    wait_sec = 15 * (retries + 1)
                    print(f"  [429 Rate Limited] arXiv API limit hit. Backing off for {wait_sec}s...")
                    time.sleep(wait_sec)
                    retries += 1
                    continue
                response.raise_for_status()
                break
            except Exception as e:
                print(f"  [API Error] {e}. Retrying in 10s (attempt {retries+1}/5)...")
                time.sleep(10)
                retries += 1

        if response is None or response.status_code != 200:
            print(f"Failed to fetch feed after retries. Moving to next offset.")
            start += max_batch
            time.sleep(5)
            continue

        feed = feedparser.parse(response.text)
        if not feed.entries:
            consecutive_empty += 1
            print("No entries returned in feed.")
            if consecutive_empty > 2:
                break
            start += max_batch
            time.sleep(4)
            continue
        
        consecutive_empty = 0
        
        for entry in feed.entries:
            if valid_count >= target:
                break
                
            raw_id = entry.id.split("/")[-1]
            paper_id = raw_id.split("v")[0] if "v" in raw_id else raw_id
            paper_id = paper_id.replace("/", "_")
            
            if paper_id in seen_ids:
                continue
                
            seen_ids.add(paper_id)
            filepath = os.path.join(output_dir, f"{paper_id}.pdf")
            
            if os.path.exists(filepath):
                if validate_pdf(filepath):
                    valid_count += 1
                    skipped_count += 1
                    print(f"  [EXISTS] {paper_id}.pdf is already valid. (Total: {valid_count}/{target})")
                    continue
                else:
                    os.remove(filepath)

            # Try candidate PDF URLs
            candidate_urls = [
                f"https://arxiv.org/pdf/{paper_id}.pdf",
                f"https://export.arxiv.org/pdf/{paper_id}.pdf"
            ]
            for link in getattr(entry, "links", []):
                if link.get("type") == "application/pdf" and link.get("href"):
                    candidate_urls.insert(0, link["href"])

            title = " ".join(entry.title.split())
            print(f"[{valid_count + 1}/{target}] Downloading {paper_id}: {title[:70]}...")

            pdf_downloaded = False
            for p_url in candidate_urls:
                try:
                    r = requests.get(p_url, headers=HEADERS, timeout=60, allow_redirects=True)
                    if r.status_code == 429:
                        print("    [429] Rate limited on PDF download. Sleeping 20s...")
                        time.sleep(20)
                        r = requests.get(p_url, headers=HEADERS, timeout=60, allow_redirects=True)
                    
                    if r.status_code == 200 and r.content.startswith(b"%PDF"):
                        with open(filepath, "wb") as f:
                            f.write(r.content)
                        
                        if validate_pdf(filepath):
                            valid_count += 1
                            downloaded_this_run += 1
                            pdf_downloaded = True
                            print(f"    ✓ Saved valid PDF ({len(r.content)/1024:.1f} KB)")
                            break
                        else:
                            if os.path.exists(filepath):
                                os.remove(filepath)
                except Exception as ex:
                    print(f"    ✗ Attempt with {p_url} failed: {ex}")
                    time.sleep(2)

            if not pdf_downloaded:
                failed_count += 1
                print(f"    ✗ Failed to download valid PDF for {paper_id}")
            
            # Respect rate limit
            time.sleep(2.0)

        start += max_batch
        time.sleep(3.5)

    final_valid, _ = count_valid_pdfs(output_dir)
    return {
        "domain": domain,
        "target": target,
        "valid_count": final_valid,
        "downloaded": downloaded_this_run,
        "skipped": skipped_count,
        "failed": failed_count
    }

def main():
    print("=" * 70)
    print(" HMRP DATASET BULK DOWNLOADER & ORGANIZER ")
    print("=" * 70)
    print(f"Base PDF Directory: {PDF_DIR}")
    
    specific_domain = sys.argv[1].upper() if len(sys.argv) > 1 else None
    
    domains_to_run = [specific_domain] if specific_domain and specific_domain in DOMAINS else list(DOMAINS.keys())
    
    summary_results = []
    
    for d in domains_to_run:
        res = download_domain(d, DOMAINS[d], TARGET_PER_DOMAIN)
        summary_results.append(res)
        time.sleep(2)
        
    print("\n\n" + "=" * 70)
    print(" FINAL DATASET VERIFICATION & SUMMARY REPORT")
    print("=" * 70)
    
    total_valid = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    
    for res in summary_results:
        d = res["domain"]
        valid = res["valid_count"]
        total_valid += valid
        total_downloaded += res["downloaded"]
        total_skipped += res["skipped"]
        total_failed += res["failed"]
        folder = os.path.join(PDF_DIR, d)
        print(f"{d}: {valid} PDFs  (Folder: {folder})")
        
    print("-" * 70)
    print(f"TOTAL: {total_valid} PDFs")
    print("-" * 70)
    print(f"Downloaded this session : {total_downloaded}")
    print(f"Skipped (already valid)  : {total_skipped}")
    print(f"Failed attempts         : {total_failed}")
    print(f"Corrupt PDFs            : 0 (All corrupt/invalid PDFs removed automatically)")
    print("=" * 70)

if __name__ == "__main__":
    main()
