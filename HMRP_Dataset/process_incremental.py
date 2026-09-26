"""
HMRP Incremental Dataset Processor & Splitter
=============================================
Processes only newly added/unprocessed PDFs across HMRP_Dataset/PDFs/{NLP,IR,AI,ML,CV},
preserves existing processed results, updates TF-IDF, partitions leakage-free splits,
and generates full updated dataset metadata and statistics.
"""

import os
import sys
import json
import glob
import random
import hashlib
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import AnalysisPipeline
from extraction.pdf_extractor import PDFExtractor
from extraction.text_processor import TextProcessor
from utils.logger import get_logger

log = get_logger("HMRP_Dataset_Processor")

PDF_DIR = Path(__file__).resolve().parent / "PDFs"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = ["NLP", "IR", "AI", "ML", "CV"]
SEED = 42

def find_all_pdfs():
    """Find all PDFs in subfolders or base directory."""
    all_pdfs = []
    for d in DOMAINS:
        dp = PDF_DIR / d
        if dp.exists():
            for f in sorted(dp.glob("*.pdf")):
                all_pdfs.append((d, f))
    # Also check base PDF_DIR for any loose PDFs
    for f in sorted(PDF_DIR.glob("*.pdf")):
        all_pdfs.append(("UNCATEGORIZED", f))
    return all_pdfs

def process_incremental_dataset():
    print("=" * 70)
    print("HMRP INCREMENTAL DATASET PROCESSOR")
    print("=" * 70)
    
    # 1. Load existing processed data
    dataset_full_path = PROCESSED_DIR / "dataset_full.json"
    existing_papers = []
    if dataset_full_path.exists():
        try:
            with open(dataset_full_path, "r", encoding="utf-8") as f:
                existing_papers = json.load(f)
            print(f"[*] Loaded {len(existing_papers)} previously processed papers.")
        except Exception as e:
            print(f"[!] Warning reading dataset_full.json: {e}")
            existing_papers = []

    # Map processed papers by file_hash and file_name
    processed_hashes = {p.get("file_hash") for p in existing_papers if p.get("file_hash")}
    processed_filenames = {p.get("file_name") for p in existing_papers if p.get("file_name")}
    
    # 2. Discover all PDFs in dataset folders
    all_pdf_entries = find_all_pdfs()
    print(f"[*] Total PDFs found on disk: {len(all_pdf_entries)}")

    # Group by domain
    domain_counts = {}
    unprocessed_entries = []
    
    for dom, pdf_path in all_pdf_entries:
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        
        # Calculate file hash to check if already processed
        with open(pdf_path, "rb") as fh:
            f_hash = hashlib.sha256(fh.read()).hexdigest()
            
        if f_hash in processed_hashes or pdf_path.name in processed_filenames:
            continue
        unprocessed_entries.append((dom, pdf_path, f_hash))

    print("\n[*] Domain PDF Breakdown on Disk:")
    for d in DOMAINS:
        print(f"    - {d:4s}: {domain_counts.get(d, 0)} PDFs")

    print(f"\n[*] Status:")
    print(f"    - Previously processed: {len(existing_papers)} papers")
    print(f"    - Newly added / Unprocessed: {len(unprocessed_entries)} papers")
    
    pipeline = AnalysisPipeline()
    pdf_extractor = PDFExtractor()
    text_processor = TextProcessor()

    newly_processed = []
    skipped = []
    seen_titles = {
        "".join(filter(str.isalnum, p.get("title", "").lower()))
        for p in existing_papers if p.get("title")
    }

    # 3. Process ONLY newly added / unprocessed PDFs
    if unprocessed_entries:
        print("\n[*] Processing newly added PDFs...")
        for idx, (dom, p_path, file_hash) in enumerate(unprocessed_entries):
            print(f"\n[{idx+1}/{len(unprocessed_entries)}] [{dom}] Processing: {p_path.name}")
            
            try:
                # Extract text & metadata
                ext = pdf_extractor.extract(p_path)
                raw_text = ext.get("raw_text", "")
                title = ext.get("title", "").strip()

                if len(raw_text.strip()) < 100:
                    print(f"  [!] Skipping empty/unreadable text in {p_path.name}")
                    skipped.append({"file": p_path.name, "domain": dom, "reason": "empty_text"})
                    continue

                # Title duplication check
                title_norm = "".join(filter(str.isalnum, title.lower()))
                if title_norm and title_norm in seen_titles:
                    print(f"  [!] Skipping duplicate title: {title[:60]}")
                    skipped.append({"file": p_path.name, "domain": dom, "reason": "duplicate_title"})
                    continue
                if title_norm:
                    seen_titles.add(title_norm)

                # Process text
                proc = text_processor.process(ext.get("full_text", ""))

                # Build paper object via pipeline
                paper_obj = pipeline._build_paper(
                    ext=ext,
                    proc=proc,
                    tfidf_kws=[],
                    corpus_texts=[proc.get("cleaned_text", "")]
                )

                paper_dict = {
                    "paper_id": paper_obj.paper_id,
                    "file_hash": file_hash,
                    "file_name": p_path.name,
                    "file_size_bytes": os.path.getsize(p_path),
                    "title": paper_obj.title,
                    "authors": paper_obj.authors,
                    "year": paper_obj.year,
                    "page_count": paper_obj.page_count,
                    "char_count": paper_obj.char_count,
                    "word_count": paper_obj.word_count,
                    "abstract": paper_obj.abstract,
                    "introduction": paper_obj.introduction,
                    "problem": paper_obj.problem,
                    "objective": paper_obj.objective,
                    "proposed_method": paper_obj.proposed_method,
                    "models": paper_obj.models,
                    "algorithms": paper_obj.algorithms,
                    "datasets": paper_obj.datasets,
                    "evaluation_metrics": paper_obj.evaluation_metrics,
                    "results": paper_obj.results,
                    "advantages": paper_obj.advantages,
                    "limitations": paper_obj.limitations,
                    "research_gap": paper_obj.research_gap,
                    "future_work": paper_obj.future_work,
                    "all_keywords": paper_obj.all_keywords,
                    "research_concepts": paper_obj.research_concepts,
                    "technical_terms": paper_obj.technical_terms,
                    "named_entities": paper_obj.named_entities,
                    "concept_hierarchy": paper_obj.concept_hierarchy,
                    "research_domains": paper_obj.research_domains,
                    "cleaned_text_preview": paper_obj.cleaned_text[:500] if paper_obj.cleaned_text else "",
                    "full_text": paper_obj.full_text,
                    "cleaned_text": paper_obj.cleaned_text,
                }

                newly_processed.append(paper_dict)
                print(f"  [✓] Success: {paper_obj.title[:60]}")
                print(f"      Pages: {paper_obj.page_count} | Words: {paper_obj.word_count} | Concepts: {len(paper_obj.research_concepts)}")

            except Exception as e:
                print(f"  [✗] Failed to process {p_path.name}: {e}")
                skipped.append({"file": p_path.name, "domain": dom, "reason": str(e)})

    # 4. Merge existing results with newly processed papers
    combined_papers = list(existing_papers) + newly_processed
    print(f"\n==========================================")
    print(f"Combined Total Valid Processed Papers: {len(combined_papers)}")
    print(f"Newly Processed This Run: {len(newly_processed)}")
    print(f"Skipped / Failed: {len(skipped)}")
    print(f"==========================================")

    # 5. Compute global TF-IDF over full combined corpus
    print("[*] Updating TF-IDF representations across the full dataset...")
    corpus_cleaned = [p["cleaned_text"] for p in combined_papers]
    tfidf_batch = pipeline._keywords.extract_tfidf_corpus(corpus_cleaned)
    for p, tfidf_kws in zip(combined_papers, tfidf_batch):
        p["tfidf_keywords"] = tfidf_kws

    # 6. Leakage-Free Stratified / Random Document Split (70% Train, 15% Val, 15% Test)
    print("[*] Re-partitioning leakage-free document splits (70% Train, 15% Val, 15% Test)...")
    shuffled = list(combined_papers)
    random.seed(SEED)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_papers = shuffled[:n_train]
    val_papers = shuffled[n_train:n_train + n_val]
    test_papers = shuffled[n_train + n_val:]

    for p in train_papers: p["split"] = "train"
    for p in val_papers: p["split"] = "val"
    for p in test_papers: p["split"] = "test"

    print(f"    - Train Split: {len(train_papers)} papers ({len(train_papers)/n_total*100:.1f}%)")
    print(f"    - Val Split:   {len(val_papers)} papers ({len(val_papers)/n_total*100:.1f}%)")
    print(f"    - Test Split:  {len(test_papers)} papers ({len(test_papers)/n_total*100:.1f}%)")

    # 7. Save outputs
    with open(PROCESSED_DIR / "dataset_full.json", "w", encoding="utf-8") as f:
        json.dump(combined_papers, f, indent=2, ensure_ascii=False)

    with open(PROCESSED_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_papers, f, indent=2, ensure_ascii=False)

    with open(PROCESSED_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_papers, f, indent=2, ensure_ascii=False)

    with open(PROCESSED_DIR / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_papers, f, indent=2, ensure_ascii=False)

    # 8. Generate updated summary table CSV
    summary_rows = []
    for p in combined_papers:
        summary_rows.append({
            "paper_id": p["paper_id"],
            "file_name": p["file_name"],
            "title": p["title"],
            "authors": ", ".join(p["authors"]) if p["authors"] else "N/A",
            "year": p["year"] or "N/A",
            "split": p.get("split", "unassigned"),
            "page_count": p["page_count"],
            "word_count": p["word_count"],
            "abstract_length": len(p["abstract"]),
            "num_keywords": len(p["all_keywords"]),
            "num_concepts": len(p["research_concepts"]),
            "domains": ", ".join(p["research_domains"]) if p["research_domains"] else "N/A"
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(PROCESSED_DIR / "metadata_summary.csv", index=False)

    # 9. Compute comprehensive statistics
    total_pages = int(df_summary["page_count"].sum())
    total_words = int(df_summary["word_count"].sum())
    total_keywords = int(df_summary["num_keywords"].sum())
    total_concepts = int(df_summary["num_concepts"].sum())

    stats = {
        "total_pdfs_scanned": len(all_pdf_entries),
        "domain_pdf_counts": domain_counts,
        "previously_processed": len(existing_papers),
        "newly_processed": len(newly_processed),
        "valid_papers_processed": len(combined_papers),
        "failed_or_skipped": len(skipped),
        "skipped_details": skipped,
        "splits": {
            "train_count": len(train_papers),
            "val_count": len(val_papers),
            "test_count": len(test_papers)
        },
        "totals": {
            "total_pages": total_pages,
            "total_words": total_words,
            "total_keywords": total_keywords,
            "total_concepts": total_concepts
        },
        "averages": {
            "avg_pages": float(df_summary["page_count"].mean()),
            "min_pages": int(df_summary["page_count"].min()),
            "max_pages": int(df_summary["page_count"].max()),
            "avg_words": float(df_summary["word_count"].mean()),
            "min_words": int(df_summary["word_count"].min()),
            "max_words": int(df_summary["word_count"].max()),
            "avg_concepts": float(df_summary["num_concepts"].mean()),
            "avg_keywords": float(df_summary["num_keywords"].mean())
        }
    }

    with open(PROCESSED_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[✓] Processing complete! Processed artifacts saved to: {PROCESSED_DIR}")
    return stats

if __name__ == "__main__":
    process_incremental_dataset()
