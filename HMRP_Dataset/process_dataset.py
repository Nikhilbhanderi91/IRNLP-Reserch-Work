"""
HMRP Dataset Processor & Splitter
==================================
Processes all PDFs in HMRP_Dataset/PDFs, performs deduplication, text cleaning,
structured extraction, validation, and creates train/val/test splits strictly
at the document level to prevent data leakage.
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

SEED = 42
random.seed(SEED)

def process_all_pdfs():
    print(f"[*] Scanning PDFs from: {PDF_DIR}")
    pdf_files = sorted(glob.glob(str(PDF_DIR / "*.pdf")))
    print(f"[*] Found {len(pdf_files)} PDF files.")

    pipeline = AnalysisPipeline()
    pdf_extractor = PDFExtractor()
    text_processor = TextProcessor()

    processed_papers = []
    skipped = []
    seen_hashes = set()
    seen_titles = set()

    for idx, pdf_path in enumerate(pdf_files):
        p_path = Path(pdf_path)
        print(f"\n[{idx+1}/{len(pdf_files)}] Processing: {p_path.name}")
        
        # 1. Hash & Duplicate Check
        with open(p_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        if file_hash in seen_hashes:
            print(f"  [!] Skipping duplicate file hash: {file_hash}")
            skipped.append({"file": p_path.name, "reason": "duplicate_hash"})
            continue
        seen_hashes.add(file_hash)

        # 2. Extract with PDFExtractor
        try:
            ext = pdf_extractor.extract(p_path)
            raw_text = ext.get("raw_text", "")
            title = ext.get("title", "").strip()

            if len(raw_text.strip()) < 100:
                print(f"  [!] Skipping empty/unreadable text in {p_path.name}")
                skipped.append({"file": p_path.name, "reason": "empty_text"})
                continue

            # Check title duplication
            title_norm = "".join(filter(str.isalnum, title.lower()))
            if title_norm and title_norm in seen_titles:
                print(f"  [!] Skipping duplicate title: {title}")
                skipped.append({"file": p_path.name, "reason": "duplicate_title"})
                continue
            if title_norm:
                seen_titles.add(title_norm)

            # 3. Clean and process text
            proc = text_processor.process(ext.get("full_text", ""))

            # 4. Extract Keywords & Concepts via pipeline's single builder
            paper_obj = pipeline._build_paper(
                ext=ext,
                proc=proc,
                tfidf_kws=[],
                corpus_texts=[proc.get("cleaned_text", "")]
            )

            # 5. Build serializable dictionary
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

            processed_papers.append(paper_dict)
            print(f"  [✓] Successfully processed: {paper_obj.title[:60]}")
            print(f"      Pages: {paper_obj.page_count} | Words: {paper_obj.word_count} | Concepts: {len(paper_obj.research_concepts)}")

        except Exception as e:
            print(f"  [✗] Failed to process {p_path.name}: {e}")
            skipped.append({"file": p_path.name, "reason": str(e)})

    # Compute TF-IDF over the full processed corpus
    corpus_cleaned = [p["cleaned_text"] for p in processed_papers]
    tfidf_batch = pipeline._keywords.extract_tfidf_corpus(corpus_cleaned)
    for p, tfidf_kws in zip(processed_papers, tfidf_batch):
        p["tfidf_keywords"] = tfidf_kws

    print(f"\n==========================================")
    print(f"Total Valid Processed Papers: {len(processed_papers)}")
    print(f"Total Skipped/Duplicate Papers: {len(skipped)}")
    print(f"==========================================")

    # 6. Leakage-Free Stratified / Random Document Split (70% Train, 15% Val, 15% Test)
    shuffled = list(processed_papers)
    random.seed(SEED)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_papers = shuffled[:n_train]
    val_papers = shuffled[n_train:n_train + n_val]
    test_papers = shuffled[n_train + n_val:]

    print(f"[*] Dataset Splits (Leakage-Free by Paper ID):")
    print(f"    - Train Split: {len(train_papers)} papers ({len(train_papers)/n_total*100:.1f}%)")
    print(f"    - Val Split:   {len(val_papers)} papers ({len(val_papers)/n_total*100:.1f}%)")
    print(f"    - Test Split:  {len(test_papers)} papers ({len(test_papers)/n_total*100:.1f}%)")

    # Mark split attribute in objects
    for p in train_papers: p["split"] = "train"
    for p in val_papers: p["split"] = "val"
    for p in test_papers: p["split"] = "test"

    # 7. Save outputs
    # All papers json
    with open(PROCESSED_DIR / "dataset_full.json", "w", encoding="utf-8") as f:
        json.dump(processed_papers, f, indent=2, ensure_ascii=False)

    # Individual split files
    with open(PROCESSED_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_papers, f, indent=2, ensure_ascii=False)

    with open(PROCESSED_DIR / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_papers, f, indent=2, ensure_ascii=False)

    with open(PROCESSED_DIR / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_papers, f, indent=2, ensure_ascii=False)

    # Summary table CSV
    summary_rows = []
    for p in processed_papers:
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

    # Write summary stats
    stats = {
        "total_pdfs_scanned": len(pdf_files),
        "valid_papers_processed": len(processed_papers),
        "duplicates_or_corrupted": len(skipped),
        "splits": {
            "train_count": len(train_papers),
            "val_count": len(val_papers),
            "test_count": len(test_papers)
        },
        "averages": {
            "avg_pages": float(df_summary["page_count"].mean()),
            "avg_words": float(df_summary["word_count"].mean()),
            "avg_concepts": float(df_summary["num_concepts"].mean()),
            "avg_keywords": float(df_summary["num_keywords"].mean())
        }
    }
    with open(PROCESSED_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[✓] Processing Complete. Files written to: {PROCESSED_DIR}")
    return stats

if __name__ == "__main__":
    process_all_pdfs()
