import os
import time
import requests
import feedparser
from urllib.parse import quote

OUTPUT_DIR = os.path.expanduser("~/HMRP_Dataset/PDFs")
TARGET = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

queries = [
    "cat:cs.CL",
    "cat:cs.IR",
]

downloaded = 0
seen = set()

headers = {
    "User-Agent": "HMRP-Academic-Research/1.0"
}

for query in queries:
    if downloaded >= TARGET:
        break

    start = 0

    while downloaded < TARGET and start < 100:
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={quote(query)}"
            "&start="
            f"{start}"
            "&max_results=20"
            "&sortBy=submittedDate"
            "&sortOrder=descending"
        )

        print(f"\nFetching papers: {query}, start={start}")

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        feed = feedparser.parse(response.text)

        if not feed.entries:
            break

        for entry in feed.entries:
            if downloaded >= TARGET:
                break

            paper_id = entry.id.split("/")[-1]
            paper_id = paper_id.replace("/", "_")

            if paper_id in seen:
                continue

            seen.add(paper_id)

            pdf_url = None
            for link in entry.links:
                if link.get("type") == "application/pdf":
                    pdf_url = link.href
                    break

            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{paper_id}"

            filename = os.path.join(OUTPUT_DIR, f"{paper_id}.pdf")

            if os.path.exists(filename):
                print(f"SKIP: {paper_id}")
                continue

            try:
                print(f"[{downloaded + 1}/{TARGET}] {entry.title[:70]}")

                r = requests.get(
                    pdf_url,
                    headers=headers,
                    timeout=60
                )

                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    with open(filename, "wb") as f:
                        f.write(r.content)

                    downloaded += 1
                    print(f"  ✓ Saved: {filename}")

                else:
                    print(f"  ✗ Not a valid PDF")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

            time.sleep(1)

        start += 20
        time.sleep(2)

print("\n==============================")
print(f"Downloaded PDFs: {downloaded}")
print(f"Location: {OUTPUT_DIR}")
print("==============================")