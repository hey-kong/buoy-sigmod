import os
import json
import random
from datasets import load_dataset


dataset = load_dataset("allenai/qasper", split="test")

os.makedirs("documents", exist_ok=True)
for item in dataset:
    paper_id = item["id"]
    fname = paper_id + ".txt"

    out = {
        "title": item["title"],
        "abstract": item["abstract"],
    }

    section_names = item["full_text"]["section_name"]
    section_paragraphs = item["full_text"]["paragraphs"]
    for sec_name, paras in zip(section_names, section_paragraphs):
        out[sec_name] = paras

    with open(os.path.join("documents", fname), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

pairs = []
for item in dataset:
    paper_id = item["id"]
    questions = item.get("qas", {}).get("question", []) or []
    for q in questions:
        pairs.append({"paper_id": paper_id, "question": q})

random.seed(42)
random.shuffle(pairs)

with open("request.jsonl", "w", encoding="utf-8") as f:
    for rid, p in enumerate(pairs):
        rec = {"request_id": rid, **p}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
