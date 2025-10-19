#!/usr/bin/env python3
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="baker-street/maib-incident-classifier",
    tokenizer="baker-street/maib-incident-classifier",
    return_all_scores=True
)

print("MAIB Incident Classifier — type a report (or 'exit' to quit)\n")

while True:
    text = input("> ").strip()
    if text.lower() in {"exit", "quit"}:
        print("Exit")
        break
    if not text:
        continue

    results = classifier(text)
    results = results[0] if isinstance(results[0], list) else results
    top3 = sorted(results, key=lambda x: x["score"], reverse=True)[:3]

    print("→ Top 3 predictions:")
    for r in top3:
        print(f"   {r['label']:<25} {r['score']:.3f}")
    print()
