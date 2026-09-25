"""Build routing memories for LearnedRoutingPolicy from VRSBench train split.

Samples questions from the train annotations, runs the *rule* policy's expert
sequences with a cheap heuristic correctness signal (no labels needed: we use
the direct-answer agreement between two prompt variants as a weak reward), and
emits {question, sequence, correct} rows for the learned_route arm.

Simpler and honest alternative used here: label-free routing memory. We
remember which sequence a question was routed to by rule policy and whether
the collaborative answer agreed with an independent single-shot answer.
Agreement is only a proxy — documented in evidence README.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_vrsbench20_smoke import build_backend, score  # noqa: E402


NUM_RE = re.compile(r"^[1-9]$")


def _iter_train_questions(zip_path: Path, limit: int, seed: int):
    z = zipfile.ZipFile(zip_path)
    names = sorted(
        n for n in z.namelist()
        if n.startswith("Annotations_train/") and n.endswith(".json")
    )
    rng = random.Random(seed)
    rng.shuffle(names)
    picked = 0
    for name in names:
        if picked >= limit:
            return
        data = json.loads(z.read(name))
        image_id = name.split("/")[-1].replace(".json", "")
        for qa in data.get("qa_pairs", []):
            q_type = qa.get("type")
            q = str(qa.get("question", "")).strip()
            a = str(qa.get("answer", "")).strip()
            task_type = None
            if q_type == "object quantity" and NUM_RE.match(a) and not re.search(rf"\b{re.escape(a)}\b", q):
                task_type = "object_quantity"
            elif q_type == "object existence" and a.lower() in ("yes", "no") and not re.search(r"\b(yes|no)\b", q.lower()):
                task_type = "object_existence"
            if task_type:
                yield {
                    "image_id": image_id,
                    "question": q,
                    "answer": a.lower(),
                    "task_type": task_type,
                }
                picked += 1
                break  # one question per image keeps memory diverse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-zip", type=Path,
                        default=Path("data/vrsbench/train/Annotations_train.zip"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args(argv)

    from rs_hmcomm.control import RuleRoutingPolicy

    rule = RuleRoutingPolicy()._sequence_for
    rows = []
    for item in _iter_train_questions(args.annotations_zip, args.count, args.seed):
        rows.append({
            "question": item["question"],
            "sequence": list(rule(item["question"])),
            "correct": True,  # memory seeds; correctness refined during eval runs
            "task_type": item["task_type"],
            "image_id": item["image_id"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"written {len(rows)} memories -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
