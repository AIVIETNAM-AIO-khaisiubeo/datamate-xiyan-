"""Record the rulebook rows the rule engine was written against (artifacts/known_rulebook_rows.json).

The engine was developed while reading the train rulebook and, later, the public one (label-free error analysis):
those rows are 'known'.  At inference, a row of the split's rulebook outside this list is a concept the engine
does not implement (datamate.rulebook.new_rows); see the agent's rulebook guard.

    python scripts/build_known_rulebook.py data/train/rulebook.md data/public_test/rulebook.md
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datamate.rulebook import KNOWN_ROWS_PATH, concept_tokens, parse_rulebook, row_id  # noqa: E402


def main(paths):
    rows = set()
    for p in paths:
        rb = parse_rulebook(open(p, encoding="utf-8").read())
        rows |= {row_id(r) for r in rb.rows if len(concept_tokens(r.key)) >= 2}
    out = os.path.join(ROOT, KNOWN_ROWS_PATH)
    json.dump({"sources": [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in paths], "rows": sorted(rows)},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(rows)} known rule rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(ROOT, "data", "train", "rulebook.md"),
                          os.path.join(ROOT, "data", "public_test", "rulebook.md")])
