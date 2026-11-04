"""Same shops, same questions, same answers - different presentation of the files (metamorphic test of ingest).

Private shops will export their books differently.  This script rewrites the data files of chosen train shops with
changes that do not alter a single value, so every gold answer stays valid, and the agent must answer exactly as
on the original files:

  cols_reversed     every sheet / CSV with its columns in reverse order
  sheets_reversed   the sheets of every workbook in reverse order
  headers_upper     the header row (first row with >= 2 filled cells) in upper case
  csv_semicolon     CSV files with ';' as delimiter

    python scripts/perturb_split.py data/train runs/perturb --fold 0 [--kinds cols_reversed,sheets_reversed]
    python -m datamate.predict --data runs/perturb/cols_reversed --llm none --out runs/perturb_cols/submission.jsonl

No value is invented or changed: this is the original data, laid out differently.
"""
import argparse
import csv
import io
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datamate.folds import fold_shops  # noqa: E402

KINDS = ("cols_reversed", "sheets_reversed", "headers_upper", "csv_semicolon")


def _header_index(rows):
    for i, r in enumerate(rows):
        if sum(1 for v in r if v not in (None, "")) >= 2:
            return i
    return None


def _transform_rows(rows, kind):
    rows = [list(r) for r in rows]
    if kind == "cols_reversed":
        width = max((len(r) for r in rows), default=0)
        rows = [list(reversed(r + [None] * (width - len(r)))) for r in rows]
    elif kind == "headers_upper":
        h = _header_index(rows)
        if h is not None:
            rows[h] = [v.upper() if isinstance(v, str) else v for v in rows[h]]
    return rows


def _decode(raw):
    for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


def perturb_csv(src, dst, kind):
    text, enc = _decode(open(src, "rb").read())
    try:
        delim = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|").delimiter
    except csv.Error:
        delim = ","
    rows = _transform_rows(list(csv.reader(io.StringIO(text), delimiter=delim)), kind)
    buf = io.StringIO()
    csv.writer(buf, delimiter=";" if kind == "csv_semicolon" else delim, lineterminator="\n").writerows(
        [["" if v is None else v for v in r] for r in rows])
    open(dst, "w", encoding="utf-8-sig" if enc == "utf-8-sig" else "utf-8", newline="").write(buf.getvalue())


def perturb_xlsx(src, dst, kind):
    import openpyxl
    wb = openpyxl.load_workbook(src, read_only=True, data_only=True)       # streaming: MB-sized books in seconds
    sheets = [(ws.title, [list(r) for r in ws.iter_rows(values_only=True)]) for ws in wb.worksheets]
    wb.close()
    if kind == "sheets_reversed":
        sheets = sheets[::-1]
    out = openpyxl.Workbook(write_only=True)
    for title, rows in sheets:
        ws = out.create_sheet(title)
        for r in _transform_rows(rows, kind):
            ws.append(r)
    out.save(dst)


def make(split, out, kind, fold=None, folds=5):
    qs = [json.loads(l) for l in open(os.path.join(split, "questions.jsonl"), encoding="utf-8") if l.strip()]
    shops = sorted({q["business_id"] for q in qs})
    if fold is not None:
        shops = sorted(fold_shops(shops, fold, folds))
    d = os.path.join(out, kind)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "data"))
    keep = set(shops)
    with open(os.path.join(d, "questions.jsonl"), "w", encoding="utf-8") as fo:
        for q in qs:
            if q["business_id"] in keep:
                fo.write(json.dumps(q, ensure_ascii=False) + "\n")
    ids = {q["id"] for q in qs if q["business_id"] in keep}
    if os.path.exists(os.path.join(split, "answers.jsonl")):
        with open(os.path.join(d, "answers.jsonl"), "w", encoding="utf-8") as fo:
            for l in open(os.path.join(split, "answers.jsonl"), encoding="utf-8"):
                if l.strip() and json.loads(l)["id"] in ids:
                    fo.write(l if l.endswith("\n") else l + "\n")
    for name in os.listdir(split):
        if name not in ("questions.jsonl", "answers.jsonl", "data"):
            shutil.copy(os.path.join(split, name), os.path.join(d, name))
    changed = 0
    for s in shops:
        sd, dd = os.path.join(split, "data", s), os.path.join(d, "data", s)
        os.makedirs(dd)
        for f in os.listdir(sd):
            src, dst = os.path.join(sd, f), os.path.join(dd, f)
            ext = os.path.splitext(f)[1].lower()
            if ext == ".csv" and kind != "sheets_reversed":
                perturb_csv(src, dst, kind)
                changed += 1
            elif ext == ".xlsx" and kind != "csv_semicolon":
                perturb_xlsx(src, dst, kind)
                changed += 1
            else:
                shutil.copy(src, dst)
    return d, shops, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("out")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--kinds", default=",".join(KINDS))
    a = ap.parse_args()
    for kind in a.kinds.split(","):
        d, shops, n = make(a.split, a.out, kind, a.fold, a.folds)
        print(f"{kind:16s} {n:3d} files rewritten, {len(shops)} shops -> {d}")


if __name__ == "__main__":
    main()
