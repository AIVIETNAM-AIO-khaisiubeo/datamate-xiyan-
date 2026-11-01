import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamate.ingest.reader import read_business_files
from datamate.ingest.roles import classify, assign_roles, parse_readme_columns
root = sys.argv[1]
for biz in sorted(os.listdir(root))[: int(sys.argv[2]) if len(sys.argv) > 2 else 999]:
    d = os.path.join(root, biz)
    t0 = time.time()
    tabs = read_business_files(d)
    rd = parse_readme_columns(open(os.path.join(d, 'README.md'), encoding='utf-8').read())
    print('####', biz, f'{time.time()-t0:.1f}s')
    seen = set()
    for t in tabs:
        rc = rd.get(t.file) or rd.get(f"{t.file}::{t.sheet}") or {}
        k = classify(t, rc)
        r = assign_roles(t, k, rc)
        key = (t.file.split('_')[0], k, tuple(sorted(r.items())))
        if key in seen: continue
        seen.add(key)
        unk = [c for c in t.columns if c not in r]
        print(f"  {t.name[:50]:50s} {k:14s} n={len(t.df)} y={t.sheet_year} | " + ', '.join(f"{c}->{v}" for c, v in r.items()) + (f" | UNMAPPED {unk}" if unk else ''))
