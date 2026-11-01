import sys, os, json, re, datetime as dt, calendar, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATAMATE_CACHE", ".cache")
from datamate.ingest.builder import build_biz, load_biz
from datamate.textnorm import normalize_question
root = sys.argv[1]
qs = [json.loads(l) for l in open(os.path.join(root, 'questions.jsonl'), encoding='utf-8')]
ans = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(root, 'answers.jsonl'), encoding='utf-8')}
pat = re.compile(r"^(?:cho (?:toi|minh) biet |cho hoi )?(?:tong )?doanh thu thang (\d{1,2})/(\d{4})(?: la bao nhieu| bao nhieu)?$")
pat2 = re.compile(r"thang (\d{1,2})/(\d{4}) cua hang ban duoc tong cong bao nhieu tien")
bizs = sys.argv[2:] or sorted(set(q['business_id'] for q in qs))
for bid in bizs:
    cp = f".cache/biz_{bid}.pkl"
    b = load_biz(os.path.join(root, "data", bid))
    O = b.orders; L = b.lines
    ok = bad = 0
    for q in qs:
        if q['business_id'] != bid: continue
        nq = normalize_question(q['question'])
        m = pat.search(nq) or pat2.search(nq)
        if not m: continue
        mo, yr = int(m.group(1)), int(m.group(2))
        d1 = dt.date(yr, mo, 1); d2 = dt.date(yr, mo, calendar.monthrange(yr, mo)[1])
        if not O.empty:
            sel = O[(O['date'] >= d1) & (O['date'] <= d2) & O['valid']]
            got = sel['revenue'].sum()
        else:
            sel = L[(L['date'] >= d1) & (L['date'] <= d2) & L['valid'] & L['kind'].isin(['item','service_fee'])]
            got = sel['amount'].sum()
        a = ans[q['id']]
        if a['type'] != 'number':
            print('  nonnum', q['question'], a); continue
        if abs(got - a['answer']) < 0.5: ok += 1
        else:
            bad += 1
            if bad <= 3: print(f"  BAD {q['question']} got={got:.0f} exp={a['answer']} diff={got-a['answer']:.0f}")
    print(bid, 'ok', ok, 'bad', bad, flush=True)
