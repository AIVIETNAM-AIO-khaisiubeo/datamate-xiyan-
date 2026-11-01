"""Evaluate the rule generator alone on a labelled split. Usage: eval_rules.py SPLIT_DIR [biz ...]"""
import sys, os, json, time, collections, datetime as dt, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATAMATE_CACHE", ".cache")
from datamate.ingest.builder import load_biz
from datamate.nlu.rules import RuleParser
from datamate.nlu.timeparse import LunarCalendar
from datamate.engine.executor import Executor
from datamate.engine.answer import FrameExecutor
from datamate.scorer import score_one

root = sys.argv[1]
only = set(sys.argv[2:])
qs = [json.loads(l) for l in open(os.path.join(root, 'questions.jsonl'), encoding='utf-8')]
refs = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(root, 'answers.jsonl'), encoding='utf-8')}
cal = LunarCalendar(os.path.join(root, 'calendar.csv'))
by_biz = collections.defaultdict(list)
for q in qs:
    if not only or q['business_id'] in only:
        by_biz[q['business_id']].append(q)
stats = collections.defaultdict(lambda: [0, 0, 0])
tot = [0, 0, 0]
out = open('eval_rules_out.jsonl', 'w', encoding='utf-8')
for bid, items in sorted(by_biz.items()):
    t0 = time.time()
    b = load_biz(os.path.join(root, 'data', bid))
    rp = RuleParser(b, cal)
    for q in items:
        today = dt.date.fromisoformat(q['as_of_date'])
        try:
            fr = rp.parse(q['question'], today)
        except Exception as e:
            fr = None
            err = traceback.format_exc(limit=2)
        if fr is None:
            ans = {"id": q['id'], "type": "abstain"}; intent = "NONE"
        else:
            fe = FrameExecutor(Executor(b, today), cal)
            try:
                a = fe.run(fr)
            except Exception as e:
                from datamate.engine.answer import Answer
                a = Answer("abstain", reason=traceback.format_exc(limit=3))
            ans = a.to_submission(q['id']); intent = fr.intent
        s = score_one(ans, refs[q['id']])
        idx = 0 if s == 1 else 1 if s == -1 else 2
        stats[intent][idx] += 1; tot[idx] += 1
        out.write(json.dumps({"id": q['id'], "biz": bid, "q": q['question'], "asof": q['as_of_date'], "intent": intent,
                              "pred": ans, "ref": refs[q['id']], "score": s,
                              "frame": fr.to_json() if fr else None}, ensure_ascii=False, default=str) + "\n")
    print(f"{bid} {time.time()-t0:.0f}s", flush=True)
print("TOTAL correct/wrong/abstain", tot, "PA=", (tot[0]-tot[1])/max(1,sum(tot)))
for k, v in sorted(stats.items(), key=lambda kv: -sum(kv[1])):
    print(f"  {k:22s} ok={v[0]:4d} wrong={v[1]:4d} abst={v[2]:4d}")
