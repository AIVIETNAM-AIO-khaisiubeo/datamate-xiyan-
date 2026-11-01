import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATAMATE_CACHE", ".cache")
from datamate.ingest.builder import build_biz
root = sys.argv[1]
names = sys.argv[2:] or sorted(os.listdir(root))
for biz in names:
    t0 = time.time()
    try:
        b = build_biz(os.path.join(root, biz))
    except Exception as e:
        import traceback; traceback.print_exc(); print('FAIL', biz); continue
    L, O = b.lines, b.orders
    print(f"#### {biz} {time.time()-t0:.1f}s lines={len(L)} orders={len(O)} pay={len(b.payments)} prod={len(b.products)} units={len(b.unit_factor)} "
          f"purch={len(b.purchases)}/{len(b.purchase_lines)} stock={len(b.stock_end)} lots={len(b.lots)}/{len(b.lot_sales)} deliv={len(b.deliveries)} "
          f"terms={len(b.debt_terms)} prices={len(b.prices)} exp={len(b.expenses)} cust={len(b.customers)} range={b.data_start}..{b.data_end} flags={b.flags}")
    if not O.empty:
        print('   status:', O['status_final'].value_counts().to_dict(), 'raw:', (O['status'].map(lambda v: str(v)).value_counts().head(8).to_dict() if 'status' in O else None))
        print('   pay:', O['pay_method_n'].value_counts().to_dict() if 'pay_method_n' in O else None, 'chan:', O['channel_n'].value_counts().to_dict() if 'channel_n' in O else None)
    if not L.empty:
        print('   kinds:', L['kind'].value_counts().to_dict(), 'neg:', int(L['neg'].sum()), 'nan_amount:', int(L['amount'].isna().sum()), 'nan_date:', int(L['date'].isna().sum()), 'nan_base:', int(L['base_qty'].isna().sum()))
    for l in b.log:
        if 'dropped' in l: print('   ', l)
