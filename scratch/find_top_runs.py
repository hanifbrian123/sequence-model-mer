import csv

with open('results/results.csv') as f:
    rows = [r for r in csv.DictReader(f) if not r['run'].startswith(('200', '201', '202')) and r.get('complete') == 'True']

for split_name in ['LOSO 26', 'LOSO 21', 'grouped 4-fold']:
    sub = [r for r in rows if r['split'] == split_name]
    sub = sorted(sub, key=lambda x: float(x.get('UF1', 0) or 0), reverse=True)
    print(f"=== TOP 5 FOR {split_name} ===")
    for r in sub[:5]:
        print(f"  {r['run']}: UF1={float(r['UF1']):.4f} | UAR={float(r['UAR']):.4f} | ACC={float(r['ACC']):.4f} | backbone={r['backbone']}")
    print()
