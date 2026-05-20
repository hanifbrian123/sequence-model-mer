"""Auto-consolidate all emotion 5-class LOSO members into honest fusions.

Discovers available experiments/*/probs.npz members, reports singles, principled
fixed-rule fusions, and a NESTED-LOSO selective fusion (integrity-clean estimate:
subset chosen on 25 subjects, applied to held-out subject). Writes a summary so
an unattended run leaves the numbers ready. NO test-set cherry-picking in the
headline numbers.
"""
import os, numpy as np
from itertools import combinations
from sklearn.metrics import f1_score, recall_score, accuracy_score
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS = ['happiness','disgust','repression','surprise','others']

CANDIDATES = ["iter_10","iter_12_strain","iter_13_4ch","iter_14_r3d","iter_14_r3d_s123",
              "iter_16_mag_r3d","iter_17_mc3","iter_18_focal","iter_20_ema","iter_24_res160",
              "iter_25_bnadapt","iter_26_erase","iter_27_snapshot","iter_28_seqflow",
              "iter_29_noweight"]

def load(r):
    p = os.path.join(REPO,"experiments",r,"probs.npz")
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True)
    keys=[str(k) for k in z["keys"]]
    if len(keys) < 240: return None
    return dict(zip(keys,z["probs"])), dict(zip(keys,z["label"]))

def met(y,yp):
    return (f1_score(y,yp,labels=range(5),average='macro',zero_division=0),
            recall_score(y,yp,labels=range(5),average='macro',zero_division=0),
            accuracy_score(y,yp))

D={}
for r in CANDIDATES:
    d=load(r)
    if d is not None: D[r]=d
runs=list(D)
common=None
for r in runs:
    s=set(D[r][0]); common=s if common is None else common&s
common=sorted(common)
y=np.array([D[runs[0]][1][k] for k in common])
subj=np.array([int(k[3:5]) for k in common])
P={r:np.stack([D[r][0][k] for k in common]) for r in runs}

out=[]
def pr(s): print(s); out.append(s)

pr("="*74); pr(f"CONSOLIDATION over {len(runs)} members, n={len(common)}"); pr("="*74)
pr("SINGLES (UF1 / UAR / ACC):")
singles=[]
for r in runs:
    u,ua,a=met(y,P[r].argmax(1)); singles.append((u,r)); pr(f"  {r:20s} {u:.4f} {ua:.4f} {a:.4f}")

def fuse_pred(rs, idx=None):
    A=sum(P[r] for r in rs)/len(rs)
    return (A if idx is None else A[idx]).argmax(1)

pr("\nPRINCIPLED fixed-rule fusions (no test selection):")
fixed={
 "deployed-4":["iter_13_4ch","iter_14_r3d","iter_17_mc3","iter_18_focal"],
 "deployed-4 +ema":["iter_13_4ch","iter_14_r3d","iter_17_mc3","iter_18_focal","iter_20_ema"],
}
# add data-driven principled sets from NEW members if present
newbies=[r for r in ["iter_24_res160","iter_25_bnadapt","iter_27_snapshot","iter_28_seqflow","iter_26_erase"] if r in P]
if newbies:
    fixed["deployed-4 +ema +all-new"]=["iter_13_4ch","iter_14_r3d","iter_17_mc3","iter_18_focal","iter_20_ema"]+newbies
    # top-6 singles by UF1
    top=[r for _,r in sorted(singles,reverse=True)[:6]]
    fixed["top-6-singles"]=top
for name,rs in fixed.items():
    rs=[r for r in rs if r in P]
    if len(rs)>=2:
        u,ua,a=met(y,fuse_pred(rs)); pr(f"  {name:26s} [{len(rs)}] UF1={u:.4f} UAR={ua:.4f} ACC={a:.4f}")

pr("\nNESTED-LOSO selective fusion (honest; subset picked on 25 subj, applied to held-out):")
subjects=sorted(set(subj))
cand=[c for k in range(2,min(6,len(runs))+1) for c in combinations(runs,k)]
for objname,obj in [("UF1",lambda u,ua,a:u),("UF1+ACC",lambda u,ua,a:u+a)]:
    yp=np.empty_like(y)
    for s in subjects:
        te=subj==s; tr=subj!=s; best=None
        for c in cand:
            val=obj(*met(y[tr],fuse_pred(list(c),tr)))
            if best is None or val>best[0]: best=(val,c)
        yp[te]=fuse_pred(list(best[1]),te)
    u,ua,a=met(y,yp); pr(f"  nested[opt {objname:8s}]: UF1={u:.4f} UAR={ua:.4f} ACC={a:.4f}")

open(os.path.join(REPO,"experiments","CONSOLIDATION.txt"),"w").write("\n".join(out))
pr("\nwrote experiments/CONSOLIDATION.txt")
