import os, json, glob
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

results_df = pd.read_csv('results/results.csv')

selected_runs = [
    # Ch 1: Input Representation
    ('Ch 1: Input Representation', '002_baseline_r2plus1d', 'R(2+1)D-18 Raw RGB Baseline', 'Identity overfitting on static appearance; network memorizes subject facial morphology rather than motion.'),
    ('Ch 1: Input Representation', '003_onsetref_r2plus1d', 'R(2+1)D-18 Frame Differencing ($I_t - I_0$)', 'Onset frame subtraction eliminates static texture bias, yielding immediate +24% ACC surge.'),
    ('Ch 1: Input Representation', '004_flow_r2plus1d', 'R(2+1)D-18 TV-L1 Onset Flow ($u, v$)', 'Major representation leap; dense optical flow captures subtle sub-pixel facial muscle displacement.'),
    ('Ch 1: Input Representation', '006_twostream_flow_onsetref', 'R(2+1)D-18 Two-Stream (RGB + TV-L1 Flow)', 'Adding RGB appearance stream degrades UF1 from 0.6601 to 0.6390; confirms appearance is noise in LOSO.'),
    ('Ch 1: Input Representation', '045_seqflow', 'R(2+1)D-18 Sequential Flow ($t \\to t+1$)', 'Adjacent frame flow collapses; proves absolute deformation trajectory from onset is required for micro-movements.'),
    
    # Ch 2: Spatiotemporal Backbone
    ('Ch 2: Spatiotemporal Backbone', '010_flow_ensemble_tta', 'R(2+1)D-18 + Multi-Crop TTA Ensemble', 'Multi-scale spatial pooling and test-time augmentation elevates R(2+1)D baseline performance.'),
    ('Ch 2: Spatiotemporal Backbone', '020_mc3', 'MC3-18 Mixed Convolution Backbone', 'Mixed 3D/2D convolutions achieve competitive spatiotemporal modeling (UF1 0.6990).'),
    ('Ch 2: Spatiotemporal Backbone', '037_resnetgru', 'ResNet-18 + GRU Hybrid Backbone', 'Sequential RNN architecture struggles to maintain fine spatiotemporal gradients compared to pure 3D CNNs.'),
    ('Ch 2: Spatiotemporal Backbone', '017_r3d', 'R3D-18 Full 3D Spatiotemporal CNN', 'Decisive backbone winner; 3D spatiotemporal kernels achieve 0.7145 UF1 on full LOSO 26 benchmark.'),
    
    # Ch 3: The Apex Dilemma
    ('Ch 3: The Apex Dilemma', '068_fusion_baseline_iter43_50full_50apex', 'R3D-18 50:50 Oracle Fusion (Full + GT Apex)', 'Theoretical ceiling (UF1 0.7104) leveraging manual dataset ground-truth apex annotations.'),
    ('Ch 3: The Apex Dilemma', '091_r3d_auto_apex_s42_v2_dev_p5', 'R3D-18 Auto-Apex Single Model', 'Label-free optical flow energy peak detector estimates apex timing without ground-truth annotations.'),
    ('Ch 3: The Apex Dilemma', '096_fusion_deployable_50full_50auto47', 'R3D-18 50:50 Deployable Fusion (Full + Auto-Apex)', 'Production deployable champion; closes 95% of oracle gap (UF1 0.7021) without manual apex labels.'),
    
    # Ch 4: Anatomical Region Attention & Head Motion
    ('Ch 4: Anatomical Region & Head Motion', '102_r3d_hq_stabilized_v2_dev_p5', 'R3D-18 with Rigid Head Stabilization (ECC)', 'Catastrophic collapse (-9 pts UF1); demonstrates subtle head motions carry intrinsic affective micro-signals.'),
    ('Ch 4: Anatomical Region & Head Motion', '163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5', 'R3D-18 + Multi-Region Soft Attention (LOSO 21)', 'Landmark-guided spatial attention on eyes/brows/mouth boosts dev score to 0.7283 UF1.'),
    ('Ch 4: Anatomical Region & Head Motion', '182_r3d_focus_region_full_s42_v2_dev_loso_all_p5', 'R3D-18 + Facial Region Focus Attention (LOSO 26)', 'Validates anatomical region focus on full 26 subjects, achieving 0.7121 UF1 / 69.51% ACC.'),
    
    # Ch 5: The Champion Paradigm
    ('Ch 5: The Champion Paradigm', '156_r3d_au01_full_s42_v2_dev_p5', 'R3D-18 Multi-Task AU ($\\lambda_{\\text{AU}}=0.01$)', 'Initial multi-task screening: small AU regularization weight shows modest gain on validation folds.'),
    ('Ch 5: The Champion Paradigm', '140_r3d_au05_full_s42_v2_dev_p5', 'R3D-18 Multi-Task AU ($\\lambda_{\\text{AU}}=0.5$, Dev 21)', 'Optimal AU loss weighting balances emotional classification and anatomical muscle signals.'),
    ('Ch 5: The Champion Paradigm', '188_r3d_au05_full_s42_v2_dev_loso_all_p5', 'R3D-18 Multi-Task 11-AU Champion (LOSO 26)', 'ALL-TIME REPOSITORY CHAMPION: 0.7211 UF1, 0.7378 UAR, 69.92% ACC, 0.8949 AUC on full 26 folds.'),
    
    # Ch 6: Paradigm Audit & Competing Models
    ('Ch 6: Paradigm Audit & Competing Models', '200_expA_flow_on_replica_v2_dev_loso_all_p5', 'External Paper Replica (Non-LOSO Leakage Audit)', 'Audits external 88% claim; in genuine LOSO drops to 53.89% UF1 due to subject identity leakage in random splits.'),
    ('Ch 6: Paradigm Audit & Competing Models', '201_expB_vivit_on_megc_v2_dev_loso_all_p5', 'CNN-ViViT Spatiotemporal Transformer', 'Severe collapse (0.2093 UF1 / 28.05% ACC); Transformer lacks inductive bias for small-sample ME.'),
    ('Ch 6: Paradigm Audit & Competing Models', '202_expC_gcngru_on_megc_v2_dev_loso_all_p5', 'GCN-GRU Facial Landmark Graph Model', 'Failure mode (0.1799 UF1 / 17.89% ACC); sparse landmark coordinates miss subtle sub-pixel texture flux.'),
    ('Ch 6: Paradigm Audit & Competing Models', '203_expD_flow_strain_v2_dev_loso_all_p5', 'R3D-18 + Flow & Strain Tensor ($\\varepsilon_{xx}, \\varepsilon_{yy}$)', 'Strong physical baseline (0.6938 UF1); demonstrates biological AU supervision beats mechanical strain.')
]

rows = []
for ch, r_name, desc, takeaway in selected_runs:
    run_dir = os.path.join('runs', r_name)
    prefix = r_name.split('_')[0]
    
    cfg_candidates = glob.glob(f'configs/{prefix}_*.json')
    cfg_file = os.path.basename(cfg_candidates[0]) if cfg_candidates else f'{r_name}.json'
    cfg_rel = f'../configs/{cfg_file}' if os.path.exists(os.path.join('configs', cfg_file)) else f'../runs/{r_name}/config.json'
    
    sum_path = os.path.join(run_dir, 'summary.json')
    s_data = {}
    if os.path.exists(sum_path):
        with open(sum_path) as f:
            s_data = json.load(f)
            
    match = results_df[results_df['run'].astype(str).str.startswith(prefix)]
    
    acc = s_data.get('accuracy')
    if acc is None and not match.empty:
        acc = match.iloc[0]['ACC']
        
    uf1 = s_data.get('macro_f1') or s_data.get('UF1')
    if uf1 is None and not match.empty:
        uf1 = match.iloc[0]['UF1']
        
    uar = s_data.get('macro_recall') or s_data.get('UAR') or s_data.get('balanced_accuracy')
    if uar is None and not match.empty:
        uar = match.iloc[0]['UAR']
        
    split = match.iloc[0]['split'] if not match.empty else ('LOSO 26' if 'all' in r_name or 'loso_all' in r_name else 'Grouped 4-fold')
    protocol = match.iloc[0]['protocol'] if not match.empty else 'protocol_v2'
    backbone = match.iloc[0]['backbone'] if not match.empty else ('r3d_18' if 'r3d' in r_name else 'r2plus1d_18')
    
    auc_val = None
    probs_path = os.path.join(run_dir, 'probs.npz')
    if os.path.exists(probs_path):
        d_p = np.load(probs_path)
        probs = d_p['probs']
        labels = d_p['label']
        nc = probs.shape[1]
        labels_bin = label_binarize(labels, classes=list(range(nc)))
        fpr = dict()
        tpr = dict()
        for i in range(nc):
            if np.sum(labels_bin[:, i]) > 0:
                fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nc) if i in fpr]))
        mean_tpr = np.zeros_like(all_fpr)
        valid_c = [i for i in range(nc) if i in fpr]
        for i in valid_c:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(valid_c)
        auc_val = auc(all_fpr, mean_tpr)
        
    files = set(os.listdir(run_dir)) if os.path.exists(run_dir) else set()
    
    art_links = []
    if 'confusion_matrix.png' in files:
        art_links.append(f'[CM](../runs/{r_name}/confusion_matrix.png)')
    if 'roc_curve.png' in files:
        art_links.append(f'[ROC](../runs/{r_name}/roc_curve.png)')
    if 'curve_training.png' in files:
        art_links.append(f'[Curves](../runs/{r_name}/curve_training.png)')
    if 'per_fold.csv' in files:
        art_links.append(f'[Folds](../runs/{r_name}/per_fold.csv)')
    elif 'predictions.csv' in files:
        art_links.append(f'[Preds](../runs/{r_name}/predictions.csv)')
    if 'run.log' in files:
        art_links.append(f'[Log](../runs/{r_name}/run.log)')
    art_str = ' • '.join(art_links)
    
    rows.append({
        'chapter': ch,
        'prefix': prefix,
        'run_dir': r_name,
        'desc': desc,
        'split': split,
        'protocol': protocol,
        'backbone': backbone,
        'acc': acc,
        'uf1': uf1,
        'uar': uar,
        'auc': auc_val,
        'cfg_rel': cfg_rel,
        'art_str': art_str,
        'takeaway': takeaway
    })

with open('scratch/table_data.json', 'w') as f:
    json.dump(rows, f, indent=2)

print('Successfully exported scratch/table_data.json!')
