import os
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

NB_DIR = r"D:\AI-Projects\emotion Detection - v1\train"

notebooks = [
    "ViViT_Training_Testing - GCN.ipynb",
    "ViViT_Training_Testing copy 2.ipynb",
    "ViViT_Training_Testing copy.ipynb",
    "ViViT_Training_Testing.ipynb",
    "ViViT_Training_Testing_LOSO copy.ipynb",
    "ViViT_Training_Testing_LOSO.ipynb"
]

for nb_name in notebooks:
    path = os.path.join(NB_DIR, nb_name)
    print("=" * 80)
    print(f"NOTEBOOK: {nb_name}")
    if not os.path.exists(path):
        print("  FILE NOT FOUND")
        continue
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    cells_with_output = [c for c in code_cells if c.get("outputs")]
    print(f"  Total cells: {len(cells)} | Code cells: {len(code_cells)} | Executed cells with outputs: {len(cells_with_output)}")
    
    # Check dataset selection in code
    full_text = "\n".join(["".join(c.get("source", [])) for c in cells])
    datasets_selected = []
    for line in full_text.splitlines():
        if "SELECTED_DATASETS" in line:
            datasets_selected.append(line.strip())
        if "ACTIVE_CLASSES" in line:
            datasets_selected.append(line.strip())
    print(f"  Configuration found: {datasets_selected[:3]}")
    
    # Check outputs for key evaluation metrics
    eval_outputs = []
    for c in cells_with_output:
        out_text = ""
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                out_text += "".join(o.get("text", []))
            elif "data" in o and "text/plain" in o["data"]:
                out_text += "".join(o["data"]["text/plain"])
        for kw in ["accuracy", "classification report", "best val", "final", "confusion matrix", "loso", "fold"]:
            if kw in out_text.lower():
                for l in out_text.splitlines():
                    if any(k in l.lower() for k in ["accuracy", "val f1", "val acc", "macro avg", "weighted avg", "best val", "fold", "overall", "precision", "recall"]):
                        eval_outputs.append(l.strip())
                break
    if eval_outputs:
        print("  Key Output Snippets:")
        for line in eval_outputs[:15]:
            print(f"    {line}")
    else:
        print("  No evaluation output found.")
