import os
import sys
import json

NB_DIR = r"D:\AI-Projects\emotion Detection - v1\train"
OUT_PATH = r"scratch\deep_inspection_results.txt"

def inspect_notebook(nb_name, out_f):
    path = os.path.join(NB_DIR, nb_name)
    out_f.write("=" * 80 + "\n")
    out_f.write(f"DETAILS FOR: {nb_name}\n")
    out_f.write("=" * 80 + "\n")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        outputs = c.get("outputs", [])
        
        has_output = len(outputs) > 0
        is_relevant = any(k in src.lower() for k in ["dataset", "model", "loso", "epochs", "active_classes", "selected_datasets", "cross_validation", "report", "confusion"])
        if is_relevant or has_output:
            out_f.write(f"\n--- Cell {i} [{c.get('cell_type')}] ---\n")
            lines = [l for l in src.splitlines() if l.strip()]
            for l in lines[:8]:
                out_f.write(f"  src: {l}\n")
            if len(lines) > 8:
                out_f.write(f"  ... ({len(lines)-8} more lines)\n")
            if outputs:
                out_f.write("  >>> OUTPUT:\n")
                for o in outputs:
                    if o.get("output_type") == "stream":
                        out_lines = "".join(o.get("text", [])).splitlines()
                        for ol in out_lines[:15]:
                            out_f.write(f"    {ol}\n")
                        if len(out_lines) > 15:
                            out_f.write(f"    ... ({len(out_lines)-15} lines truncated, last line: {out_lines[-1]})\n")
                    elif "data" in o and "text/plain" in o["data"]:
                        out_lines = "".join(o["data"]["text/plain"]).splitlines()
                        for ol in out_lines[:5]:
                            out_f.write(f"    {ol}\n")
                    elif o.get("output_type") == "error":
                        out_f.write(f"    ERROR: {o.get('ename')}: {o.get('evalue')}\n")

with open(OUT_PATH, "w", encoding="utf-8") as out_f:
    for name in [
        "ViViT_Training_Testing_LOSO.ipynb",
        "ViViT_Training_Testing_LOSO copy.ipynb",
        "ViViT_Training_Testing copy.ipynb",
        "ViViT_Training_Testing.ipynb",
        "ViViT_Training_Testing copy 2.ipynb",
        "ViViT_Training_Testing - GCN.ipynb"
    ]:
        inspect_notebook(name, out_f)

print(f"Saved UTF-8 to {OUT_PATH}")
