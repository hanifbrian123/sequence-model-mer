import os
import json
import glob

NB_DIR = r"D:\AI-Projects\emotion Detection - v1\train"
OUTPUT_FILE = r"scratch\notebooks_analysis.txt"

notebooks = [
    "ViViT_Training_Testing - GCN.ipynb",
    "ViViT_Training_Testing copy 2.ipynb",
    "ViViT_Training_Testing copy.ipynb",
    "ViViT_Training_Testing.ipynb",
    "ViViT_Training_Testing_LOSO copy.ipynb",
    "ViViT_Training_Testing_LOSO.ipynb"
]

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for nb_name in notebooks:
        path = os.path.join(NB_DIR, nb_name)
        out_f.write(f"\n{'#'*80}\n")
        out_f.write(f"NOTEBOOK: {nb_name}\n")
        out_f.write(f"{'#'*80}\n\n")
        
        if not os.path.exists(path):
            out_f.write(f"FILE NOT FOUND: {path}\n")
            continue
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                nb = json.load(f)
        except Exception as e:
            out_f.write(f"ERROR READING: {e}\n")
            continue
            
        cells = nb.get("cells", [])
        out_f.write(f"Total cells: {len(cells)}\n")
        
        # Scan for dataset references
        full_text = ""
        for cell in cells:
            full_text += "".join(cell.get("source", [])) + "\n"
            
        dataset_detected = "Unknown"
        if "casme" in full_text.lower():
            dataset_detected = "CASME II detected"
        if "samm" in full_text.lower():
            dataset_detected += " / SAMM detected"
        if "smic" in full_text.lower():
            dataset_detected += " / SMIC detected"
            
        out_f.write(f"Dataset clues: {dataset_detected}\n")
        
        # Extract cells with outputs, especially evaluation/results or model definitions
        for i, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "")
            src = "".join(cell.get("source", []))
            outputs = cell.get("outputs", [])
            
            # Check if cell is interesting (imports, model, dataset path, train loop, evaluation, results)
            interesting = False
            keywords = ["dataset", "casme", "model", "vivit", "gcn", "loso", "accuracy", "classification_report", "confusion_matrix", "epoch", "val_acc", "test", "evaluate"]
            if any(k in src.lower() for k in keywords):
                interesting = True
            if outputs:
                out_str = ""
                for o in outputs:
                    if o.get("output_type") == "stream":
                        out_str += "".join(o.get("text", []))
                    elif "data" in o and "text/plain" in o["data"]:
                        out_str += "".join(o["data"]["text/plain"])
                if any(k in out_str.lower() for k in ["accuracy", "precision", "recall", "f1", "epoch", "loss", "fold", "cm", "report"]):
                    interesting = True
            
            if interesting:
                out_f.write(f"\n--- Cell {i} [{cell_type}] ---\n")
                # write first 15 lines of source if long, or full if short
                lines = src.splitlines()
                if len(lines) > 20:
                    out_f.write("\n".join(lines[:10]) + "\n... [truncated] ...\n" + "\n".join(lines[-5:]) + "\n")
                else:
                    out_f.write(src + "\n")
                    
                if outputs:
                    out_f.write(">>> OUTPUT:\n")
                    for o in outputs:
                        if o.get("output_type") == "stream":
                            txt = "".join(o.get("text", []))
                            out_f.write(txt)
                        elif "data" in o and "text/plain" in o["data"]:
                            txt = "".join(o["data"]["text/plain"])
                            out_f.write(txt + "\n")
                        elif o.get("output_type") == "error":
                            out_f.write(f"ERROR: {o.get('ename')}: {o.get('evalue')}\n")
                    out_f.write("\n")

print(f"Done analyzing all notebooks. Output saved to {OUTPUT_FILE}")
