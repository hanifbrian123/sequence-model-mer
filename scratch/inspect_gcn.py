import os
import sys

GCN_MODEL_PATH = r"D:\AI-Projects\emotion Detection - v1\core\gcn_gru_model.py"
EXTERNAL_ROOT = r"D:\AI-Projects\emotion Detection - v1"

print("=" * 80)
print("INSPECTING GCN MODEL & GRAPH DATA")
print("=" * 80)

if os.path.exists(GCN_MODEL_PATH):
    print(f"Found GCN model at: {GCN_MODEL_PATH}")
    with open(GCN_MODEL_PATH, "r", encoding="utf-8") as f:
        code = f.read()
    print(f"Total lines: {len(code.splitlines())}")
    with open(r"scratch\gcn_gru_model_extracted.py", "w", encoding="utf-8") as f:
        f.write(code)
    print("Saved code to scratch/gcn_gru_model_extracted.py")
else:
    print("GCN model not found at", GCN_MODEL_PATH)

# Check for precomputed graph folders
print("\nSearching for graph directories in external repo:")
for root, dirs, files in os.walk(EXTERNAL_ROOT):
    for d in dirs:
        if "graph" in d.lower() or "landmark" in d.lower() or "gcn" in d.lower():
            p = os.path.join(root, d)
            num_files = len(os.listdir(p)) if os.path.exists(p) else 0
            print(f"  Found dir: {p} ({num_files} items)")
