"""Move root loose shell scripts, CSVs, and app scripts to their designated directories.
"""
import glob
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def cleanup():
    # 1. Move run_loso26.sh and run_all_remaining.sh to scripts/
    for special in ["run_loso26.sh", "run_all_remaining.sh"]:
        src = os.path.join(ROOT, special)
        dst = os.path.join(ROOT, "scripts", special)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {special} to scripts/")

    # 2. Move old run_queue*.sh to scripts/legacy_queues/
    for sh_file in glob.glob(os.path.join(ROOT, "run_queue*.sh")):
        fname = os.path.basename(sh_file)
        dst = os.path.join(ROOT, "scripts", "legacy_queues", fname)
        shutil.move(sh_file, dst)
        print(f"Moved {fname} to scripts/legacy_queues/")

    # 3. Copy results CSVs to results/ if not present, then remove from root
    for csv_name in ["results_ledger.csv", "results_protocol_v2.csv"]:
        src = os.path.join(ROOT, csv_name)
        dst = os.path.join(ROOT, "results", csv_name)
        if os.path.exists(src):
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            os.remove(src)
            print(f"Moved {csv_name} to results/")

    # 4. Remove root app_predict.py (already saved in deploy/ and src/casme/serving/)
    root_app = os.path.join(ROOT, "app_predict.py")
    if os.path.exists(root_app):
        deploy_app = os.path.join(ROOT, "deploy", "app_predict.py")
        if not os.path.exists(deploy_app):
            shutil.copy2(root_app, deploy_app)
        os.remove(root_app)
        print("Moved app_predict.py to deploy/")

    # 5. Remove root __pycache__ if present
    root_pycache = os.path.join(ROOT, "__pycache__")
    if os.path.exists(root_pycache):
        shutil.rmtree(root_pycache, ignore_errors=True)
        print("Removed root __pycache__")

    print("Root cleanup completed successfully.")

if __name__ == "__main__":
    cleanup()
