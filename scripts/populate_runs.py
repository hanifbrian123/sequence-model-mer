"""Populate runs/ with cleanly numbered NNN_nama junctions pointing to experiments.

Ensures runs/ looks and acts exactly like fer/runs/:
- Every folder is NNN_nama
- 0 disk overhead via directory junctions
- All artifacts (summary.json, probs.npz, etc.) accessible under standard path
"""
import os
import re
import subprocess

def to_numbered(name):
    m = re.match(r'^iter_(\d+)([a-z0-9_]*)$', name)
    if m:
        num = int(m.group(1))
        suffix = m.group(2)
        return f'{num:03d}{suffix}' if suffix else f'{num:03d}_iter_{num:02d}'
    if name.startswith('smoke'):
        return f'000_{name}'
    if name.startswith('apex_'):
        return f'000_{name}'
    if name.startswith('obj6_'):
        return f'013_{name}'
    if name.startswith('fusion10'):
        return f'100_{name}'
    return name

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(root, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    p2 = os.path.join(root, "experiments", "protocol_v2")
    count = 0

    if os.path.exists(p2):
        for entry in sorted(os.listdir(p2)):
            src_path = os.path.join(p2, entry)
            if not os.path.isdir(src_path) or entry in ('comparisons', 'fusions'):
                continue
            new_name = to_numbered(entry)
            dst_path = os.path.join(runs_dir, new_name)
            if not os.path.exists(dst_path):
                cmd = f'cmd /c mklink /J "{dst_path}" "{src_path}"'
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if res.returncode == 0:
                    count += 1

    exp_dir = os.path.join(root, "experiments")
    for entry in sorted(os.listdir(exp_dir)):
        src_path = os.path.join(exp_dir, entry)
        if not os.path.isdir(src_path) or entry == 'protocol_v2':
            continue
        new_name = to_numbered(entry)
        dst_path = os.path.join(runs_dir, new_name)
        if not os.path.exists(dst_path):
            cmd = f'cmd /c mklink /J "{dst_path}" "{src_path}"'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode == 0:
                count += 1

    print(f"Successfully populated {count} runs in {runs_dir}.")

if __name__ == "__main__":
    main()
