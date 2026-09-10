"""Build & validate the CASME II sample index from the coding xlsx.

Produces a CSV with one row per usable sample:
  subject, sub_folder, filename, onset, apex, offset, emotion, label, n_frames_avail

Standard 5-class protocol on Estimated Emotion:
  {happiness, disgust, repression, surprise, others}  (246 samples)
"repression" spelled as in the xlsx. sadness/fear dropped (too few).

Does NOT open image pixels — only lists filenames to validate coverage.
"""
import os
import re
import argparse
import pandas as pd

DATASET = r"D:/AI-Projects/casmeII-facesleuth-r/dataset"
CROPPED = os.path.join(DATASET, "Cropped")
CODING = os.path.join(DATASET, "CASME2-coding-20140508.xlsx")
OBJECTIVE = os.path.join(DATASET, "CASME2-ObjectiveClasses.xlsx")

FIVE_CLASS = ["happiness", "disgust", "repression", "surprise", "others"]
FIVE_CLASS_MAP = {c: i for i, c in enumerate(FIVE_CLASS)}

IMG_RE = re.compile(r"reg_img(\d+)\.jpg$", re.IGNORECASE)


def list_frame_numbers(folder):
    """Return sorted list of integer frame numbers present in a sequence folder."""
    nums = []
    if not os.path.isdir(folder):
        return nums
    for fn in os.listdir(folder):
        m = IMG_RE.search(fn)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def build(scheme="emotion", out_csv=None):
    df = pd.read_excel(CODING)
    df = df.rename(columns={
        "Subject": "subject", "Filename": "filename",
        "OnsetFrame": "onset", "ApexFrame": "apex", "OffsetFrame": "offset",
        "Estimated Emotion": "emotion",
    })
    if scheme == "objective":
        obj = pd.read_excel(OBJECTIVE).rename(columns={
            "Subject": "subject", "Filename": "filename", "Objective Class": "obj_class"})
        df = df.merge(obj[["subject", "filename", "obj_class"]],
                      on=["subject", "filename"], how="left")

    rows = []
    dropped = {"wrong_class": 0, "missing_folder": 0, "bad_frames": 0, "no_frames": 0}
    for _, r in df.iterrows():
        subj = int(r["subject"])
        sub_folder = f"sub{subj:02d}"
        fname = str(r["filename"]).strip()
        emo = str(r["emotion"]).strip().lower()

        if scheme == "emotion":
            if emo not in FIVE_CLASS_MAP:
                dropped["wrong_class"] += 1
                continue
            label = FIVE_CLASS_MAP[emo]
        else:  # objective: 7 classes 1..7 -> 0..6
            if pd.isna(r.get("obj_class")):
                dropped["wrong_class"] += 1
                continue
            label = int(r["obj_class"]) - 1

        folder = os.path.join(CROPPED, sub_folder, fname)
        if not os.path.isdir(folder):
            dropped["missing_folder"] += 1
            print(f"[MISSING FOLDER] {sub_folder}/{fname}")
            continue

        frames = list_frame_numbers(folder)
        if not frames:
            dropped["no_frames"] += 1
            print(f"[NO FRAMES] {sub_folder}/{fname}")
            continue

        # parse onset/apex/offset robustly (some apex may be '/' or missing)
        def to_int(x, default):
            try:
                return int(x)
            except (ValueError, TypeError):
                return default
        onset = to_int(r["onset"], frames[0])
        offset = to_int(r["offset"], frames[-1])
        apex = to_int(r["apex"], -1)

        # clamp onset/offset to available frame range
        lo, hi = frames[0], frames[-1]
        onset_c = max(onset, lo)
        offset_c = min(offset, hi)
        if offset_c <= onset_c:
            # fall back to full available range
            onset_c, offset_c = lo, hi
        n_avail = sum(1 for f in frames if onset_c <= f <= offset_c)
        if n_avail < 2:
            dropped["bad_frames"] += 1
            print(f"[BAD FRAMES] {sub_folder}/{fname} onset={onset} offset={offset} avail={frames[:3]}..{frames[-3:]}")
            continue

        rows.append({
            "subject": subj, "sub_folder": sub_folder, "filename": fname,
            "onset": onset_c, "apex": apex, "offset": offset_c,
            "frame_lo": lo, "frame_hi": hi,
            "emotion": emo, "label": label, "n_frames_avail": n_avail,
        })

    out = pd.DataFrame(rows)
    if out_csv:
        out.to_csv(out_csv, index=False)
    print("\n=== SUMMARY ===")
    print("scheme:", scheme)
    print("usable samples:", len(out))
    print("dropped:", dropped)
    print("subjects:", sorted(out["subject"].unique().tolist()))
    print("label distribution:")
    print(out["label"].value_counts().sort_index().to_string())
    if scheme == "emotion":
        print("class names:", FIVE_CLASS)
    print("frames/seq stats: min=%d max=%d mean=%.1f" % (
        out["n_frames_avail"].min(), out["n_frames_avail"].max(), out["n_frames_avail"].mean()))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="emotion", choices=["emotion", "objective"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build(args.scheme, args.out)
