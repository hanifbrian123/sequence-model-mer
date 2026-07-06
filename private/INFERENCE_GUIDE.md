# PANDUAN INFERENSI — CASME II Micro-Expression (untuk User & Claude Code Agent pembangun software)

Dokumen ini adalah **spesifikasi lengkap & mandiri** untuk memakai model terlatih di aplikasi.
Ditujukan untuk: (a) Anda menjalankan inferensi cepat, dan (b) agent Claude Code yang membangun software integrasinya.
**Ikuti kontrak preprocessing PERSIS** — kalau input tidak diproses sama seperti saat training, prediksi akan ngawur.

---

## 0. TL;DR (jalan dalam 30 detik)

```bash
conda activate facesleuth   # env berisi torch, torchvision, opencv-contrib

# SINGLE model (RINGAN, direkomendasikan untuk aplikasi realtime)
python src/infer.py --frames <folder_klip> --models models/emotion_single

# ENSEMBLE (akurasi maksimum, 4 model, lebih berat)
python src/infer_ensemble.py --frames <folder_klip> --models_root models/emotion_ensemble
```
Output: kelas prediksi + probabilitas tiap kelas.
`<folder_klip>` = folder berisi frame **wajah ter-crop** satu ekspresi (onset→offset), nama file terurut.

---

## 1. Apa yang tersedia

| Path | Isi | Untuk |
|---|---|---|
| `models/emotion_single/` | 1 model r3d_18 + `deploy.json` | **Emotion 5-kelas — REKOMENDASI app/realtime** |
| `models/emotion_ensemble/{r3d,mc3,focal,r2p1d4ch}/` | 4 model + `deploy.json` masing-masing | Emotion — akurasi maks (offline) |
| `models/objective6_single/` | 1 model r3d_18 | Objective 6-kelas — single |
| `models/objective6_ensemble/…` | 4 model | Objective — akurasi maks |

**Kelas Emotion (5):** `happiness, disgust, repression, surprise, others`
**Kelas Objective (6):** `obj1, obj2, obj3, obj4, obj5, obj7` (AU-defined; obj6 dibuang krn 1 sampel)
> Nama kelas SELALU dibaca dari `deploy.json` (`class_names`) — jangan hardcode.

**Akurasi realistis (subjek baru, LOSO):** Emotion ~**0.69 ACC / 0.72 UF1**; Objective ~**0.70 ACC / 0.59 UF1**. (Angka demo pada sampel latih akan terlihat lebih tinggi — itu optimistik, jangan dijadikan patokan.)

---

## 2. KONTRAK INPUT (WAJIB dipenuhi aplikasi)

Model **TIDAK** menerima video mentah. Aplikasi harus menyediakan **satu klip micro-expression** sebagai urutan frame:

1. **Wajah sudah di-crop & align** — sama seperti dataset CASME II `Cropped` (wajah memenuhi frame, posisi mata relatif stabil). Aplikasi butuh **face detector + aligner** di depan (mis. dlib/mediapipe/RetinaFace). Frame utuh berisi latar akan menurunkan akurasi drastis.
2. **Satu klip = satu ekspresi**, urutan **onset → offset** (dari netral sampai selesai). 
3. **Frame pertama = onset (netral)** — optical flow dihitung relatif ke frame pertama. Ini krusial.
4. Format file: `.jpg/.jpeg/.png`. **Urutan diambil dari angka di nama file** (regex `(\d+)\.(jpg|png)`), jadi beri nama `frame001.jpg, frame002.jpg, …` atau `img_0001.png`. (Kalau tidak ada angka, urutan tak terjamin.)
5. Minimal **2 frame**; ideal belasan–puluhan frame.

---

## 3. PIPELINE PREPROCESSING (persis seperti training — jangan diubah)

Untuk tiap klip, `infer.py` melakukan (semua parameter dibaca dari `deploy.json`):

1. **Baca frame** grayscale, urutkan by angka nama file.
2. **Optical flow TV-L1 onset-referenced:** untuk tiap frame_i hitung `flow(frame_0 → frame_i)` di resolusi `base_size` (144). Hasil `(L, 144, 144, 2)` = medan `[u, v]`.
   - Pakai `cv2.optflow.DualTVL1OpticalFlow_create()` — **butuh opencv-contrib-python**.
3. **Temporal sampling:** ambil `T` (=16) frame merata sepanjang klip (`sample_indices`).
4. **Spatial:** center-crop dari 144 ke `img_size` (=128).
5. **Bangun channel input (`build_flow`)** dan normalisasi ke ~[-1,1]:
   - 3-channel (r3d/mc3/focal): `[u, v, magnitude]`, di-clip `flow_clip=3.0`.
   - 4-channel (model r2p1d4ch): `[u, v, magnitude, strain]`, `strain_clip=0.3`.
6. **Susun tensor** `(C, T, H, W)` → tambah batch dim → `(1, C, T, H, W)`.
7. **TTA ringan:** ulangi untuk view normal + **horizontal-flip** (saat flip, komponen `u` **dinegasikan**). Rata-ratakan softmax.
8. **Ensemble:** untuk `infer_ensemble.py`, hitung flow SEKALI, lalu tiap model membangun input-nya sendiri (3ch/4ch) dan probs dirata-ratakan.

> Model = 3D-CNN video (torchvision `r3d_18`/`mc3_18`/`r2plus1d_18`), input **(B, C, T, H, W)**, pretrained flag `false` (bobot dimuat dari checkpoint).

---

## 4. DEPENDENCIES (environment aplikasi)

```
python 3.10+
torch >= 2.0            # + CUDA opsional (CPU juga jalan, lebih lambat)
torchvision >= 0.15     # menyediakan models.video.r3d_18 dll
opencv-contrib-python   # WAJIB: TV-L1 (cv2.optflow) ada di CONTRIB, bukan opencv-python biasa
numpy
```
> ⚠️ **Jangan pakai `opencv-python` biasa** — `cv2.optflow.DualTVL1OpticalFlow_create` tidak ada di situ. Harus `opencv-contrib-python`.
> Checkpoint dilatih dgn torch 2.6/torchvision 0.21 (CUDA). Untuk load di torch lain, arsitektur torchvision standar → kompatibel; kalau ada warning `weights_only`, load dengan `torch.load(..., map_location=...)` (sudah begitu di `infer.py`).

---

## 5. INTEGRASI PROGRAMATIK (untuk agent — copy-paste)

CLI di atas cukup untuk tes. Untuk software, **impor fungsi** dan bungkus jadi API. Contoh fungsi `predict_clip` mandiri (reuse util yang sudah ada di `src/`):

```python
# app_predict.py  (letakkan di root repo agar 'src' importable, atau sesuaikan sys.path)
import os, sys, json, numpy as np, torch, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dataset import build_flow, sample_indices, _to_cthw
from models import build_model
from infer import list_frames, compute_onset_flow   # reuse

class MicroExpressionModel:
    def __init__(self, models_dir, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # models_dir bisa = satu model (deploy.json langsung) ATAU folder berisi sub-model (ensemble)
        subdirs = [os.path.join(models_dir, d) for d in sorted(os.listdir(models_dir))
                   if os.path.isdir(os.path.join(models_dir, d))
                   and os.path.exists(os.path.join(models_dir, d, "deploy.json"))]
        self.dirs = subdirs if subdirs else [models_dir]  # ensemble vs single
        self.members = []
        for d in self.dirs:
            dep = json.load(open(os.path.join(d, "deploy.json")))
            cfg = {"backbone": dep["backbone"], "dropout": dep["dropout"],
                   "pretrained": False, "modality": dep["modality"],
                   "in_channels": dep.get("in_channels", 3)}
            for ck in dep["checkpoints"]:
                m = build_model(cfg, dep["num_classes"]).to(self.device).eval()
                m.load_state_dict(torch.load(os.path.join(d, ck), map_location=self.device))
                self.members.append((m, dep))
        self.class_names = self.members[0][1]["class_names"]
        self.base_size = self.members[0][1]["base_size"]

    @torch.no_grad()
    def predict(self, frame_paths):
        """frame_paths: list path frame wajah ter-crop, urut kronologis (frame[0]=onset)."""
        flow = compute_onset_flow(frame_paths, self.base_size)   # (L,base,base,2)
        probs = None
        for m, dep in self.members:
            T, s = dep["T"], dep["img_size"]
            idx = sample_indices(flow.shape[0], T, train=False)
            clip = flow[idx]; H, W = clip.shape[1], clip.shape[2]
            top, left = (H - s) // 2, (W - s) // 2
            for flip in (False, True):                            # light TTA
                x = build_flow(clip, top, left, s, flip, dep["flow_clip"],
                               third=dep.get("flow_third", "mag"),
                               strain_clip=dep.get("strain_clip", 1.0))
                xt = _to_cthw(x).unsqueeze(0).to(self.device)
                p = torch.softmax(m(xt).float(), 1).cpu().numpy()[0]
                probs = p if probs is None else probs + p
        probs /= (2 * len(self.members))
        i = int(probs.argmax())
        return {"label": self.class_names[i], "confidence": float(probs[i]),
                "probs": {c: float(p) for c, p in zip(self.class_names, probs)}}

# --- pemakaian ---
# model = MicroExpressionModel("models/emotion_single")      # atau models/emotion_ensemble
# res = model.predict(list_frames("path/ke/folder_klip"))    # list_frames = urutkan by angka nama
# print(res["label"], res["confidence"])
```
`MicroExpressionModel` otomatis: 1 folder dengan `deploy.json` → single; folder berisi sub-folder model → ensemble. Ganti `models/emotion_single` → `models/objective6_single` untuk task objective (nama kelas otomatis ikut).

---

## 6. DESAIN INTEGRASI VIDEO REALTIME

Model = **sequence**, jadi cocok realtime (tak perlu deteksi apex — apex-spotting itu rapuh & sudah terbukti buruk):

```
Loop kamera:
  1. Deteksi + align + CROP wajah tiap frame  → simpan ke buffer (ring buffer).
  2. Tentukan "klip" untuk diklasifikasi:
     - Opsi A (paling sederhana): sliding window T_win frame terakhir (mis. 16–32),
       frame paling awal di window = referensi onset.
     - Opsi B (lebih benar): deteksi awal ekspresi (mis. dari gerak/AU) untuk set onset,
       kumpulkan sampai reda, baru klasifikasi klip itu.
  3. Panggil model.predict(frames_window).
  4. (Opsional) smoothing hasil antar-window (mis. majority / EMA probabilitas).
```

**Catatan penting realtime:**
- **Bottleneck = optical flow TV-L1** (~beberapa ratus ms/klip di CPU). Untuk FPS tinggi: hitung flow di thread terpisah, atau klasifikasi tiap N frame (bukan tiap frame), atau batasi resolusi. Mengganti ke Farneband/GPU-flow akan **mismatch** training (dilatih TV-L1) → akurasi turun; kalau terpaksa, re-kalibrasi/latih ulang.
- **Onset/netral:** akurasi terbaik bila frame[0] window benar-benar netral. Sliding-window kasar (frame awal window sbg onset) tetap jalan tapi kurang optimal.
- **Single model** (bukan ensemble) untuk realtime — 4× lebih ringan, akurasi hampir sama.

---

## 7. FORMAT OUTPUT

`predict()` mengembalikan dict:
```json
{
  "label": "happiness",
  "confidence": 0.83,
  "probs": {"happiness":0.83,"disgust":0.05,"repression":0.03,"surprise":0.02,"others":0.07}
}
```
CLI mencetak `PREDICTION: <label> (p=…)` + daftar probabilitas terurut.

---

## 8. GOTCHAS & CATATAN JUJUR

1. **opencv-contrib-python wajib** (TV-L1). Ini penyebab #1 error saat deploy.
2. **Wajah harus ter-crop/align** — ini tanggung jawab aplikasi (face detector+aligner). Model dilatih pada wajah yang sudah rapi.
3. **Frame pertama = onset** (netral). Salah onset → flow salah → prediksi salah.
4. **Nama file harus mengandung angka** untuk pengurutan.
5. **Akurasi realistis ~0.69 (emotion)** pada subjek baru — micro-expression itu sulit & dataset kecil (246 sampel). Jangan berharap angka demo. Target 0.80 TIDAK tercapai (lihat `reports/emotion_report.md` untuk alasan struktural).
6. **Jangan tampilkan gambar dataset** ke UI publik (CASME II berlisensi) — untuk piksel milik user aplikasi sendiri bebas.
7. **CPU vs GPU:** jalan di CPU (lebih lambat, terutama flow). GPU mempercepat forward model, tapi flow tetap di CPU (cv2).

---

## 9. REFERENSI `deploy.json` (field yang dibaca inferensi)

| Field | Arti | Contoh |
|---|---|---|
| `checkpoints` | list file bobot (.pt) di folder itu | `["final_seed42.pt"]` |
| `class_names` | nama kelas terurut index | `["happiness",…]` |
| `num_classes` | jumlah kelas | 5 atau 6 |
| `backbone` | arsitektur torchvision video | `r3d_18` |
| `modality` | selalu `flow` | `flow` |
| `T` | jumlah frame temporal | 16 |
| `img_size` | ukuran crop akhir | 128 |
| `base_size` | resolusi flow dihitung | 144 |
| `flow_clip` | skala normalisasi u,v,mag | 3.0 |
| `flow_third` | channel ke-3: `mag` atau `both`(→4ch) | `mag` |
| `strain_clip` | skala strain (kalau 4ch) | 0.3 |
| `in_channels` | 3 atau 4 | 3 |
| `tta` | jumlah TTA saat training (inferensi app pakai 2: center+flip) | 5 |

---

## 10. Checklist untuk agent pembangun software

- [ ] Env punya `opencv-contrib-python` (bukan opencv-python biasa) + torch + torchvision.
- [ ] Pipeline app: kamera → face detect → **align + crop** → buffer frame.
- [ ] Bungkus `MicroExpressionModel` (§5) jadi service; muat model SEKALI saat start.
- [ ] Realtime: single model, sliding window, flow di thread terpisah, klasifikasi tiap N frame.
- [ ] Tampilkan `label` + `confidence`; pertimbangkan threshold confidence & smoothing antar-window.
- [ ] Uji dulu dengan `python src/infer.py --frames <folder> --models models/emotion_single`.
- [ ] Set ekspektasi akurasi ~0.69 (subjek baru), bukan angka demo.
