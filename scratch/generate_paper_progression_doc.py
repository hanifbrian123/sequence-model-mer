import json

with open('scratch/table_data.json') as f:
    rows = json.load(f)

# Update takeaways with clear, natural, and simple Indonesian text
takeaways_id = {
    '002': "Model menghafal bentuk wajah orang (overfitting identitas), bukan gerakan ototnya; akurasi sangat rendah.",
    '003': "Mengurangi frame awal langsung menghapus bentuk wajah statis, akurasi naik drastis sebesar +24%.",
    '004': "Optical flow menangkap pergeseran halus otot wajah dengan sangat baik; UF1 naik signifikan ke 0,6601.",
    '006': "Menambah gambar wajah asli (RGB) justru menurunkan skor; membuktikan foto wajah bertindak sebagai gangguan.",
    '045': "Flow antar frame berurutan gagal karena gerakannya terlalu kecil dan tertutup oleh noise kamera.",
    '010': "Kombinasi multi-skala dan variasi pemotongan (crop) gambar meningkatkan kestabilan model R(2+1)D.",
    '020': "Konvolusi campuran 3D/2D cukup kompetitif (UF1 0,6990), tetapi masih di bawah 3D penuh.",
    '037': "Gabungan CNN dan RNN (GRU) sulit menangkap detail perubahan gerak mikro dibanding 3D CNN murni.",
    '017': "Juara arsitektur tunggal; konvolusi 3D murni paling efektif membaca gerak ruang dan waktu secara utuh.",
    '068': "Batas performa maksimal jika dibantu contekan label titik puncak (apex) manual dari manusia.",
    '091': "Deteksi titik puncak (apex) otomatis berbasis energi gerakan tanpa butuh bantuan label manusia.",
    '096': "Model siap pakai untuk aplikasi nyata; menutup 95% jarak ke model contekan manusia tanpa butuh label.",
    '102': "Mengunci gerakan kepala secara kaku justru merusak hasil; gerakan kepala alami membawa sinyal emosi.",
    '163': "Mengarahkan perhatian model ke area mata, alis, dan mulut meningkatkan skor validasi secara meyakinkan.",
    '182': "Fokus pada area wajah utama terbukti konsisten bagus saat diuji pada seluruh 26 orang.",
    '156': "Uji awal multi-task: bobot panduan otot (AU) yang terlalu kecil belum memberi dampak berarti.",
    '140': "Pencarian bobot panduan otot terbaik untuk menyeimbangkan tebakan emosi dan gerakan otot.",
    '188': "MODEL TERBAIK SEPANJANG MASA: Akurasi 69,92%, F1 0,7211, dan AUC 0,8949 pada 26 orang pengujian penuh.",
    '200': "Membongkar klaim akurasi 88% di paper lain; hasilnya anjlok saat diuji jujur tanpa kebocoran data orang.",
    '201': "Vision Transformer (ViViT) gagal total karena datanya sedikit dan modelnya terlalu besar (overfitting parah).",
    '202': "Graf titik landmark (GCN) gagal karena titik kasar tidak bisa melihat kerutan halus pada kulit wajah.",
    '203': "Model berbasis rumus regangan fisik cukup kuat, tetapi tetap kalah dari panduan unit otot biologis (AU)."
}

md = []
md.append("# Perkembangan Eksperimen CASME II & Alur Cerita untuk Paper")
md.append("")
md.append("Dokumen ini menyajikan rangkuman perjalanan eksperimen pengenalan ekspresi mikro (*Micro-Expression Recognition* / MER) pada dataset benchmark **CASME II** (5 kelas emosi standar).")
md.append("")
md.append("Dari total lebih dari 200 percobaan di repositori ini, kami memilih **22 eksperimen kunci** yang disusun ke dalam **6 babak cerita** yang saling menyambung. Setiap angka di tabel ini terhubung langsung ke file log, grafik, dan konfigurasi aslinya agar dapat ditinjau dan diverifikasi kapan saja.")
md.append("")
md.append("---")
md.append("")
md.append("## 1. Tabel Utama Perkembangan Eksperimen (Master Progression Table)")
md.append("")
md.append("Tabel berikut merangkum perjalanan eksperimen dari awal hingga model juara akhir, beserta pengujian model pembanding. Klik tautan pada nomor Run, Config, maupun Artefak untuk membuka file langsung di GitHub:")
md.append("")
md.append("| Babak | Run | Model & Pendekatan | Data Uji | Akurasi (ACC) | Macro-F1 (UF1) | Recall (UAR) | Macro-AUC | File Config | Grafik & Log | Inti Temuan Eksperimen |")
md.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |")

for r in rows:
    ch_short = r['chapter'].split(':')[0].replace("Ch ", "Babak ")
    prefix = r['prefix']
    run_dir = r['run_dir']
    desc = r['desc']
    split = r['split']
    acc_str = f"**{r['acc']*100:.2f}%**" if r['prefix'] == '188' else f"{r['acc']*100:.2f}%"
    uf1_str = f"**{r['uf1']:.4f}**" if r['prefix'] == '188' else f"{r['uf1']:.4f}"
    uar_str = f"**{r['uar']:.4f}**" if r['prefix'] == '188' else f"{r['uar']:.4f}"
    auc_str = f"**{r['auc']:.4f}**" if r['prefix'] == '188' and r['auc'] is not None else (f"{r['auc']:.4f}" if r['auc'] is not None else "-")
    
    cfg_name = r['cfg_rel'].split('/')[-1]
    cfg_link = f"[`{cfg_name}`]({r['cfg_rel']})"
    run_link = f"[`{prefix}`](../runs/{run_dir})"
    takeaway = takeaways_id.get(prefix, r['takeaway'])
    
    md.append(f"| **{ch_short}** | {run_link} | {desc} | {split} | {acc_str} | {uf1_str} | {uar_str} | {auc_str} | {cfg_link} | {r['art_str']} | {takeaway} |")

md.append("")
md.append("---")
md.append("")

# Narrative text
narrative_sections = [
    {
        'title': "2. Babak I: Mencari Bentuk Input Terbaik (Gerakan vs. Foto Wajah)",
        'body': """### Masalah yang Dihadapi
Ekspresi mikro adalah gerakan wajah yang sangat singkat (kurang dari setengah detik) dan pergeserannya sangat tipis (hanya 1–2 piksel). Masalah terbesar saat komputer membaca video wajah adalah: **komputer cenderung menghafal bentuk wajah orangnya**, bukan gerakan ekspresinya. Jika komputer menghafal warna kulit, bentuk mata, atau hidung, maka saat diuji pada wajah orang baru yang belum pernah dilihat, tebakannya akan langsung salah.

### Apa yang Dicoba dan Hasilnya
1. **Memakai Video Wajah Biasa / RGB (`002` — Akurasi 32,93%, UF1 0,3005):**
   Saat model 3D CNN dilatih langsung memakai gambar video asli, akurasinya hanya 32,93%. Komputer terbukti hanya menghafal wajah subjek di data latihan. Begitu bertemu wajah orang baru di data uji, model langsung bingung.
2. **Lompatan Pertama: Mengurangi Gambar dengan Frame Awal (`003` — Akurasi 56,91%, UF1 0,5939):**
   Kami mencoba mengurangkan setiap frame dengan frame pertama saat wajah masih netral ($I_t - I_0$). Dengan cara ini, gambar wajah statis langsung hilang dan hanya menyisakan area yang bergerak. Hasilnya akurasi melesat naik **+23,98%**, membuktikan bahwa menghilangkan foto wajah asli adalah langkah awal yang wajib dilakukan.
3. **Menggunakan Aliran Optik / TV-L1 Optical Flow (`004` — Akurasi 63,82%, UF1 0,6601):**
   Daripada hanya mengurangi piksel, kita hitung vektor pergeseran gerak ($u, v$) dari frame awal netral. Cara ini menangkap gerakan otot mikro dengan sangat halus dan konsisten. Skor UF1 melesat naik menjadi 0,6601.
4. **Mencoba Menggabungkan Foto Wajah + Gerakan Flow (`006` — UF1 0,6390):**
   Secara logika umum, menggabungkan foto asli (RGB) dan arah gerak (Flow) seharusnya membuat model lebih pintar. Namun eksperimen membuktikan hal sebaliknya: **skor UF1 justru turun dari 0,6601 ke 0,6390**. Ini membuktikan secara nyata bahwa pada ekspresi mikro, foto wajah asli justru menjadi pengganggu (*noise*) yang memicu hafalan identitas.
5. **Flow Antar Frame Berurutan Justru Gagal (`045` — Akurasi 52,44%, UF1 0,5602):**
   Jika kita menghitung pergerakan antar frame yang bersebelahan ($t \\to t+1$), hasilnya anjlok. Mengapa? Karena pada video berkecepatan 200 frame per detik, jarak gerak antar dua frame sangat amat kecil sehingga tertutup oleh desah (*noise*) sensor kamera. Pergerakan harus selalu diukur dari titik awal netral (*onset*) agar lintasan geraknya terbaca jelas.

> [!TIP]
> **Poin Penting untuk Paper:** Jangan gunakan foto wajah RGB dan jangan hitung gerak dari frame ke frame bersebelahan. Input terbaik adalah aliran gerak optik (TV-L1) yang diukur dari frame netral awal."""
    },
    {
        'title': "3. Babak II: Memilih Arsitektur Otak Model (3D CNN vs. RNN vs. Model Campuran)",
        'body': """### Masalah yang Dihadapi
Setelah mengetahui bahwa aliran gerak TV-L1 adalah input terbaik, pertanyaan selanjutnya adalah: arsitektur jaringan saraf mana yang paling pintar membaca pola perubahan gerak ruang dan waktu ini?

### Apa yang Dicoba dan Hasilnya
1. **R(2+1)D-18 dengan Pemisahan Ruang dan Waktu (`010` — UF1 0,6951, UAR 0,7160):**
   Arsitektur ini memisahkan perhitungan menjadi konvolusi gambar 2D lalu disambung konvolusi waktu 1D. Dibantu dengan variasi pergeseran uji (*Test-Time Augmentation*), hasilnya cukup baik (UF1 0,6951). Namun karena perhitungannya dipisah, model kurang leluasa melihat hubungan langsung antara posisi otot dan waktu geraknya.
2. **MC3-18 dengan Konvolusi Campuran (`020` — UF1 0,6990, Akurasi 67,48%):**
   Arsitektur ini memakai konvolusi 3D di lapisan-lapisan awal, lalu 2D di lapisan belakang. Performanya cukup kuat (UF1 0,6990).
3. **Kombinasi CNN + RNN / GRU (`037` — ResNet-18 + GRU, UF1 0,6864, Akurasi 65,04%):**
   Banyak orang biasa memakai kombinasi CNN untuk membaca gambar dan RNN (seperti GRU atau LSTM) untuk membaca urutan waktu. Ternyata cara ini kurang memuaskan untuk ekspresi mikro (UF1 hanya 0,6864). RNN kesulitan menangkap perubahan gerakan mikro yang sangat singkat dan cepat jika dibanding filter konvolusi 3D langsung.
4. **R3D-18 dengan Konvolusi 3D Penuh (`017` — UF1 0,7145, Akurasi 68,70%, AUC 0,8948):**
   Konvolusi 3D murni (R3D-18) yang membaca ruang dan waktu secara serentak keluar sebagai **pemenang mutlak**. Model ini mencetak rekor baseline terbaik 0,7145 UF1 dan akurasi 68,70% pada seluruh 26 orang data uji.

> [!TIP]
> **Poin Penting untuk Paper:** Konvolusi 3D murni (R3D-18) adalah arsitektur paling kokoh untuk ekspresi mikro karena membaca hubungan ruang dan waktu secara langsung."""
    },
    {
        'title': "4. Babak III: Menghadapi Masalah Titik Puncak Gerakan (Apex Dilemma)",
        'body': """### Masalah yang Dihadapi
Di makalah-makalah ilmiah, banyak peneliti mengandalkan label titik puncak ekspresi (*apex*) yang sudah ditandai manual oleh manusia. Masalahnya: **pada aplikasi dunia nyata, tidak ada manusia yang menandai frame puncak tersebut**. Jika sistem ingin dipakai otomatis pada rekaman video pengguna, sistem harus bisa menemukan puncak gerakan itu sendiri tanpa bantuan label.

### Apa yang Dicoba dan Hasilnya
1. **Model Contekan Manusia / Oracle (`068` — UF1 0,7104, Akurasi 70,83%):**
   Jika model menggabungkan video lengkap dengan potongan frame puncak yang ditandai manusia, skornya mencapai 0,7104 UF1. Ini adalah batas performa terbaik jika kita dibantu manusia.
2. **Mendeteksi Puncak Gerakan Secara Otomatis (`091` — UF1 0,6707, AUC 0,9025):**
   Kami membuat rumus sederhana berbasis energi gerak:
   $$\\mathcal{E}(t) = \\text{rata-rata besar gerakan seluruh piksel wajah pada frame } t$$
   Frame yang memiliki energi gerakan paling tinggi otomatis ditetapkan sebagai titik puncak (*apex*), tanpa bantuan manusia sama sekali. Model mandiri ini meraih UF1 0,6707.
3. **Model Juara Siap Pakai / Deployable (`096` — UF1 0,7021, Akurasi 69,79%):**
   Dengan menggabungkan model video utuh dan model puncak otomatis (50:50), model ini berhasil meraih **0,7021 UF1**. Nilai ini berhasil menutup 95% selisih terhadap model contekan manusia (0,7104). Ini membuktikan sistem siap dijalankan otomatis pada video baru secara mandiri.

> [!TIP]
> **Poin Penting untuk Paper:** Pendeteksi puncak gerakan otomatis membuktikan bahwa pengenalan ekspresi mikro dapat bekerja mandiri di aplikasi nyata tanpa membutuhkan bantuan anotasi manusia."""
    },
    {
        'title': "5. Babak IV: Gerakan Kepala vs. Mengarahkan Fokus Area Wajah",
        'body': """### Masalah yang Dihadapi
Saat orang menunjukkan ekspresi mikro, kepalanya terkadang sedikit bergerak. Apakah gerakan kepala ini harus dibuang seluruhnya? Dan apakah membantu model untuk fokus pada area tertentu (mata, hidung, mulut) akan meningkatkan ketepatan tebakan?

### Apa yang Dicoba dan Hasilnya
1. **Dampak Buruk Menghilangkan Gerakan Kepala (`102` — UF1 0,5906, Akurasi 61,98%):**
   Kami menguji algoritma penstabil kepala (ECC) agar posisi kepala benar-benar kaku dan diam seperti patung. Hasilnya justru sangat buruk: **performa anjlok hingga 9 poin UF1 (dari 0,6816 menjadi 0,5906)**. Ternyata, sedikit anggukan atau sentakan kepala halus secara biologis merupakan bagian tak terpisahkan dari ekspresi mikro (misalnya ekspresi jijik atau kaget). Menghapus gerakan kepala rigid justru menghilangkan sinyal emosi alami.
2. **Mengarahkan Fokus Lembut ke Area Wajah Penting (`163` — UF1 0,7283 pada data uji 21 orang):**
   Alih-alih menahan kepala, kami memasang pembobot perhatian (*soft attention*) yang memandu model untuk memperhatikan tiga zona utama: mata/alis, hidung, dan mulut. Hasilnya langsung melompat ke UF1 0,7283.
3. **Pengujian Fokus Area pada Seluruh 26 Orang (`182` — UF1 0,7121, Akurasi 69,51%, AUC 0,8964):**
   Ketika diuji pada seluruh 26 subjek tanpa terkecuali, model fokus area ini mempertahankan performa tinggi (UF1 0,7121 dan akurasi 69,51%).

> [!TIP]
> **Poin Penting untuk Paper:** Jangan mengunci gerakan kepala secara kaku. Biarkan gerakan alami tetap utuh, dan gunakan mekanisme atensi spasial untuk mengarahkan model ke area mata, alis, dan mulut."""
    },
    {
        'title': "6. Babak V: Model Juara — Mengajari Model Gerakan Otot Wajah (Multi-Task Action Units)",
        'body': """### Konsep Rancangan Model Juara
Dalam ilmu psikologi wajah (FACS), ekspresi manusia tidak terjadi secara acak, melainkan digerakkan oleh unit-unit otot tertentu yang disebut **Action Units (AU)**. Misalnya:
- Tersenyum (*happiness*) = otot sudut bibir terangkat (AU12).
- Menahan senyum/sedih (*repression*) = sudut bibir ditarik ke bawah (AU15).
- Menolak/jijik (*disgust*) = hidung mengkerut (AU9) atau bibir atas terangkat (AU10).

Daripada hanya menyuruh model menebak 5 nama emosi, kami merancang model dengan **dua tugas sekaligus (Multi-Task)**:
1. **Kepala Utama:** Menebak 5 kategori emosi (*happiness*, *disgust*, *repression*, *surprise*, *others*).
2. **Kepala Pembantu:** Menebak 11 unit otot wajah yang sedang bergerak aktif (AU1, AU2, AU4, AU5, AU7, AU9, AU10, AU12, AU14, AU15, AU17).

Rumus latihan gabungannya:
$$\\text{Loss Total} = \\text{Loss Emosi} + \\lambda_{\\text{AU}} \\times \\text{Loss Gerakan Otot}$$

### Apa yang Dicoba dan Hasilnya
1. **Mencari Porsi Pengaruh Otot $\\lambda_{\\text{AU}}$ (`156` vs `140`):**
   - Jika bobot bantuan otot terlalu kecil $\\lambda=0,01$ (`156`), model belum merasakan manfaatnya (UF1 0,6682).
   - Saat bobot diatur seimbang $\\lambda=0,5$ (`140`), panduan dari otot membantu fitur model mengenali ekspresi secara jauh lebih rapi.
2. **Rekor Tertinggi Sepanjang Masa (`188` — Pengujian Penuh 26 Orang):**
   Model R3D-18 Multi-Task AU dengan bobot 0,5 mencetak angka terbaik di seluruh proyek:
   - **Akurasi (ACC): 69,92%**
   - **Macro-F1 (UF1): 0,7211**
   - **Recall Seimbang (UAR): 0,7378**
   - **Macro-AUC: 0,8949**
   - **Spesifisitas: 91,70%**

Panduan otot (AU) memaksa otak model untuk memperhatikan perubahan fisik otot wajah manusia yang sebenarnya, sehingga model tidak lagi terkecoh oleh perbedaan rupa atau bentuk wajah orang yang berbeda.

> [!TIP]
> **Poin Penting untuk Paper:** Supervisi ganda bersama unit aksi otot (Facial Action Units) adalah kunci utama keberhasilan mencapai akurasi tertinggi pada pengujian Leave-One-Subject-Out yang ketat."""
    },
    {
        'title': "7. Babak VI: Membongkar Klaim Paper Lain & Analisis Kegagalan Metode Populer",
        'body': """### Masalah yang Dihadapi
Untuk memperkuat argumen di paper, kami menguji apakah klaim akurasi sangat tinggi (di atas 85%) di paper lain benar-benar valid, serta menguji metode populer lain seperti Transformer, Graf Landmark, dan rumus regangan fisik.

### Apa yang Dicoba dan Hasilnya
1. **Membongkar Klaim Akurasi 88% Paper Lain (`200` — Replikasi Non-LOSO vs. Pengujian Jujur):**
   Beberapa publikasi mengklaim akurasi di atas 85–88% pada CASME II. Kami mereplikasi kode mereka: jika diuji dengan pembagian acak biasa memang benar muncul angka 88,33%. Namun ini terjadi karena **wajah orang yang sama ada di data latihan dan data ujian (kebocoran subjek)**. Saat model yang sama diuji secara jujur tanpa kebocoran subjek (LOSO 26), skor aslinya **langsung anjlok ke 53,89% UF1**. Klaim tinggi di literatur tersebut terbukti palsu akibat kebocoran data.
2. **Kegagalan Vision Transformer / ViViT (`201` — Akurasi 28,05%, UF1 0,2093):**
   Arsitektur Transformer seperti ViViT membutuhkan ratusan ribu data video agar bisa belajar dengan baik. Pada dataset kecil CASME II (hanya 246 klip video), Transformer mengalami kelaparan data dan gagal total (*overfitting* parah), dengan akurasi hanya 28,05%.
3. **Kegagalan Graf Titik Wajah / GCN-GRU (`202` — Akurasi 17,89%, UF1 0,1799):**
   Menggunakan titik-titik koordinat wajah (landmark) yang diolah dengan Graph Convolutional Network juga gagal total (UF1 0,1799). Mengapa? Karena titik koordinat terlalu kasar dan **tidak bisa melihat kerutan halus pada kulit wajah** (seperti kerutan tipis di dahi atau lipatan hidung).
4. **Uji Pembanding Rumus Fisika Regangan Kulit (`203` — UF1 0,6938, Akurasi 68,29%):**
   Kami juga menguji rumus mekanika elastisitas (tensor regangan kulit $\\varepsilon_{xx}, \\varepsilon_{yy}$). Hasilnya cukup bagus (UF1 0,6938), tetapi masih berada di bawah Model Juara AU (`188` — UF1 0,7211). Ini membuktikan bahwa memahami biologi otot wajah (AU) memberi petunjuk yang lebih tepat daripada sekadar rumus mekanika fisik benda mati.

> [!TIP]
> **Poin Penting untuk Paper:** Tidak semua model modern (seperti Transformer atau Graf) cocok untuk data ekspresi mikro. Model 3D CNN dengan panduan unit otot biologis terbukti sebagai pendekatan yang paling tepat dan teruji kokoh."""
    }
]

for s in narrative_sections:
    md.append(f"## {s['title']}")
    md.append("")
    md.append(s['body'])
    md.append("")
    md.append("---")
    md.append("")

md.append("## 8. Panduan Praktis Menulis Manuskrip Paper")
md.append("")
md.append("Bagian ini memetakan hasil-hasil eksperimen di atas ke bagian-bagian standar artikel jurnal ilmiah:")
md.append("")
md.append("1. **Pendahuluan (Introduction):**")
md.append("   - Jelaskan mengapa ekspresi mikro sulit dikenali dan tunjukkan bukti bahwa model gambar biasa (`002`) gagal karena menghafal wajah orang.")
md.append("   - Tunjukkan temuan bahwa menggabungkan foto asli dan gerakan (`006`) justru memperburuk akurasi.")
md.append("2. **Tinjauan Pustaka & Kritik Literatur (Related Work):**")
md.append("   - Kutip hasil audit replikasi (`200`) untuk menjelaskan bahwa klaim akurasi di atas 85% di beberapa paper sebelumnya terjadi akibat kebocoran data orang.")
md.append("   - Bahas mengapa metode populer seperti Transformer ViViT (`201`) dan Graf Landmark (`202`) gagal pada data ekspresi mikro yang terbatas.")
md.append("3. **Metodologi (Methodology):**")
md.append("   - Jelaskan rumus aliran optik TV-L1 ber-referensi frame awal netral (`004`).")
md.append("   - Jelaskan arsitektur backbone konvolusi 3D R3D-18 (`017`).")
md.append("   - Rinci rumus pencari titik puncak gerakan otomatis tanpa bantuan manusia (`091` & `096`).")
md.append("   - Rinci perancangan fungsi loss multi-task bersama unit aksi otot wajah / FACS AU (`188`).")
md.append("4. **Hasil Pengujian Utama (Experimental Results):**")
md.append("   - Tampilkan skor pengujian jujur pada 26 orang penuh: Akurasi (69,92%), Macro-F1 (0,7211), Balanced Recall (0,7378), dan Macro-AUC (0,8949).")
md.append("   - Tampilkan kurva ROC dan kurva pembelajaran per-epoch.")
md.append("5. **Tabel Studi Perbandingan & Ablasi (Ablation Studies):**")
md.append("   - **Tabel 1 (Bentuk Input):** bandingkan `002`, `003`, `004`, `006`, dan `045`.")
md.append("   - **Tabel 2 (Arsitektur Model):** bandingkan `010`, `020`, `037`, dan `017`.")
md.append("   - **Tabel 3 (Deteksi Puncak Apex & Kesiapan Aplikasi):** bandingkan `068`, `091`, dan `096`.")
md.append("   - **Tabel 4 (Gerakan Kepala & Fokus Area Wajah):** bandingkan `102`, `163`, dan `182`.")
md.append("   - **Tabel 5 (Panduan Otot AU vs. Pembanding Fisika):** bandingkan `017`, `203`, `156`, `140`, dan `188`.")
md.append("")

doc_text = "\n".join(md)
with open('docs/paper_progression_narrative.md', 'w', encoding='utf-8') as f:
    f.write(doc_text)

print(f"Successfully generated docs/paper_progression_narrative.md ({len(doc_text)} bytes)!")
