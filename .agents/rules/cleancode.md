# Antigravity Agent Clean Code & Maintainability Contract

Sebagai agen otonom AI di repositori ini, saya berjanji untuk selalu:

1. **Menjaga Struktur Direktori**: Semua hasil eksekusi eksperimen harus berada pada direktori yang telah disepakati (misalnya `runs/`). Jika ada skrip *legacy* yang menggunakan direktori lain (seperti `experiments/`), saya wajib memodifikasinya agar sesuai dengan struktur `runs/` yang bersih, bukan malah menyebarkan file secara acak.
2. **Clean Code**: Semua skrip baru, terutama arsitektur model dan *data loader*, harus menggunakan penamaan variabel yang deskriptif, komentar yang jelas, dan menghindari duplikasi.
3. **Penyelesaian Akar Masalah (Root Cause)**: Saya tidak akan menggunakan solusi sementara (*hack* atau *workaround*) jika akar masalah bisa diperbaiki (contoh: merapikan path *output* di dalam skrip `run_protocol_v2.py` langsung daripada terus memindahkan direktori secara manual setiap saat).
4. **Active Self-Looping & Monitoring (Wajib)**: Mulai sekarang, saya **DILARANG KERAS** menjalankan metode "fire-and-forget" pada *background task*. Setiap kali sebuah proses diluncurkan, saya wajib menggunakan mekanisme *Event-Driven* atau fitur alarm (`schedule`) untuk memonitor, mengevaluasi hasil secara *real-time*, mendeteksi *error* (seperti gagal _path_ atau logika *fold* yang salah), dan melakukan siklus *self-correction* otomatis sebelum berani menyatakan bahwa saya sedang "menunggu".
5. **Verifikasi Metodologi Silang**: Sebelum mereplikasi metodologi dari *paper* atau repositori eksternal (seperti LOSO), saya wajib memverifikasi secara sadar parameter pengujian (seperti jumlah *fold* atau metrik evaluasi) agar replikasi tersebut 100% *apple-to-apple*.
