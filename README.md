# Workflow CI Sri Ulan Muharromi

Repository ini digunakan untuk memenuhi Kriteria 3 (Workflow CI).

Komponen utama:

* MLflow Project
* GitHub Actions Workflow
* Training model otomatis saat terjadi push ke branch main
* Penyimpanan artefak hasil training pada GitHub Actions

Struktur:

* MLProject/
* .github/workflows/train.yml

Workflow akan menjalankan proses training secara otomatis setiap kali trigger terpantik.
