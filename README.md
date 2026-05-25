# EcoScan Streamlit - Python 3.14.x

Versi ini dibuat untuk Python 3.14.x tanpa TensorFlow. Model `.keras` dimuat menggunakan Keras 3 dengan backend OpenVINO untuk inference.

## Cara menjalankan di PowerShell

```powershell
cd C:\klasifikasi_sampah\ecoscan_streamlit_py314
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Catatan penting

- Jangan install `tensorflow` pada Python 3.14 karena belum tersedia wheel resmi yang stabil.
- Backend harus OpenVINO. Di `app.py`, backend sudah diset otomatis sebelum import Keras.
- Preprocessing mengikuti notebook: RGB, resize 224x224, dan rescale 1/255.
- Mapping kelas: O = Organik = 0, R = Anorganik = 1.

Jika model gagal dimuat, kemungkinan file `.keras` berisi komponen yang bergantung langsung pada TensorFlow. Solusi stabilnya adalah memakai Python 3.11/3.12 untuk inference atau konversi model ke OpenVINO IR.
