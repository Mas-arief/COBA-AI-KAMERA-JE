"""
SMART ATTENDANCE - HAARCASCADE (NO DEPENDENCIES)
Install: pip install opencv-python numpy
"""

import cv2
import numpy as np
import os
from datetime import datetime
import pickle

# ========================================
# KONFIGURASI
# ========================================

FOTO_FOLDER = "foto_mahasiswa"
MODEL_FILE = "face_data.pkl"

# Load haarcascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Data mahasiswa
data_mahasiswa = {
    "mahasiswa1.jpg": {"nama": "Budi Santoso", "nim": "2021001"},
    "mahasiswa2.jpg": {"nama": "Siti Aminah", "nim": "2021002"},
    "mahasiswa3.jpg": {"nama": "Ahmad Rizki", "nim": "2021003"},
    "mahasiswa4.jpg": {"nama": "Dewi Lestari", "nim": "2021004"}
}

print("="*50)
print("SMART ATTENDANCE SYSTEM (HaarCascade)")
print("="*50)

# ========================================
# FUNGSI EKSTRAK FITUR WAJAH
# ========================================

def extract_face_histogram(image, face):
    """Ekstrak histogram wajah sebagai fitur"""
    x, y, w, h = face
    face_roi = image[y:y+h, x:x+w]
    
    # Resize untuk konsistensi
    face_roi = cv2.resize(face_roi, (100, 100))
    
    # Hitung histogram dari grayscale image
    hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

def compare_histograms(hist1, hist2):
    """Bandingkan dua histogram"""
    if hist1 is None or hist2 is None:
        return 0
    
    # Gunakan Bhattacharyya Distance
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

# ========================================
# LOAD ATAU TRAINING DATA
# ========================================

print("\nMemuat data wajah mahasiswa...")

known_histograms = []
known_names = []
known_nims = []

# Cek apakah sudah ada model tersimpan
if os.path.exists(MODEL_FILE):
    print("📂 Loading data dari file...")
    with open(MODEL_FILE, 'rb') as f:
        saved_data = pickle.load(f)
        known_histograms = saved_data['histograms']
        known_names = saved_data['names']
        known_nims = saved_data['nims']
    print(f"✓ Loaded {len(known_names)} mahasiswa")
else:
    print("🔄 Training model dari foto...")
    
    for filename, info in data_mahasiswa.items():
        filepath = os.path.join(FOTO_FOLDER, filename)
        
        if os.path.exists(filepath):
            # Load image
            image = cv2.imread(filepath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Deteksi wajah
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Ambil wajah pertama
                face = faces[0]
                hist = extract_face_histogram(gray, face)
                
                known_histograms.append(hist)
                known_names.append(info["nama"])
                known_nims.append(info["nim"])
                print(f"✓ {info['nama']} - {filename}")
            else:
                print(f"✗ Tidak ada wajah di: {filename}")
        else:
            print(f"✗ File tidak ada: {filepath}")
    
    # Simpan model
    if len(known_names) > 0:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({
                'histograms': known_histograms,
                'names': known_names,
                'nims': known_nims
            }, f)
        print(f"\n💾 Model tersimpan di: {MODEL_FILE}")

if len(known_names) == 0:
    print("\n❌ TIDAK ADA DATA MAHASISWA!")
    exit()

print(f"\n✅ Total {len(known_names)} mahasiswa terdaftar")
print("="*50)

# ========================================
# DETEKSI REAL-TIME
# ========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi!")
    exit()

absensi_hari_ini = {}

print("\n🎥 Kamera aktif!")
print("📌 Tekan 'q' untuk keluar")
print("📌 Tekan 's' untuk simpan absensi")
print("📌 Tekan 'r' untuk reset training\n")

frame_counter = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Gagal mengambil frame")
        break
    
    frame_counter += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah (setiap 5 frame untuk performa)
    if frame_counter % 5 == 0:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Ekstrak histogram
            face_roi = gray[y:y+h, x:x+w]
            current_hist = extract_face_histogram(frame, (x, y, w, h))
            
            # Bandingkan dengan database
            best_match = None
            best_distance = float('inf')
            
            for i, known_hist in enumerate(known_histograms):
                distance = compare_histograms(current_hist, known_hist)
                if distance < best_distance:
                    best_distance = distance
                    best_match = i
            
            # Threshold untuk match
            if best_distance < 0.3:  # Semakin kecil, semakin mirip
                name = known_names[best_match]
                nim = known_nims[best_match]
                confidence = (1 - best_distance) * 100
                
                # Catat absensi
                if nim not in absensi_hari_ini:
                    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    absensi_hari_ini[nim] = {
                        "nama": name,
                        "waktu": waktu
                    }
                    print(f"✓ HADIR: {name} ({nim}) - {confidence:.1f}%")
                
                # Draw box hijau
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                # Draw box merah untuk unknown
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, "Unknown", (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Info di layar
    cv2.putText(frame, f"Hadir: {len(absensi_hari_ini)}/{len(known_names)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(frame, "q=keluar | s=simpan | r=reset", 
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1)
    
    cv2.imshow('Smart Attendance', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        print("\n💾 Menyimpan...")
        break
    elif key == ord('r'):
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
            print("\n🔄 Model direset! Restart program.")
            break

# ========================================
# SIMPAN HASIL
# ========================================

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("REKAPITULASI ABSENSI")
print("="*50)

if len(absensi_hari_ini) == 0:
    print("Tidak ada yang hadir")
else:
    for nim, data in sorted(absensi_hari_ini.items()):
        print(f"✓ {data['nama']} ({nim}) - {data['waktu']}")

print("="*50)

# Simpan ke file
filename = f"absensi_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
with open(filename, "w", encoding="utf-8") as f:
    f.write("="*50 + "\n")
    f.write("DAFTAR HADIR MAHASISWA\n")
    f.write("="*50 + "\n")
    f.write(f"Tanggal: {datetime.now().strftime('%d %B %Y')}\n\n")
    
    f.write("HADIR:\n")
    for nim, data in sorted(absensi_hari_ini.items()):
        f.write(f"  ✓ {data['nama']} ({nim}) - {data['waktu']}\n")
    
    f.write(f"\nTotal: {len(absensi_hari_ini)}/{len(known_names)}\n")

print(f"\n✅ Tersimpan: {filename}")
