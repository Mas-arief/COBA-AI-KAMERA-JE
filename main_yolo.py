"""
SMART ATTENDANCE SYSTEM - YOLOv8 + Deep Feature Extraction
Install: pip install ultralytics opencv-python numpy scikit-learn
"""

import cv2
import numpy as np
import os
from datetime import datetime
import pickle
import time
from pathlib import Path

# YOLOv8 untuk deteksi wajah
try:
    from ultralytics import YOLO
    yolo_available = True
    print("✓ YOLOv8 tersedia")
except ImportError:
    print("⚠️  YOLOv8 tidak terinstall. Install dengan: pip install ultralytics")
    yolo_available = False

# ========================================
# KONFIGURASI
# ========================================

FOTO_FOLDER = "foto_mahasiswa"
MODEL_FILE = "face_features.pkl"
YOLO_MODEL = "yolov8n-face.pt"

# Similarity threshold untuk recognition
SIMILARITY_THRESHOLD = 0.6

print("="*60)
print("SMART ATTENDANCE SYSTEM - YOLOv8 + Deep Features")
print("="*60)

# ========================================
# INISIALISASI YOLO
# ========================================

if not yolo_available:
    print("\n❌ YOLOv8 diperlukan!")
    print("Install dengan: pip install ultralytics")
    exit(1)

try:
    print("\nMemuat YOLOv8 Face Detection Model...")
    model_yolo = YOLO(YOLO_MODEL)
    print("✓ YOLOv8 Model loaded")
except Exception as e:
    print(f"⚠️  Error loading YOLOv8: {e}")
    print("Model akan di-download otomatis pada penggunaan pertama...")
    try:
        model_yolo = YOLO(YOLO_MODEL)
    except Exception as e2:
        print(f"❌ Gagal load YOLOv8: {e2}")
        exit(1)

# ========================================
# FUNGSI UTILITY
# ========================================

def improve_image_quality_color(bgr):
    """Improve kualitas image warna"""
    denoised = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

def extract_deep_features(image):
    """Extract deep features dari wajah menggunakan histogram + texture"""
    if image is None or image.size == 0:
        return None
    
    try:
        # Resize ke 160x160
        image = cv2.resize(image, (160, 160))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Feature 1: Histogram (256 bins)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Feature 2: HOG (Histogram of Oriented Gradients)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Quantize angles (8 bins) dan magnitude
        h_bins = 8
        mag_bins = 8
        
        hist_hog = np.zeros((h_bins * mag_bins,))
        for i in range(h_bins):
            for j in range(mag_bins):
                mask = ((ang >= i * np.pi / h_bins) & (ang < (i + 1) * np.pi / h_bins) &
                       (mag >= j * mag.max() / mag_bins) & (mag < (j + 1) * mag.max() / mag_bins))
                hist_hog[i * mag_bins + j] = mag[mask].sum()
        
        hist_hog = cv2.normalize(hist_hog, hist_hog).flatten() if hist_hog.sum() > 0 else hist_hog
        
        # Feature 3: Contour/Shape info
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_features = np.array([len(contours), 
                                  cv2.moments(edges)['m00'] if edges.sum() > 0 else 0,
                                  edges.sum() / (gray.shape[0] * gray.shape[1])])
        shape_features = shape_features / (np.linalg.norm(shape_features) + 1e-6)
        
        # Combine semua features
        combined = np.concatenate([hist[:64], hist_hog[:32], shape_features])
        return combined / (np.linalg.norm(combined) + 1e-6)
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def euclidean_distance(feature1, feature2):
    """Hitung euclidean distance antara dua feature"""
    if feature1 is None or feature2 is None:
        return float('inf')
    return np.linalg.norm(feature1 - feature2)

def detect_faces_yolo(image):
    """Deteksi wajah menggunakan YOLOv8"""
    try:
        results = model_yolo.predict(image, conf=0.5, imgsz=640, verbose=False)
        
        faces = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # Validasi koordinat
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h))
                y2 = max(0, min(y2, h))
                
                if x2 > x1 and y2 > y1:
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
        
        return faces
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return []

# ========================================
# TRAINING DATA
# ========================================

print("\n📂 Memuat data training dari foto_mahasiswa...")

known_features = []
known_names = []
known_nims = []
mahasiswa_data = {}

# Scanning folder foto_mahasiswa
if os.path.exists(FOTO_FOLDER):
    # Collect all unique mahasiswa
    mahasiswa_folders = set()
    for filename in os.listdir(FOTO_FOLDER):
        if filename.endswith('_face.jpg'):
            parts = filename.split('_')
            mahasiswa_name = parts[0]
            mahasiswa_folders.add(mahasiswa_name)
    
    print(f"   Ditemukan {len(mahasiswa_folders)} mahasiswa")
    
    # Cek apakah sudah ada saved features
    if os.path.exists(MODEL_FILE):
        print("📂 Loading saved features...")
        with open(MODEL_FILE, 'rb') as f:
            saved_data = pickle.load(f)
            known_features = saved_data['features']
            known_names = saved_data['names']
            known_nims = saved_data['nims']
        
        print(f"✓ Loaded {len(known_names)} mahasiswa features")
    else:
        print("🔄 Extracting features dari foto training...")
        
        # Default data mahasiswa
        default_data = {
            "mahasiswa1": {"nama": "Mahasiswa 1", "nim": "2025001"},
            "mahasiswa2": {"nama": "Mahasiswa 2", "nim": "2025002"},
            "mahasiswa3": {"nama": "Mahasiswa 3", "nim": "2025003"},
            "mahasiswa4": {"nama": "Mahasiswa 4", "nim": "2025004"},
        }
        
        for mahasiswa_folder in sorted(mahasiswa_folders):
            # Get data
            if mahasiswa_folder in default_data:
                info = default_data[mahasiswa_folder]
            else:
                info = {
                    "nama": mahasiswa_folder.replace("mahasiswa", "Mahasiswa "),
                    "nim": f"2025{str(len(mahasiswa_data)+1).zfill(3)}"
                }
            
            mahasiswa_data[mahasiswa_folder] = info
            
            # Collect features dari semua foto _face.jpg
            person_features = []
            valid_count = 0
            
            for filename in sorted(os.listdir(FOTO_FOLDER)):
                if filename.startswith(mahasiswa_folder) and filename.endswith('_face.jpg'):
                    filepath = os.path.join(FOTO_FOLDER, filename)
                    try:
                        image = cv2.imread(filepath)
                        if image is not None:
                            features = extract_deep_features(image)
                            if features is not None:
                                person_features.append(features)
                                valid_count += 1
                    except Exception as e:
                        pass
            
            if len(person_features) > 0:
                # Average features
                avg_features = np.mean(person_features, axis=0)
                known_features.append(avg_features)
                known_names.append(info['nama'])
                known_nims.append(info['nim'])
                print(f"✓ {info['nama']} ({info['nim']}) - {valid_count} features extracted")
            else:
                print(f"✗ Tidak ada foto untuk: {mahasiswa_folder}")
        
        # Simpan features
        if len(known_features) > 0:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump({
                    'features': known_features,
                    'names': known_names,
                    'nims': known_nims
                }, f)
            
            print(f"\n💾 Features tersimpan: {MODEL_FILE}")
else:
    print(f"❌ Folder {FOTO_FOLDER} tidak ditemukan!")
    exit(1)

if len(known_names) == 0:
    print("\n❌ TIDAK ADA DATA TRAINING!")
    print("Jalankan ambil_foto.py terlebih dahulu untuk collect foto training")
    exit(1)

print(f"\n✅ Total {len(known_names)} mahasiswa terdaftar\n")

# ========================================
# REAL-TIME DETECTION & ATTENDANCE
# ========================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi!")
    exit()

print("🎥 Kamera aktif!")
print("📌 Tekan 's' untuk simpan absensi")
print("📌 Tekan 'q' untuk keluar\n")

# Track absensi
attendance_log = {}
for i, name in enumerate(known_names):
    attendance_log[name] = {'nim': known_nims[i], 'count': 0, 'last_seen': None}

detection_cooldown = {}
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_enhanced = improve_image_quality_color(frame)
    
    # Detect faces dengan YOLOv8
    faces = detect_faces_yolo(frame)
    
    # Process setiap wajah
    for face_info in faces:
        x1, y1, x2, y2 = face_info['bbox']
        conf = face_info['confidence']
        
        # Extract region
        face_region = frame[y1:y2, x1:x2]
        
        # Extract features
        features = extract_deep_features(face_region)
        
        if features is not None:
            # Compare dengan known features
            distances = []
            for known_feat in known_features:
                dist = euclidean_distance(features, known_feat)
                distances.append(dist)
            
            min_dist = min(distances)
            min_idx = distances.index(min_dist)
            
            # Tentukan apakah match
            if min_dist < SIMILARITY_THRESHOLD:
                name = known_names[min_idx]
                nim = known_nims[min_idx]
                label = f"{name} ({nim})"
                color = (0, 255, 0)  # Green
                confidence_text = f"Conf: {(1 - min_dist/SIMILARITY_THRESHOLD)*100:.1f}%"
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red
                confidence_text = f"Dist: {min_dist:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Put label
            cv2.putText(frame, label, (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, confidence_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Track attendance (dengan cooldown 3 detik)
            if min_dist < SIMILARITY_THRESHOLD:
                if name not in detection_cooldown or (time.time() - detection_cooldown[name]) > 3:
                    if attendance_log[name]['count'] == 0:
                        attendance_log[name]['count'] = 1
                        attendance_log[name]['last_seen'] = datetime.now().strftime("%H:%M:%S")
                    detection_cooldown[name] = time.time()
    
    # Display info
    hadir = sum(1 for v in attendance_log.values() if v['count'] > 0)
    cv2.putText(frame, f"Hadir: {hadir}/{len(known_names)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, "Tekan 's' untuk simpan | 'q' untuk keluar", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    cv2.imshow('Smart Attendance - YOLOv8', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # 's' untuk simpan absensi
    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"absensi_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"LAPORAN ABSENSI - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            hadir_count = 0
            f.write("HADIR:\n")
            for name, data in attendance_log.items():
                if data['count'] > 0:
                    f.write(f"  ✓ {data['nim']} - {name} (Jam: {data['last_seen']})\n")
                    hadir_count += 1
            
            f.write(f"\nTIDAK HADIR:\n")
            for name, data in attendance_log.items():
                if data['count'] == 0:
                    f.write(f"  ✗ {data['nim']} - {name}\n")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"Total Hadir: {hadir_count}/{len(known_names)}\n")
            f.write(f"{'='*60}\n")
        
        print(f"\n✅ Absensi tersimpan: {filename}")
    
    # 'q' untuk keluar
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✅ Program selesai")
