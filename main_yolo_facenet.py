"""
SMART ATTENDANCE SYSTEM - YOLOv8 + FaceNet
Install: pip install ultralytics opencv-python numpy tensorflow keras-facenet pillow scikit-learn
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
except ImportError:
    print("⚠️  YOLOv8 tidak terinstall. Gunakan: pip install ultralytics")
    yolo_available = False

# FaceNet untuk embedding
try:
    from keras_facenet import FacenetModel
    facenet_available = True
except ImportError:
    print("⚠️  keras-facenet tidak terinstall. Gunakan: pip install keras-facenet")
    facenet_available = False

# ========================================
# KONFIGURASI
# ========================================

FOTO_FOLDER = "foto_mahasiswa"
MODEL_FILE = "face_embeddings.pkl"
DATA_MAHASISWA_FILE = "mahasiswa_data.pkl"

# Similarity threshold untuk recognition
SIMILARITY_THRESHOLD = 0.6

print("="*60)
print("SMART ATTENDANCE SYSTEM - YOLOv8 + FaceNet")
print("="*60)

# ========================================
# INISIALISASI MODELS
# ========================================

# Load YOLOv8 face detection model
if yolo_available:
    try:
        model_yolo = YOLO('yolov8n-face.pt')  # nano face detection model
        print("✓ YOLOv8 Face Detection Model loaded")
    except Exception as e:
        print(f"⚠️  YOLOv8 model load error: {e}")
        print("  Downloading model otomatis...")
        try:
            model_yolo = YOLO('yolov8n-face.pt')
        except:
            yolo_available = False
            print("❌ Gagal load YOLOv8")

# Load FaceNet model
if facenet_available:
    try:
        facenet = FacenetModel()
        print("✓ FaceNet Embedding Model loaded")
    except Exception as e:
        print(f"⚠️  FaceNet model load error: {e}")
        facenet_available = False

if not yolo_available or not facenet_available:
    print("\n❌ Models tidak tersedia. Install dependencies terlebih dahulu:")
    print("  pip install ultralytics keras-facenet tensorflow opencv-python")
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

def extract_face_embedding(image, face_region):
    """Extract FaceNet embedding dari region wajah"""
    try:
        x1, y1, x2, y2 = face_region
        face_crop = image[y1:y2, x1:x2]
        
        # Resize ke 160x160 (standar FaceNet)
        face_resized = cv2.resize(face_crop, (160, 160))
        
        # Normalize
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Extract embedding
        embedding = facenet.embeddings([face_normalized])
        return embedding[0] if len(embedding) > 0 else None
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

def euclidean_distance(embedding1, embedding2):
    """Hitung euclidean distance antara dua embedding"""
    if embedding1 is None or embedding2 is None:
        return float('inf')
    return np.linalg.norm(embedding1 - embedding2)

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

known_embeddings = []
known_names = []
known_nims = []
mahasiswa_data = {}

# Scanning folder foto_mahasiswa
if os.path.exists(FOTO_FOLDER):
    # Collect all unique mahasiswa
    mahasiswa_folders = set()
    for filename in os.listdir(FOTO_FOLDER):
        if filename.endswith('_face.jpg'):
            # Extract mahasiswa name (e.g., "mahasiswa1" dari "mahasiswa1_001_face.jpg")
            parts = filename.split('_')
            mahasiswa_name = parts[0]  # e.g., "mahasiswa1"
            mahasiswa_folders.add(mahasiswa_name)
    
    print(f"   Found {len(mahasiswa_folders)} mahasiswa folders")
    
    # Cek apakah sudah ada saved embeddings
    if os.path.exists(MODEL_FILE) and os.path.exists(DATA_MAHASISWA_FILE):
        print("📂 Loading saved embeddings...")
        with open(MODEL_FILE, 'rb') as f:
            saved_data = pickle.load(f)
            known_embeddings = saved_data['embeddings']
            known_names = saved_data['names']
            known_nims = saved_data['nims']
        
        with open(DATA_MAHASISWA_FILE, 'rb') as f:
            mahasiswa_data = pickle.load(f)
        
        print(f"✓ Loaded {len(known_names)} mahasiswa embeddings")
    else:
        print("🔄 Extracting FaceNet embeddings dari foto training...")
        
        # Manual data mahasiswa - bisa disimpan dari UI nanti
        default_data = {
            "mahasiswa1": {"nama": "Mahasiswa 1", "nim": "2025001"},
            "mahasiswa2": {"nama": "Mahasiswa 2", "nim": "2025002"},
            "mahasiswa3": {"nama": "Mahasiswa 3", "nim": "2025003"},
            "mahasiswa4": {"nama": "Mahasiswa 4", "nim": "2025004"},
        }
        
        for mahasiswa_folder in sorted(mahasiswa_folders):
            # Get data (default jika tidak ada)
            if mahasiswa_folder in default_data:
                info = default_data[mahasiswa_folder]
            else:
                # Auto-generate nama dari folder
                info = {
                    "nama": mahasiswa_folder.replace("mahasiswa", "Mahasiswa "),
                    "nim": f"2025{str(len(mahasiswa_data)+1).zfill(3)}"
                }
            
            mahasiswa_data[mahasiswa_folder] = info
            
            # Collect embeddings dari semua foto _face.jpg
            person_embeddings = []
            valid_count = 0
            
            for filename in os.listdir(FOTO_FOLDER):
                if filename.startswith(mahasiswa_folder) and filename.endswith('_face.jpg'):
                    filepath = os.path.join(FOTO_FOLDER, filename)
                    try:
                        image = cv2.imread(filepath)
                        if image is not None:
                            # Resize ke 160x160
                            image_resized = cv2.resize(image, (160, 160))
                            image_normalized = image_resized.astype('float32') / 255.0
                            
                            # Extract embedding
                            embedding = facenet.embeddings([image_normalized])
                            if len(embedding) > 0:
                                person_embeddings.append(embedding[0])
                                valid_count += 1
                    except Exception as e:
                        print(f"   Error loading {filename}: {e}")
            
            if len(person_embeddings) > 0:
                # Average embedding dari semua foto
                avg_embedding = np.mean(person_embeddings, axis=0)
                known_embeddings.append(avg_embedding)
                known_names.append(info['nama'])
                known_nims.append(info['nim'])
                print(f"✓ {info['nama']} ({info['nim']}) - {valid_count} embeddings")
            else:
                print(f"✗ Tidak ada foto untuk: {mahasiswa_folder}")
        
        # Simpan embeddings
        if len(known_embeddings) > 0:
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump({
                    'embeddings': known_embeddings,
                    'names': known_names,
                    'nims': known_nims
                }, f)
            
            with open(DATA_MAHASISWA_FILE, 'wb') as f:
                pickle.dump(mahasiswa_data, f)
            
            print(f"\n💾 Embeddings tersimpan: {MODEL_FILE}")
else:
    print(f"❌ Folder {FOTO_FOLDER} tidak ditemukan!")
    exit(1)

if len(known_names) == 0:
    print("\n❌ TIDAK ADA DATA TRAINING!")
    exit(1)

print(f"\n✅ Total {len(known_names)} mahasiswa terdaftar\n")

# ========================================
# REAL-TIME DETECTION & ATTENDANCE
# ========================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi!")
    exit()

print("🎥 Kamera aktif!")
print("📌 Tekan 's' untuk simpan absensi")
print("📌 Tekan 'q' untuk keluar\n")

# Track absensi
attendance_log = {}
for name in known_names:
    attendance_log[name] = {'nim': '', 'count': 0, 'last_seen': None}

# Simpan NIM
for i, name in enumerate(known_names):
    attendance_log[name]['nim'] = known_nims[i]

frame_count = 0
detection_cooldown = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Improve image quality
    frame_enhanced = improve_image_quality_color(frame)
    
    # Detect faces dengan YOLOv8
    faces = detect_faces_yolo(frame)
    
    # Process setiap wajah yang terdeteksi
    for face_info in faces:
        x1, y1, x2, y2 = face_info['bbox']
        conf = face_info['confidence']
        
        # Extract embedding
        embedding = extract_face_embedding(frame, (x1, y1, x2, y2))
        
        if embedding is not None:
            # Compare dengan known embeddings
            distances = []
            for known_emb in known_embeddings:
                dist = euclidean_distance(embedding, known_emb)
                distances.append(dist)
            
            min_dist = min(distances)
            min_idx = distances.index(min_dist)
            
            # Tentukan apakah match
            if min_dist < SIMILARITY_THRESHOLD:
                name = known_names[min_idx]
                nim = known_nims[min_idx]
                label = f"{name} ({nim})"
                color = (0, 255, 0)  # Green
                confidence_text = f"Match: {(1 - min_dist/SIMILARITY_THRESHOLD)*100:.1f}%"
            else:
                label = f"Unknown"
                color = (0, 0, 255)  # Red
                confidence_text = f"Dist: {min_dist:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Put label
            cv2.putText(frame, label, (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, confidence_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Track attendance (dengan cooldown 5 detik)
            if min_dist < SIMILARITY_THRESHOLD:
                if name not in detection_cooldown or (time.time() - detection_cooldown[name]) > 5:
                    if attendance_log[name]['count'] == 0:
                        attendance_log[name]['count'] = 1
                        attendance_log[name]['last_seen'] = datetime.now().strftime("%H:%M:%S")
                    detection_cooldown[name] = time.time()
    
    # Display info
    cv2.putText(frame, f"Hadir: {sum(1 for v in attendance_log.values() if v['count'] > 0)}/{len(known_names)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, "Tekan 's' untuk simpan | 'q' untuk keluar", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    cv2.imshow('Smart Attendance - YOLOv8 + FaceNet', frame)
    
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
