"""
SMART ATTENDANCE SYSTEM - YOLOv8/Cascade + PCA-based embeddings
Dependencies: opencv-python, numpy. (optional: ultralytics for YOLOv8)
This script builds lightweight embeddings from face crops using PCA (numpy SVD)
and performs recognition by cosine similarity. Works without TensorFlow.
"""

import os
import cv2
import numpy as np
import pickle
import time
from datetime import datetime

FOTO_FOLDER = "foto_mahasiswa"
MODEL_FILE = "simple_embeddings.pkl"
IMG_SIZE = (160, 160)
N_COMPONENTS = 128
SIMILARITY_THRESHOLD = 0.7  # cosine similarity threshold

print("="*60)
print("SMART ATTENDANCE - Simple PCA Embeddings")
print("="*60)

# Try to use YOLOv8 if available
try:
    from ultralytics import YOLO
    yolo_available = True
except Exception:
    yolo_available = False

# Haarcascade fallback
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image):
    if yolo_available:
        try:
            results = model_yolo.predict(image, conf=0.45, imgsz=640, verbose=False)
            faces = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    faces.append((x1, y1, x2, y2))
            return faces
        except Exception:
            pass
    # fallback to cascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dets = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    faces = []
    for (x, y, w, h) in dets:
        faces.append((x, y, x+w, y+h))
    return faces


def load_face_crops():
    """Load all *_face.jpg from FOTO_FOLDER; return dict mapping id->list of image arrays"""
    people = {}
    if not os.path.exists(FOTO_FOLDER):
        print(f"Folder {FOTO_FOLDER} tidak ada")
        return people

    for fn in sorted(os.listdir(FOTO_FOLDER)):
        if fn.lower().endswith('_face.jpg'):
            key = fn.split('_')[0]  # mahasiswa1
            path = os.path.join(FOTO_FOLDER, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            people.setdefault(key, []).append(img)
    return people


def build_embeddings(people_dict, n_components=N_COMPONENTS):
    """Build PCA components from face images and produce embeddings per person"""
    X = []
    labels = []
    for person, images in people_dict.items():
        for img in images:
            img_resized = cv2.resize(img, IMG_SIZE)
            arr = img_resized.astype('float32').flatten() / 255.0
            X.append(arr)
            labels.append(person)
    if len(X) == 0:
        return None
    X = np.vstack(X)  # (n_samples, D)
    n_samples, D = X.shape
    k = min(n_components, n_samples)
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]  # (k, D)
    embeddings = Xc.dot(components.T)  # (n_samples, k)
    # normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    embeddings = embeddings / norms
    # store list of embeddings per person (more robust than single average)
    person_embeddings = {}
    person_embeddings_list = {}
    idx = 0
    for person in people_dict:
        imgs = people_dict[person]
        m = len(imgs)
        if m == 0:
            continue
        person_embs = embeddings[idx:idx+m]
        # normalize and store each embedding
        norms = np.linalg.norm(person_embs, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        person_embs_norm = person_embs / norms
        # average for quick lookup (optional)
        avg = np.mean(person_embs_norm, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-10)
        person_embeddings[person] = avg
        person_embeddings_list[person] = person_embs_norm
        idx += m
    model = {
        'mean': mean,
        'components': components,
        'person_embeddings': person_embeddings,
        'person_embeddings_list': person_embeddings_list
    }
    return model


def project_image(img, model):
    img_resized = cv2.resize(img, IMG_SIZE)
    vec = img_resized.astype('float32').flatten() / 255.0
    vecc = vec - model['mean']
    emb = vecc.dot(model['components'].T)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb


# Load or build model
people = load_face_crops()
if len(people) == 0:
    print("Tidak ditemukan face crops di foto_mahasiswa. Jalankan `ambil_foto.py` dahulu atau letakkan *_face.jpg di folder.")
    # still allow runtime but will only detect unknown
    model = None
else:
    print(f"Ditemukan {len(people)} orang dengan face crops. Membangun embeddings...")
    model = build_embeddings(people)
    if model is not None:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        print(f"Embeddings sederhana tersimpan ke {MODEL_FILE}")

# Try load YOLO model if available
if yolo_available:
    try:
        model_yolo = YOLO('yolov8n-face.pt')
        print("YOLOv8 model siap digunakan")
    except Exception:
        yolo_available = False

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera tidak tersedia")
    exit(1)

print("🎥 Kamera aktif. Tekan 'q' untuk keluar, 's' untuk simpan absensi")

attendance = {}
if model is not None:
    for p in model['person_embeddings']:
        attendance[p] = {'count':0, 'last_seen':None}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    faces = detect_faces(frame)
    for (x1,y1,x2,y2) in faces:
        # ensure within bounds
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        # draw
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        label = 'Unknown'
        if model is not None:
            emb = project_image(face, model)
            # compare to person embeddings lists via cosine (take max per person)
            best = None
            best_sim = -1
            for person, pen in model['person_embeddings_list'].items():
                # pen is array (n_samples, dim)
                sims = pen.dot(emb)
                max_sim = float(np.max(sims))
                if max_sim > best_sim:
                    best_sim = max_sim; best = person
            if best_sim >= SIMILARITY_THRESHOLD:
                label = f"{best} ({best_sim:.2f})"
                # attendance
                if best in attendance:
                    attendance[best]['count'] += 1
                    attendance[best]['last_seen'] = datetime.now().strftime('%H:%M:%S')
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.putText(frame, f"Hadir: {sum(1 for v in attendance.values() if v['count']>0)}/{len(attendance)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255),2)
    cv2.imshow('Simple Attendance', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fn = f"absensi_{timestamp}.txt"
        with open(fn, 'w', encoding='utf-8') as f:
            f.write('ABSENSI\n')
            for p,data in attendance.items():
                if data['count']>0:
                    f.write(f"{p}: seen {data['count']} times at {data['last_seen']}\n")
        print('Absensi tersimpan:', fn)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Selesai')
