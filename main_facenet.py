"""
SMART ATTENDANCE - Haarcascade + FaceNet
Uses Haarcascade for face detection and keras-facenet for embeddings.
Run: python main_facenet.py
"""

import os
import cv2
import numpy as np
import pickle
import time
from datetime import datetime

# try import FaceNet
try:
    from keras_facenet import FaceNet
    facenet_ok = True
except Exception as e:
    print('keras-facenet not available:', e)
    facenet_ok = False

FOTO_FOLDER = 'foto_mahasiswa'
MODEL_FILE = 'facenet_embeddings.pkl'
SIMILARITY_THRESHOLD = 0.9  # lower is stricter for euclidean, here we use cosine

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not facenet_ok:
    print('\nFaceNet tidak tersedia. Pastikan `keras-facenet` terinstall.')
    exit(1)

# init facenet
facenet = FaceNet()

# load face crops and build embeddings if no saved model
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE,'rb') as f:
        data = pickle.load(f)
    known_embeddings = data['embeddings']
    known_names = data['names']
    print(f"Loaded {len(known_names)} known identities from {MODEL_FILE}")
else:
    # build from *_face.jpg in FOTO_FOLDER
    known_embeddings = []
    known_names = []
    if not os.path.exists(FOTO_FOLDER):
        print(f"Folder {FOTO_FOLDER} not found")
        exit(1)
    persons = {}
    for fn in os.listdir(FOTO_FOLDER):
        if fn.endswith('_face.jpg'):
            key = fn.split('_')[0]
            persons.setdefault(key, []).append(os.path.join(FOTO_FOLDER, fn))
    if len(persons) == 0:
        print('No face crops found. Run ambil_foto.py first.')
        exit(1)
    for person, files in sorted(persons.items()):
        embs = []
        for p in files:
            img = cv2.imread(p)
            if img is None: continue
            img = cv2.resize(img, (160,160))
            img = img.astype('float32') / 255.0
            e = facenet.embeddings([img])
            if len(e)>0:
                embs.append(e[0])
        if len(embs)>0:
            # store list of embeddings per person
            for e in embs:
                known_embeddings.append(e)
                known_names.append(person)
            print(f"Built embeddings for {person}: {len(embs)} samples")
    # save
    with open(MODEL_FILE,'wb') as f:
        pickle.dump({'embeddings': known_embeddings, 'names': known_names}, f)
    print(f"Saved embeddings to {MODEL_FILE}")

# convert to arrays
known_embeddings = np.vstack(known_embeddings)
known_names = np.array(known_names)

# start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Kamera tidak tersedia')
    exit(1)

attendance = {}
unique_names = np.unique(known_names)
for n in unique_names:
    attendance[n] = {'count':0, 'last_seen':None}

print("Kamera aktif. Tekan 'q' untuk keluar, 's' untuk simpan absensi")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80,80))
    for (x,y,w,h) in faces:
        x1,x2,y1,y2 = x, x+w, y, y+h
        crop = frame[y1:y2, x1:x2]
        if crop.size==0: continue
        crop_resized = cv2.resize(crop, (160,160)).astype('float32')/255.0
        emb = facenet.embeddings([crop_resized])[0]
        # cosine similarity against known embeddings
        sims = known_embeddings.dot(emb) / (np.linalg.norm(known_embeddings, axis=1) * (np.linalg.norm(emb)+1e-10))
        best_idx = np.argmax(sims)
        best_sim = float(sims[best_idx])
        label = 'Unknown'
        if best_sim >= 0.6:
            name = known_names[best_idx]
            label = f"{name} {best_sim:.2f}"
            attendance[name]['count'] += 1
            attendance[name]['last_seen'] = datetime.now().strftime('%H:%M:%S')
        # draw
        color = (0,255,0) if label!='Unknown' else (0,0,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,label,(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
    cv2.putText(frame, f"Hadir: {sum(1 for v in attendance.values() if v['count']>0)}/{len(attendance)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
    cv2.imshow('FaceNet Attendance', frame)
    k = cv2.waitKey(1) & 0xFF
    if k==ord('s'):
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fn = f'absensi_{ts}.txt'
        with open(fn,'w',encoding='utf-8') as f:
            f.write('ABSENSI\n')
            for n,d in attendance.items():
                if d['count']>0:
                    f.write(f"{n}: seen {d['count']} times at {d['last_seen']}\n")
        print('Saved', fn)
    if k==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Selesai')
