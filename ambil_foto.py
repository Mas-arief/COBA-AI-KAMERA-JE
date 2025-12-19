import cv2
import os
import numpy as np

def improve_image_quality_color(bgr):
    # Reduce noise while keeping edges
    denoised = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to LAB and apply CLAHE on L channel for better light/contrast
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Slight sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened

# Pastikan folder ada
if not os.path.exists("foto_mahasiswa"):
    os.makedirs("foto_mahasiswa")
    print("Folder foto_mahasiswa dibuat")

# Buka kamera
cap = cv2.VideoCapture(0)

# Set camera properties untuk kualitas terbaik
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
cap.set(cv2.CAP_PROP_CONTRAST, 50)

if not cap.isOpened():
    print("Kamera tidak terdeteksi!")
    exit()

print("\n" + "="*50)
print("AMBIL FOTO MAHASISWA (100 FOTO PER ORANG)")
print("="*50)
print("Tekan SPASI untuk mulai capture 100 foto")
print("Tekan ESC untuk keluar\n")

person_counter = 1
total_photos = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Gagal membaca frame")
        break
    
    # Flip frame agar tidak mirror
    frame = cv2.flip(frame, 1)

    # Improve kualitas gambar warna
    display_frame = improve_image_quality_color(frame)
    
    # Tampilkan instruksi di layar
    cv2.putText(display_frame, "SPASI = Capture 10 Foto | ESC = Keluar", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Total Foto: {total_photos}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow('Ambil Foto', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # SPASI (32) untuk capture 100 foto
    if key == 32:
        print(f"\nCapturing 100 foto untuk mahasiswa{person_counter}...")
        # setup face detector dengan parameter yang lebih strict
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Counter untuk foto yang valid (dengan deteksi wajah)
        valid_photo_count = 0
        attempt_count = 0
        max_attempts = 300  # Maksimal 300 frame untuk dapat 100 foto valid
        
        print("   Hanya foto dengan deteksi wajah yang jelas akan disimpan...")

        while valid_photo_count < 100 and attempt_count < max_attempts:
            attempt_count += 1
            # Capture foto
            ret, capture_frame = cap.read()
            if not ret:
                print("❌ Gagal membaca frame")
                break
            
            capture_frame = cv2.flip(capture_frame, 1)
            # Improve and keep color for saves
            improved = improve_image_quality_color(capture_frame)

            # Detect face dengan parameter yang lebih strict
            gray_for_face = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_for_face, 
                scaleFactor=1.05,      # lebih strict (default 1.1)
                minNeighbors=7,        # lebih ketat (default 5)
                minSize=(100, 100),    # minimum wajah lebih besar
                maxSize=(400, 400)     # maksimal wajah
            )

            # Skip jika tidak ada wajah terdeteksi
            if len(faces) == 0:
                continue
            
            # Pilih wajah terbesar
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            (x, y, w, h) = faces[0]
            
            # Check jika wajah terlalu dekat dengan edge (face not cutoff)
            margin = 20
            if x - margin < 0 or y - margin < 0 or x + w + margin > capture_frame.shape[1] or y + h + margin > capture_frame.shape[0]:
                continue  # Skip foto dengan wajah di edge
            
            # Valid photo - increment counter
            valid_photo_count += 1
            i = valid_photo_count
            
            # Generate filename dengan format: mahasiswa{nomor}_{no_foto}.jpg
            filename = f"mahasiswa{person_counter}_{i:03d}.jpg"
            filepath = os.path.join("foto_mahasiswa", filename)

            # Crop wajah dengan margin untuk FaceNet (160x160)
            margin = int(0.2 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(capture_frame.shape[1], x + w + margin)
            y2 = min(capture_frame.shape[0], y + h + margin)
            face_crop = capture_frame[y1:y2, x1:x2]
            
            face_saved = False
            try:
                face_resized = cv2.resize(face_crop, (160, 160))
                face_filename = f"mahasiswa{person_counter}_{i:03d}_face.jpg"
                face_path = os.path.join("foto_mahasiswa", face_filename)
                cv2.imwrite(face_path, face_resized)
                face_saved = True
            except Exception:
                face_saved = False

            # Simpan foto berwarna yang telah diperbaiki
            success = cv2.imwrite(filepath, improved)
            
            if success:
                if valid_photo_count % 10 == 0:
                    print(f"   Foto {valid_photo_count}/100 tersimpan: {filename}")
                total_photos += 1

                # Show countdown di display
                overlay = improved.copy()
                cv2.putText(overlay, f"Valid: {valid_photo_count}/100 (attempt: {attempt_count})", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(overlay, "Tetap di posisi yang sama", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow('Ambil Foto', overlay)
                cv2.waitKey(50)
            else:
                valid_photo_count -= 1  # Revert jika gagal simpan
        
        if valid_photo_count >= 100:
            print(f"Selesai capture 100 foto valid untuk mahasiswa{person_counter}")
        else:
            print(f"Hanya dapat {valid_photo_count} foto valid dari {attempt_count} attempt untuk mahasiswa{person_counter}")
        person_counter += 1
        
        print(f"Selesai capture 10 foto untuk mahasiswa{person_counter}")
        person_counter += 1
    
    # ESC (27) untuk keluar
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*50}")
print(f"Total {total_photos} foto tersimpan")
print(f"Lokasi: foto_mahasiswa/")
print(f"Jumlah mahasiswa: {person_counter - 1}")
print(f"{'='*50}")