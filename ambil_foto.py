import cv2
import os
import numpy as np

def improve_image_quality(image):
    """Improve kualitas image dari webcam"""
    # Apply Gaussian Blur untuk reduce noise
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply histogram equalization untuk brightness lebih baik
    equalized = cv2.equalizeHist(denoised)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(equalized)
    
    return enhanced

# Pastikan folder ada
if not os.path.exists("foto_mahasiswa"):
    os.makedirs("foto_mahasiswa")
    print("✓ Folder foto_mahasiswa dibuat")

# Buka kamera
cap = cv2.VideoCapture(0)

# Set camera properties untuk kualitas terbaik
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
cap.set(cv2.CAP_PROP_CONTRAST, 50)

if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi!")
    exit()

print("\n" + "="*50)
print("AMBIL FOTO MAHASISWA (10 FOTO PER ORANG)")
print("="*50)
print("📌 Tekan SPASI untuk mulai capture 10 foto")
print("📌 Tekan ESC untuk keluar\n")

person_counter = 1
total_photos = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Gagal membaca frame")
        break
    
    # Flip frame agar tidak mirror
    frame = cv2.flip(frame, 1)
    
    # Improve kualitas gambar dari webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = improve_image_quality(gray)
    
    # Apply slight sharpening untuk lebih jelas
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]]) / 1.0
    gray = cv2.filter2D(gray, -1, kernel)
    
    # Convert back to BGR untuk display
    display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Tampilkan instruksi di layar
    cv2.putText(display_frame, "SPASI = Capture 10 Foto | ESC = Keluar", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Total Foto: {total_photos}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow('Ambil Foto', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # SPASI (32) untuk capture 10 foto
    if key == 32:
        print(f"\n🎬 Capturing 10 foto untuk mahasiswa{person_counter}...")
        
        for i in range(1, 11):
            # Capture foto
            ret, capture_frame = cap.read()
            if not ret:
                print("❌ Gagal membaca frame")
                break
            
            capture_frame = cv2.flip(capture_frame, 1)
            
            # Generate filename dengan format: mahasiswa{nomor}_{no_foto}.jpg
            filename = f"mahasiswa{person_counter}_{i:02d}.jpg"
            filepath = os.path.join("foto_mahasiswa", filename)
            
            # Simpan foto
            success = cv2.imwrite(filepath, capture_frame)
            
            if success:
                print(f"   ✓ Foto {i}/10 tersimpan: {filename}")
                total_photos += 1
                
                # Show countdown di display
                gray = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2GRAY)
                gray = improve_image_quality(gray)
                gray = cv2.filter2D(gray, -1, kernel)
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                cv2.putText(display, f"Capturing: {i}/10", 
                           (display.shape[1]//2 - 100, display.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow('Ambil Foto', display)
                cv2.waitKey(300)  # Delay 300ms antar foto
            else:
                print(f"   ❌ Gagal menyimpan foto {i}")
        
        print(f"✅ Selesai capture 10 foto untuk mahasiswa{person_counter}")
        person_counter += 1
    
    # ESC (27) untuk keluar
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*50}")
print(f"✅ Total {total_photos} foto tersimpan")
print(f"📁 Lokasi: foto_mahasiswa/")
print(f"📊 Jumlah mahasiswa: {person_counter - 1}")
print(f"{'='*50}")