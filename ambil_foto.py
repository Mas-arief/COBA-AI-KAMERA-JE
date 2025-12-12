import cv2
import os

# Pastikan folder ada
if not os.path.exists("foto_mahasiswa"):
    os.makedirs("foto_mahasiswa")
    print("✓ Folder foto_mahasiswa dibuat")

# Buka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi!")
    exit()

print("\n" + "="*50)
print("AMBIL FOTO MAHASISWA")
print("="*50)
print("📌 Tekan SPASI untuk ambil foto")
print("📌 Tekan ESC untuk keluar\n")

counter = 1

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Gagal membaca frame")
        break
    
    # Tampilkan instruksi di layar
    cv2.putText(frame, "SPASI = Ambil Foto | ESC = Keluar", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Ambil Foto', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # SPASI (32) untuk ambil foto
    if key == 32:
        # Auto generate filename
        filename = f"mahasiswa{counter}.jpg"
        filepath = os.path.join("foto_mahasiswa", filename)
        
        # Simpan foto
        success = cv2.imwrite(filepath, frame)
        
        if success:
            print(f"✓ Foto tersimpan: {filename}")
            counter += 1
            
            # Tampilkan feedback visual
            cv2.putText(frame, "FOTO TERSIMPAN!", 
                       (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow('Ambil Foto', frame)
            cv2.waitKey(1000)  # Tahan 1 detik
        else:
            print(f"❌ Gagal menyimpan foto")
    
    # ESC (27) untuk keluar
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Total {counter-1} foto tersimpan")
print(f"📁 Lokasi: foto_mahasiswa/")