import cv2
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

print("Test kamera...")
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
else:
    print("✅ Kamera OK! Tekan 'q' untuk keluar")
    
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
        
        # Tambah info di layar
        cv2.putText(display_frame, "TEST KAMERA - Tekan 'q' untuk keluar", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Test Kamera', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("✅ Kamera ditutup")
            break

cap.release()
cv2.destroyAllWindows()