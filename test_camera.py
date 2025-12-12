import cv2

print("Test kamera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera tidak terdeteksi!")
else:
    print("✅ Kamera OK! Tekan 'q' untuk keluar")
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('Test Kamera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()