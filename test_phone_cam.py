import cv2

PHONE_URL = "http://172.16.12.79:4747/video"   

cap = cv2.VideoCapture(PHONE_URL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to connect to phone camera")
        break

    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
