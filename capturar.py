import cv2
import os

nome = input("Nome da pessoa: ")
pasta = f"faces/{nome}"
os.makedirs(pasta, exist_ok=True)

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

contador = 0
MAX = 200

while True:
    ok, frame = cam.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        cv2.imwrite(f"{pasta}/face_{contador}.jpg", face)
        contador += 1

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(frame, f"{contador}/{MAX}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Captura", frame)

    if contador >= MAX or cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
print("Concluído!")
