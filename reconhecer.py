import cv2
import numpy as np
import json
import os

BASE = "face"
MODELO_PATH = "modelo.yml"
LABELS_PATH = "labels.json"

# verificar ficheiros
if not os.path.exists(MODELO_PATH) or not os.path.exists(LABELS_PATH):
    print("Falta o modelo ou os labels. Corre primeiro o treinar.py")
    exit(1)

# carregar modelo LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODELO_PATH)

# carregar labels
with open(LABELS_PATH, "r") as f:
    label_por_nome = json.load(f)

# inverter dict: label -> nome
nome_por_label = {int(v): k for k, v in label_por_nome.items()}

# classificador de faces
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao abrir a câmara")
    exit(1)

# quanto mais baixo, mais “exigente”
# numa máquina fraquinha valores entre 70 e 90 costumam ser ok
LIMIAR_CONF = 80.0

while True:
    ok, frame = camera.read()
    if not ok:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        # recorte da cara
        face_gray = gray[y:y + h, x:x + w]
        face_gray = cv2.resize(face_gray, (200, 200))

        # prever quem é
        label, conf = recognizer.predict(face_gray)

        if conf < LIMIAR_CONF and label in nome_por_label:
            nome = nome_por_label[label]
        else:
            nome = "Desconhecido"

        # desenhar quadrado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # texto debaixo do quadrado
        y_texto = y + h + 30
        cv2.putText(
            frame,
            f"{nome} ({conf:.1f})",
            (x, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Reconhecimento Facial", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()
