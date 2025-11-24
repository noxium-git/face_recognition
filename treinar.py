import cv2
import os
import numpy as np
import pickle

BASE = "faces"

recognizer = cv2.face.LBPHFaceRecognizer_create()

labels = []
faces = []
label_id = {}
current_id = 0

for pessoa in os.listdir(BASE):
    pasta = os.path.join(BASE, pessoa)
    if not os.path.isdir(pasta):
        continue

    label_id[pessoa] = current_id

    for foto in os.listdir(pasta):
        if not foto.endswith(".jpg"):
            continue

        caminho = os.path.join(pasta, foto)
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

        faces.append(img)
        labels.append(current_id)

    current_id += 1

recognizer.train(faces, np.array(labels))

recognizer.save("modelo_lbph.yml")
pickle.dump(label_id, open("labels.pkl", "wb"))

print("Treino concluído!")
