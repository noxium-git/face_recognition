import cv2
import os
import numpy as np
import json

BASE = "face"
MODELO_PATH = "modelo.yml"
LABELS_PATH = "labels.json"

# garantir que há pastas
if not os.path.isdir(BASE):
    print(f"Pasta '{BASE}' não existe. Corre primeiro o capturar.py")
    exit(1)

faces = []
labels = []
label_por_nome = {}
proximo_label = 0

for nome in os.listdir(BASE):
    pasta_pessoa = os.path.join(BASE, nome)
    if not os.path.isdir(pasta_pessoa):
        continue

    # atribuir label numérico a este nome
    if nome not in label_por_nome:
        label_por_nome[nome] = proximo_label
        proximo_label += 1

    label = label_por_nome[nome]
    print(f"\nA carregar imagens de: {nome}")

    for ficheiro in os.listdir(pasta_pessoa):
        if not ficheiro.lower().endswith(".jpg"):
            continue

        caminho = os.path.join(pasta_pessoa, ficheiro)
        img_gray = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            print("  Ignorada (não deu para ler):", ficheiro)
            continue

        # garantir tamanho igual ao usado na captura
        img_gray = cv2.resize(img_gray, (200, 200))

        faces.append(img_gray)
        labels.append(label)

print("\nTotal de imagens usadas para treino:", len(faces))

if len(faces) == 0:
    print("Nenhuma imagem válida encontrada. Verifica a pasta 'faces/'.")
    exit(1)

# criar reconhecedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

print("A treinar modelo...")
recognizer.train(faces, np.array(labels))

# guardar modelo
recognizer.write(MODELO_PATH)

# guardar dicionário de labels
with open(LABELS_PATH, "w") as f:
    json.dump(label_por_nome, f)

print("\n✔ Treino concluído!")
print(f"Modelo guardado em: {MODELO_PATH}")
print(f"Labels guardados em: {LABELS_PATH}")
