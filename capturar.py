import cv2
import os

# número de imagens a capturar por pessoa
MAX_FOTOS = 200

# pasta base onde ficam as caras
BASE = "face"

# carregar classificador de rosto do OpenCV
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

nome = input("Nome da pessoa: ").strip().lower()

if not nome:
    print("Nome vazio, a sair...")
    exit(1)

pasta_pessoa = os.path.join(BASE, nome)
os.makedirs(pasta_pessoa, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao abrir a câmara")
    exit(1)

contador = 0
print("Mantém a cara centrada e muda ligeiramente a expressão de vez em quando...")
print("Pressiona ESC para parar antes de chegar ao fim.")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    # reduzir resolução para ser mais leve
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detetar caras
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # se encontrar pelo menos uma cara
    if len(faces) > 0:
        # pega na maior cara (para evitar apanhar o colega ao lado)
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces_sorted[0]

        # desenhar quadrado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # recorte em tons de cinzento
        face_gray = gray[y:y + h, x:x + w]

        # normalizar para um tamanho fixo
        face_gray = cv2.resize(face_gray, (200, 200))

        # guardar
        caminho = os.path.join(pasta_pessoa, f"face_{contador}.jpg")
        cv2.imwrite(caminho, face_gray)
        contador += 1

    # mostrar contador no canto
    texto = f"{contador}/{MAX_FOTOS}"
    cv2.putText(frame, texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Captura", frame)

    # tecla
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:  # ESC
        break
    if contador >= MAX_FOTOS:
        break

camera.release()
cv2.destroyAllWindows()

print(f"Captura concluída! {contador} imagens guardadas em {pasta_pessoa}")
