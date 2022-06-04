#Projeto simplificado de reconhecimento facila em python
#codigo disponivel no github para estudo e aperfeicoamento
#link https://github.com/cristianopimenta/reconhecimentofacial
import cv2
#variaveis
classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostras = 25

#identificador inicial do arquivo em reconhecimento facial
id = input('Digite seu identificador: ')
#tamanho do arquivo Foto
largura, altura = 220, 220

print("Capturando as faces...")

#Loop com quantidade minima de fotos
while (True):
    conectado, imagem = camera.read()

    #fotos em tons de Cinza sao melhores para fazer reconhecimento
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    #escala para foto
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5,minSize=(150,150))

    for (x,y,l,a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + 1, y+a), (0, 0, 255), 2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura,altura))
            cv2.imwrite("fotos/pessoa." + str(id)+ "." + str(amostra)+ ".jpg", imagemFace)
            print("[Foto" + str(amostra) + "capturada com sucesso]")
            amostra += 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras + 1):
        break

print("Faces capturadas com sucesso")
camera.relase()
cv2.destroyAllWindows()
