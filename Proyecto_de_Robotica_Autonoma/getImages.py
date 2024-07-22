# Basado en https://www.youtube.com/watch?v=TkwBzv9L5aY
import cv2
import os

# Asegúrate de que la ruta de destino exista
#save_path = '/home/jairo/Proyecto_de_Robotica_Autonoma/Camara_1'
save_path = '/home/jairo/Proyecto_de_Robotica_Autonoma/Camara_2'

if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(4)  # Cambia el índice si es necesario

num = 0

while cap.isOpened():
    succes, img = cap.read()

    if not succes:
        print("Failed to grab frame")
        break

    k = cv2.waitKey(5)

    if k == ord('q'):
        break
    elif k == ord('s'):  # Espera la tecla 's' para guardar y salir
        filename = os.path.join(save_path, f'img{num}.png')
        cv2.imwrite(filename, img)
        print("Image saved!")
        num += 1

    cv2.imshow('Img', img)

# Libera y destruye todas las ventanas antes de terminar
cap.release()
cv2.destroyAllWindows()
