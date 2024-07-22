import cv2
import numpy as np
import os

# Definir el tamaño del patrón de calibración
pattern_size = (7, 7)

# Inicializar las cámaras
cap_left = cv2.VideoCapture(2)  # Reemplaza con el índice de tu cámara izquierda
cap_right = cv2.VideoCapture(4)  # Reemplaza con el índice de tu cámara derecha

# Verificar que las cámaras se abran correctamente
if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: No se pudo abrir una o ambas cámaras")
    exit()

# Crear la carpeta para guardar las imágenes si no existe
output_folder = "/home/jairo/Proyecto_de_Robotica_Autonoma/Left_right"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Función para capturar y guardar imágenes
def capture_images():
    count = 0
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("Error: No se pudo capturar la imagen de una o ambas cámaras")
            continue

        # Convertir las imágenes a escala de grises
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # Buscar el patrón de calibración en ambas imágenes
        ret_l, corners_l = cv2.findChessboardCorners(gray_left, pattern_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_right, pattern_size, None)

        # Aplicar filtro de suavizado
        smoothed_left = cv2.GaussianBlur(frame_left, (3, 3), 0)
        smoothed_right = cv2.GaussianBlur(frame_right, (3, 3), 0)

        # Mostrar las imágenes capturadas en tiempo real con el filtro de suavizado aplicado
        if ret_l:
            cv2.drawChessboardCorners(smoothed_left, pattern_size, corners_l, ret_l)
        else:
            # Mostrar mensaje si no se detecta el patrón en la cámara izquierda
            cv2.putText(smoothed_left, "No Pattern Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if ret_r:
            cv2.drawChessboardCorners(smoothed_right, pattern_size, corners_r, ret_r)
        else:
            # Mostrar mensaje si no se detecta el patrón en la cámara derecha
            cv2.putText(smoothed_right, "No Pattern Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('left', smoothed_left)
        cv2.imshow('right', smoothed_right)

        # Esperar la entrada del teclado
        key = cv2.waitKey(1) & 0xFF

        # Guardar imágenes si se presiona 'S'
        if key == ord('s'):
            count += 1
            img_name_left = os.path.join(output_folder, f"left_{count:02d}.jpg")
            img_name_right = os.path.join(output_folder, f"right_{count:02d}.jpg")
            cv2.imwrite(img_name_left, frame_left)
            cv2.imwrite(img_name_right, frame_right)
            print(f"Imágenes capturadas y guardadas: {img_name_left}, {img_name_right}")

        # Salir con la tecla 'q'
        if key == ord('q'):
            break

# Capturar imágenes hasta que se presione 'q'
capture_images()

# Liberar las cámaras y cerrar las ventanas
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
