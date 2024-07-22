import cv2
import numpy as np
import os

# Definir el tamaño del patrón de calibración
pattern_size = (7, 7)

# Preparar las coordenadas de los puntos del patrón en el mundo real
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Listas para almacenar las coordenadas de los puntos del patrón
objpoints_left = []
objpoints_right = []
imgpoints_left = []
imgpoints_right = []

# Directorio de las imágenes capturadas
output_folder = "/home/jairo/Proyecto_de_Robotica_Autonoma/Left_right"

# Leer imágenes de ambas cámaras
num_images = 70  # Ajuste el número de imágenes a 70

for count in range(1, num_images + 1):
    img_name_left = os.path.join(output_folder, f"left_{count:02d}.jpg")
    img_name_right = os.path.join(output_folder, f"right_{count:02d}.jpg")
    
    img_l = cv2.imread(img_name_left)
    img_r = cv2.imread(img_name_right)
    
    if img_l is None or img_r is None:
        print(f"Imagen no encontrada: {img_name_left} o {img_name_right}")
        continue
    
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Detectar esquinas del patrón
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, None)

    if ret_l and ret_r:
        objpoints_left.append(objp)
        imgpoints_left.append(corners_l)
        objpoints_right.append(objp)
        imgpoints_right.append(corners_r)

        # Dibujar las esquinas detectadas en las imágenes
        cv2.drawChessboardCorners(img_l, pattern_size, corners_l, ret_l)
        cv2.drawChessboardCorners(img_r, pattern_size, corners_r, ret_r)

        # Mostrar imágenes con esquinas detectadas
        cv2.imshow('Left Image', img_l)
        cv2.imshow('Right Image', img_r)
        cv2.waitKey(500)  # Mostrar cada imagen durante 500 ms

print(f"Número de imágenes válidas en la cámara izquierda: {len(imgpoints_left)}")
print(f"Número de imágenes válidas en la cámara derecha: {len(imgpoints_right)}")

# Verificar si hay suficientes puntos para la calibración
if len(imgpoints_left) < 3 or len(imgpoints_right) < 3:
    print("Error: No se encontraron suficientes puntos de esquina en las imágenes. Asegúrate de que el patrón de calibración es claramente visible en todas las imágenes.")
else:
    # Calibrar cámaras individuales
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints_left, imgpoints_left, gray_l.shape[::-1], None, None)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints_right, imgpoints_right, gray_r.shape[::-1], None, None)

    print("Calibración de la cámara izquierda:")
    print("Matriz de cámara (mtx_l):")
    print(mtx_l)
    print("Coeficientes de distorsión (dist_l):")
    print(dist_l)
    print("Vectores de rotación (rvecs_l):")
    print(rvecs_l)
    print("Vectores de traslación (tvecs_l):")
    print(tvecs_l)

    print("\nCalibración de la cámara derecha:")
    print("Matriz de cámara (mtx_r):")
    print(mtx_r)
    print("Coeficientes de distorsión (dist_r):")
    print(dist_r)
    print("Vectores de rotación (rvecs_r):")
    print(rvecs_r)
    print("Vectores de traslación (tvecs_r):")
    print(tvecs_r)

    # Calibrar el sistema estéreo
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints_left, imgpoints_left, imgpoints_right, mtx_l, dist_l, mtx_r, dist_r,
        gray_l.shape[::-1], criteria=criteria, flags=flags)

    print("\nCalibración estéreo:")
    print("Rotación relativa entre las cámaras (R):")
    print(R)
    print("Traslación relativa entre las cámaras (T):")
    print(T)

# Cerrar todas las ventanas
cv2.destroyAllWindows()
'''
Rotación relativa entre las cámaras (R):
[[ 0.76908125 -0.15617836  0.61977605]
 [-0.16865219  0.88573109  0.43247761]
 [-0.61649856 -0.437137    0.65485935]]
Traslación relativa entre las cámaras (T):
[[-5.1720821 ]
 [-1.3369988 ]
 [ 3.51333005]]
'''