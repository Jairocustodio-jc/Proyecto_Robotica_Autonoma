import cv2
import numpy as np
import glob

# Definición del tamaño del tablero de ajedrez
chessboard_size = (7, 7)
square_size = 1.0  # El tamaño del cuadrado puede ser en cualquier unidad (p. ej., milímetros o pulgadas)

# Preparar los puntos del objeto, como (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

# Arrays para almacenar los puntos del objeto y los puntos de la imagen de todas las imágenes.
objpoints = []  # Puntos 3D en el espacio del mundo real
imgpoints = []  # Puntos 2D en el plano de la imagen

# Obtener las imágenes del patrón de calibración
images = glob.glob('/home/jairo/Proyecto_de_Robotica_Autonoma/Camara_2/*.png')
#images = glob.glob('/home/jairo/Proyecto_de_Robotica_Autonoma/Camara_1/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontrar las esquinas del tablero de ajedrez
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Si se encuentran las esquinas, agregar los puntos del objeto y los puntos de la imagen
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Dibujar y mostrar las esquinas
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrar la cámara
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Imprimir los resultados
print("Matriz de calibración:\n", mtx)
print("Coeficientes de distorsión:\n", dist)
'''
Camara1:
Matriz de calibración:
 [[727.25065806   0.         358.07050047]
 [  0.         688.81457417 212.54883042]
 [  0.           0.           1.        ]]
Coeficientes de distorsión:
 [[-5.62880133e-01  1.15646128e+00  3.38379465e-02 -2.59338275e-03
  -2.77290874e+00]]
'''
'''
Camara2:
Matriz de calibración:
 [[643.85293628   0.         282.73972176]
 [  0.         710.03785289 263.91868641]
 [  0.           0.           1.        ]]
Coeficientes de distorsión:
 [[-5.85931862e-01  1.45615487e+00 -2.80276334e-02 -1.43697018e-03
  -2.69141356e+00]]

'''