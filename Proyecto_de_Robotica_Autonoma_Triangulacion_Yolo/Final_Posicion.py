#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Configuración de ROS
rospy.init_node('object_detection', anonymous=True)
position_publisher = rospy.Publisher('/object_position', Point, queue_size=10)

# Cargar modelo YOLO desde el archivo .pt
model = YOLO("/home/jairo/robotica_industrial_ws/src/entrega1_ri/src/yolov8x.pt")

# Función para detectar objetos usando YOLO
def detect_objects(img, model):
    results = model.predict(img, conf=0.6, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Mover a CPU y luego convertir a numpy
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidencias
    classes = results[0].boxes.cls.cpu().numpy()  # Clases
    return boxes, confidences, classes

# Función para dibujar cuadros de delimitación en una imagen
def draw_boxes(img, boxes, confidences):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        conf = confidences[i] if i < len(confidences) else 0
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

# Inicializar cámaras (cambia los índices según sea necesario)
cap1 = cv2.VideoCapture(4)  # Cambia el índice si es necesario
cap2 = cv2.VideoCapture(2)  # Cambia el índice si es necesario

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: No se pudo abrir una o ambas cámaras.")
    exit()

last_print_time = time.time()  # Tiempo de la última impresión

try:
    while not rospy.is_shutdown():
        # Captura un frame de cada cámara
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error: No se pudo obtener un frame de una o ambas cámaras.")
            break

        # Detecta los objetos en los frames
        boxes1, confidences1, _ = detect_objects(frame1, model)
        boxes2, confidences2, _ = detect_objects(frame2, model)

        # Dibuja los cuadros de delimitación en las imágenes
        frame1 = draw_boxes(frame1, boxes1, confidences1)
        frame2 = draw_boxes(frame2, boxes2, confidences2)

        if len(boxes1) > 0 and len(boxes2) > 0:
            x1, y1, x2, y2 = boxes1[0]
            x1_center = (x1 + x2) / 2
            y1_center = (y1 + y2) / 2

            x3, y3, x4, y4 = boxes2[0]
            x2_center = (x3 + x4) / 2
            y2_center = (y3 + y4) / 2

            # Matrices de calibración actualizadas
            K1 = np.array([[727.25065806, 0, 358.07050047],
                           [0, 688.81457417, 212.54883042],
                           [0, 0, 1]])

            K2 = np.array([[643.85293628, 0, 282.73972176],
                           [0, 710.03785289, 263.91868641],
                           [0, 0, 1]])

            # Rotación y traslación relativas entre las cámaras
            R = np.array([[0.4416843,-0.48575358,0.75429333],
                          [-0.40498359,0.64226812,0.65075337],
                          [-0.80056434,-0.59290397,0.08695752]])

            T = np.array([-1.57594006, 6.67367169, 67.59077644])

            # Matrices de proyección
            P1 = np.dot(K1, np.hstack((R, np.zeros((3, 1)))))
            P2 = np.dot(K2, np.hstack((R, T.reshape(3, 1))))

            # Triangulación para encontrar la posición 3D del objeto
            points4D = cv2.triangulatePoints(P1, P2, 
                                             np.array([[x1_center, y1_center]], dtype=np.float32).T, 
                                             np.array([[x2_center, y2_center]], dtype=np.float32).T)
            points3D = points4D / points4D[3]  # Convertir de coordenadas homogéneas a 3D

            # Extraer coordenadas 3D
            x, y, z = points3D[:3].flatten()

            # Enviar la posición 3D al tópico de ROS
            position_msg = Point()
            position_msg.x = x
            position_msg.y = y
            position_msg.z = z
            position_publisher.publish(position_msg)

            # Imprimir la posición 3D cada 1 segundo
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                print(f"Posición 3D estimada del objeto: x={x}, y={y}, z={z}")
                last_print_time = current_time

        # Mostrar las imágenes en vivo con los cuadros de delimitación
        cv2.imshow("Camara 1", frame1)
        cv2.imshow("Camara 2", frame2)

        # Cierra las ventanas con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Libera las cámaras y cierra todas las ventanas de OpenCV
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
