#!/usr/bin/env python3
#   Este nodo se suscribe a una imagen de ROS, la convierte en una matriz de
#   OpenCV, ejecuta detección de objeto basada en SIFT feature matching y la muestra en pantalla
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

I1 = cv2.imread('catkin_ws/src/test.jpeg',0)
sift = cv2.SIFT_create()
keypts1, descriptores1 = sift.detectAndCompute(I1, None)

# Parámetros de correspondencia usando FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

good_matches = []
dst_corners = []

class Cam(object):
  def __init__(self, topic_name="/usb_cam/image_raw"):
    self.bridge = CvBridge()
    self.image = np.zeros((10,10))
    isub = rospy.Subscriber(topic_name, Image, self.image_callback)

  def image_callback(self, img):
    self.image = self.bridge.imgmsg_to_cv2(img, "bgr8")

  def get_image(self):
    return self.image


if __name__ == '__main__':

  # Inicializar el nodo de ROS
  rospy.init_node('camera_node')

  # Objeto que se suscribe al tópico de la cámara
  topic_name = "/usb_cam/image_raw"
  cam = Cam(topic_name)

  # Tópico para publicar una imagen de salida
  pub = rospy.Publisher('centroide_out', Point, queue_size=1)
  point_msg = Point()

  # Frecuencia del bucle principal
  freq = 10
  rate = rospy.Rate(freq)
  # Bucle principal
  while not rospy.is_shutdown():

    # Obtener la imagen del tópico de ROS en formato de OpenCV
    I2 = cam.get_image()
    Ig = cv2.normalize(I2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    Ig = cv2.GaussianBlur(Ig, (5, 5), 0)  # Aplicar un filtro Gaussiano de 5x5
    # Realizar algún tipo de procesamiento sobre la imagen

    if I2.shape != (10,10):
      # Descriptores usando SIFT
      keypts2, descriptores2 = sift.detectAndCompute(Ig, None)
      if descriptores2 is not None and (descriptores2.shape[0] > 1):
          # Correspondencia usando FLANN
          matches = flann.knnMatch(descriptores1, descriptores2, k=2)

          # Correspondencias adecuadas según el ratio
          good_matches = []
          for m, n in matches:
              if m.distance < 0.5 * n.distance:
                  good_matches.append(m)

          MIN_NUM_GOOD_MATCHES = 10

          if len(good_matches) >= MIN_NUM_GOOD_MATCHES:
              # Coordenadas 2D de los correspondientes keypoints
              src_pts = np.float32( [keypts1[m.queryIdx].pt for m in good_matches] ).reshape(-1, 1, 2)
              dst_pts = np.float32( [keypts2[m.trainIdx].pt for m in good_matches] ).reshape(-1, 1, 2)

              # Homografía
              M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
              mask_matches = mask.ravel().tolist()

              # Transformación de perspectiva: proyectar los bordes rectangulares
              # en la escena para dibujar un borde
              h, w = I1.shape
              src_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
              if (len(src_corners) > 2 and M is not None):
                  dst_corners = cv2.perspectiveTransform(src_corners, M)
                  dst_corners = dst_corners.astype(np.int32)

                  # centroide
                  centroide = np.mean(dst_corners, axis=0)
                  xc, yc = centroide[0]
                  xc = int(round(xc))
                  yc = int(round(yc))
                  zc = ((dst_corners[1,0,1]-dst_corners[0,0,1])+(dst_corners[2,0,1]-dst_corners[3,0,1]))//2
                  point_msg.x = (33/100) - (zc/2500)
                  point_msg.y = (933/9400) - (39*xc/117500)
                  point_msg.z = 0.06
                  centroide = (xc,yc)
                  cv2.line(I2, (xc+10, yc), (xc-10, yc), (0, 0, 255), 3, cv2.LINE_AA)
                  cv2.line(I2, (xc, yc+10), (xc, yc-10), (0, 0, 255), 3, cv2.LINE_AA)

                  # Bordes detectados
                  num_corners = len(dst_corners)
                  for i in range(num_corners):
                      x0, y0 = dst_corners[i][0]
                      if i == num_corners - 1:
                          next_i = 0
                      else:
                          next_i = i + 1
                      x1, y1 = dst_corners[next_i][0]
                      cv2.line(I2, (x0, y0), (x1, y1), (0, 0, 255), 3, cv2.LINE_AA)
      else:
         point_msg = Point()


    # Mostrar la imagen
    cv2.imshow("Imagen Camara", I2)

    # Esperar al bucle para actualizar
    cv2.waitKey(1)
    # Asignar los valores al mensaje
    # Publicar el mensaje
    pub.publish(point_msg)
    rate.sleep()

  cv2.destroyAllWindows()
