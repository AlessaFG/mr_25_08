#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Assicurati che i messaggi siano compilati correttamente nel workspace
from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference

from ultralytics import YOLO

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        # Caricamento del modello (gestisce il percorso con ~)
   
   # Scaricherà automaticamente i pesi di YOLOv11 Nano (la versione più leggera)
        self.model = YOLO("best_87_.pt")

        self.br = CvBridge()

        # --- MODIFICA QUI ---
        # Sottoscrizione al topic specificato: /rgb
        self.subscription = self.create_subscription(
            Image,
            '/rgb',  # Topic corretto richiesto da te
            self.camera_callback,
            10)
        # --------------------

        # Publisher
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)

        self.get_logger().info("Nodo avviato. In ascolto su topic: /rgb")

    def camera_callback(self, data):
        # Inizializza il messaggio vuoto
        yolov8_msg = Yolov8Inference()
        yolov8_msg.header.frame_id = "inference"
        yolov8_msg.header.stamp = self.get_clock().now().to_msg()

        try:
            # Conversione: ROS Image -> OpenCV (BGR)
            # Isaac spesso invia RGB, ma OpenCV lavora in BGR.
            # 'desired_encoding="bgr8"' fa la conversione corretta automaticamente.
            img = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Errore cv_bridge: {e}")
            return

        # Inferenza YOLO
        results = self.model(img, verbose=False)

        # Elaborazione risultati
        for r in results:
            boxes = r.boxes
            for box in boxes:
                inference_result = InferenceResult()
                
                # Coordinate bounding box
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                c = box.cls
                
                inference_result.class_name = self.model.names[int(c)]
                inference_result.top = int(b[1])
                inference_result.left = int(b[0])
                inference_result.bottom = int(b[3])
                inference_result.right = int(b[2])

                self.get_logger().info(f"Vedo {inference_result.class_name} a: X={inference_result.left}, Y={inference_result.top}")
                
                yolov8_msg.yolov8_inference.append(inference_result)

        # Pubblica l'immagine con i box disegnati (per debug in Rviz o rqt)
        annotated_frame = results[0].plot()
        try:
            img_msg = self.br.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.img_pub.publish(img_msg)
            self.yolov8_pub.publish(yolov8_msg)
        except Exception as e:
            self.get_logger().error(f"Errore pubblicazione: {e}")

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()
    
    try:
        rclpy.spin(camera_subscriber)
    except KeyboardInterrupt:
        pass
    
    camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()