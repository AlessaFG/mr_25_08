#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference

from ultralytics import YOLO

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.model = YOLO("best_87_.pt")
        self.br = CvBridge()
        self.latest_depth_img = None

        # Parametri EMA
        self.ema_alpha = 0.3
        # Il dizionario ora salverà: { ID_NUMERICO: DISTANZA }
        # Esempio: { 1: 1.54, 2: 3.20 }
        self.ema_history = {} 

        # SOTTOSCRIZIONE RGB
        self.subscription = self.create_subscription(
            Image,
            '/rgb',  
            self.camera_callback,
            10)

        # SOTTOSCRIZIONE DEPTH
        self.depth_subscription = self.create_subscription(
            Image,
            '/depth', # <--- Controlla sempre che sia giusto
            self.depth_callback,
            10)

        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)

        self.get_logger().info("Nodo Tracker + EMA avviato.")

    def depth_callback(self, data):
        try:
            self.latest_depth_img = self.br.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Errore depth bridge: {e}")

    def camera_callback(self, data):
        yolov8_msg = Yolov8Inference()
        yolov8_msg.header.frame_id = "inference"
        yolov8_msg.header.stamp = self.get_clock().now().to_msg()

        try:
            img = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Errore cv_bridge: {e}")
            return

        # --- MODIFICA FONDAMENTALE: ABILITIAMO IL TRACKING ---
        # persist=True mantiene l'ID tra un frame e l'altro
        results = self.model.track(img, persist=True, verbose=False)
        
        # YOLO disegna già i box e anche gli ID se usiamo .track()
        annotated_frame = results[0].plot(labels=False, conf=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                inference_result = InferenceResult()
                
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                c = box.cls
                
                # --- RECUPERO ID ---
                # Se il tracker non è sicuro, box.id potrebbe essere None
                track_id = -1
                if box.id is not None:
                    track_id = int(box.id.item())
                # -------------------

                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                class_name = self.model.names[int(c)]

                inference_result.class_name = class_name
                inference_result.top = y1
                inference_result.left = x1
                inference_result.bottom = y2
                inference_result.right = x2
                
                # --- CALCOLO DISTANZA + EMA CON ID ---
                dist_str = ""
                if self.latest_depth_img is not None:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    h_depth, w_depth = self.latest_depth_img.shape

                    if 0 <= cx < w_depth and 0 <= cy < h_depth:
                        raw_dist = self.latest_depth_img[cy, cx]
                        
                        if not np.isnan(raw_dist) and not np.isinf(raw_dist):
                            raw_dist = float(raw_dist)

                            # LOGICA EMA BASATA SU ID
                            # Applichiamo il filtro solo se abbiamo un ID valido
                            if track_id != -1:
                                if track_id in self.ema_history:
                                    prev_dist = self.ema_history[track_id]
                                    smoothed_dist = (self.ema_alpha * raw_dist) + ((1.0 - self.ema_alpha) * prev_dist)
                                else:
                                    smoothed_dist = raw_dist
                                
                                self.ema_history[track_id] = smoothed_dist
                                dist_str = f"ID:{track_id} | {smoothed_dist:.2f}m"
                            else:
                                # Se non ha ID (succede nei primi frame), usa valore grezzo
                                dist_str = f"{raw_dist:.2f}m"
                            
                            self.get_logger().info(f"{class_name} (ID {track_id}): {dist_str}")

                            # Disegna ID e Distanza sull'immagine
                            cv2.putText(annotated_frame, dist_str, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        else:
                            pass # Distanza non valida
                # -------------------------------

                yolov8_msg.yolov8_inference.append(inference_result)

        try:
            img_msg = self.br.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.img_pub.publish(img_msg)
            self.yolov8_pub.publish(yolov8_msg)
        except Exception as e:
            self.get_logger().error(f"Errore pubblicazione: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()