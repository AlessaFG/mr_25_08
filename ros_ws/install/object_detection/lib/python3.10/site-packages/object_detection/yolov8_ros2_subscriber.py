#!/usr/bin/env python3

import cv2
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Assicurati che il messaggio sia compilato
from yolov8_msgs.msg import Yolov8Inference

bridge = CvBridge()
img = None # Inizializziamo a None per evitare crash all'avvio

class Camera_subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        # Topic corretto per Isaac Sim
        self.subscription = self.create_subscription(
            Image,
            '/rgb', 
            self.camera_callback,
            10)
        self.get_logger().info("Camera Subscriber avviato su /rgb")

    def camera_callback(self, data):
        global img
        try:
            # Isaac manda RGB, convertiamo in BGR per OpenCV
            img = bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Errore conversione frame: {e}")

class Yolo_subscriber(Node):

    def __init__(self):
        super().__init__('yolo_subscriber')

        self.subscription = self.create_subscription(
            Yolov8Inference,
            '/Yolov8_Inference',
            self.yolo_callback,
            10)
        
        self.cnt = 0
        self.img_pub = self.create_publisher(Image, "/inference_result_cv2", 1)
        self.get_logger().info("Yolo Subscriber avviato")

    def yolo_callback(self, data):
        global img
        
        # Controllo di sicurezza: Se non abbiamo ancora ricevuto un'immagine dalla camera,
        # non possiamo disegnarci sopra.
        if img is None:
            self.get_logger().warn("Ricevuti dati YOLO ma nessuna immagine dalla camera ancora.")
            return

        # Creiamo una copia dell'immagine per non sporcare il frame originale
        # se serve ad altri processi (opzionale ma consigliato)
        current_frame = img.copy() 

        for r in data.yolov8_inference:
            class_name = r.class_name
            top = r.top
            left = r.left
            bottom = r.bottom
            right = r.right
            
            # Log corretto usando 'self'
            self.get_logger().info(f"{self.cnt} {class_name} : T={top}, L={left}, B={bottom}, R={right}")
            
            # --- FIX COORDINATE ---
            # OpenCV usa (x, y) ovvero (Colonna, Riga).
            # I dati arrivano come Top(y), Left(x).
            # Quindi dobbiamo passare (left, top), (right, bottom)
            p1 = (left, top)
            p2 = (right, bottom)
            
            cv2.rectangle(current_frame, p1, p2, (0, 255, 0), 2)
            
            # Opzionale: Aggiungi il testo
            cv2.putText(current_frame, class_name, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.cnt += 1

        self.cnt = 0
        
        # Pubblica il risultato
        try:
            img_msg = bridge.cv2_to_imgmsg(current_frame, encoding="bgr8")
            self.img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Errore pubblicazione immagine: {e}")

if __name__ == '__main__':
    rclpy.init(args=None)
    
    camera_node = Camera_subscriber()
    yolo_node = Yolo_subscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(camera_node)
    executor.add_node(yolo_node)

    # Eseguiamo l'executor in un thread separato
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    # Questo loop serve solo a tenere vivo il main thread finché non premi Ctrl+C
    try:
        while rclpy.ok():
            # Un semplice sleep per non consumare CPU inutilmente nel main thread
            # Lasciamo che l'executor lavori nel background thread
            pass 
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    # Non serve joinare se il thread è daemon, ma per pulizia:
    # executor_thread.join()