#!/usr/bin/env python3
"""
Libreria per la raccolta di dataset di immagini da una camera
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class DatasetCollector(Node):

    def __init__(self):
        super().__init__('dataset_collector')

        # Crea una cartella data_collector nella root
        self.save_path = "/root/data_collector"
        # Se la cartella non esiste, creala
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            self.get_logger().info(f"Creata cartella: {self.save_path}")
        else:
            self.get_logger().info(f"Salvataggio immagini in: {self.save_path}")

        # Inizializza CvBridge
        self.br = CvBridge()
        self.last_image = None
        self.image_count = 0
        self.timer_snapshot = 10.0  

        # Sottoiscrizione al topic /rgb della camera
        self.subscription = self.create_subscription(
            Image,
            '/rgb',  
            self.image_callback,
            10)
        
        # Chiama il salvataggio ogni timer_snapshot secondi
        
        self.timer = self.create_timer(self.timer_snapshot, self.save_callback)

    def image_callback(self, data):
        # Questo callback serve solo ad aggiornare l'ultimo frame disponibile.
        try:
            self.last_image = self.br.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Errore conversione: {e}")

    def save_callback(self):
        # Niente imagine? Non fare nulla
        if self.last_image is None:
            self.get_logger().warn("In attesa del primo frame dalla camera...")
            return

        # Genera il nome del file univoco
        filename = f"box_data_{self.image_count:04d}.jpg" 
        full_path = os.path.join(self.save_path, filename)

        # Scrive il file su disco
        cv2.imwrite(full_path, self.last_image)
        
        self.get_logger().info(f"📸 Salvata immagine {self.image_count}: {full_path}")
        self.image_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = DatasetCollector()
    
    try:
        print("--- AVVIO RACCOLTA DATI ---")
        print("Muovi il robot o le scatole in Isaac Sim.")
        print("Lo script salverà una foto ogni x secondi.")
        print("Premi CTRL+C per finire.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()