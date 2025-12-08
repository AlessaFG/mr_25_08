#!/usr/bin/env python3

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

        # 1. Configurazione Cartella di Salvataggio
        # Crea una cartella "dataset_raw" nella home del tuo utente
        self.save_path = "/root/data_collector"
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            self.get_logger().info(f"Creata cartella: {self.save_path}")
        else:
            self.get_logger().info(f"Salvataggio immagini in: {self.save_path}")

        self.br = CvBridge()
        self.last_image = None
        self.image_count = 0

        # 2. Subscription (Ascolta la camera)
        self.subscription = self.create_subscription(
            Image,
            '/rgb',  # Assicurati che sia il topic corretto
            self.image_callback,
            10)

        # 3. Timer (Il "Fotografo")
        # Chiama la funzione save_callback ogni 5.0 secondi
        self.timer = self.create_timer(5.0, self.save_callback)

    def image_callback(self, data):
        # Questo callback serve SOLO ad aggiornare l'ultimo frame disponibile.
        # Non salviamo qui, altrimenti salveremmo 30 foto al secondo!
        try:
            self.last_image = self.br.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Errore conversione: {e}")

    def save_callback(self):
        # Se non abbiamo ancora ricevuto immagini, non fare nulla
        if self.last_image is None:
            self.get_logger().warn("In attesa del primo frame dalla camera...")
            return

        # Genera il nome del file univoco
        filename = f"box_data_{self.image_count:04d}.jpg" # Es: box_data_0001.jpg
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
        print("Lo script salverà una foto ogni 10 secondi.")
        print("Premi CTRL+C per finire.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()