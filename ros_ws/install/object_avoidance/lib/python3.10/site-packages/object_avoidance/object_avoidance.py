#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class ImprovedObjectAvoidanceNode(Node):

    def __init__(self):
        super().__init__('improved_object_avoidance_node')

        # Subscriber e Publisher
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10
        )
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Parametri di Evitamento
        self.safe_distance = 0.5  # Distanza di sicurezza (metri)
        self.forward_speed = 0.2  # Velocità lineare quando si procede
        self.turn_speed = 0.5     # Velocità angolare di rotazione

        self.get_logger().info('Improved Object Avoidance Node Started')
    
    def get_sector_min(self, ranges, start_angle, end_angle, angle_increment):
        """Calcola la distanza minima in un settore dello scan."""
        # Converte gli angoli (gradi) in indici del vettore ranges
        # L'indice 0 di ranges corrisponde all'angolo start_angle_rad (di solito -180 gradi o 0)
        
        # Supponendo che il LiDAR copra 360 gradi o che il centro sia 0 gradi
        # e i range siano ordinati angolarmente.
        
        # Nota: La conversione qui è generica; se lo scan ha un angolo iniziale (angle_min)
        # diverso da -pi (o 0), la logica di indicizzazione deve essere più robusta.
        
        # Per semplicità e robustezza: usiamo i primi e gli ultimi N elementi (se 360 gradi)
        # o filtriamo se non copriamo 360 gradi.
        
        # Se il LiDAR è frontale (0° al centro):
        # I primi elementi e gli ultimi elementi (vicino a 0°) sono il settore centrale/frontale.
        # I settori laterali corrispondono agli angoli 90° e -90°.

        # --- Logica Semplificata per il 2D (settore Frontale, Sinistro, Destro) ---
        
        # Se il LiDAR copre 360 gradi (es. 720 punti)
        num_ranges = len(ranges)
        
        # Esempio per 360 gradi, 720 punti (0 gradi al centro, -180° a sinistra, +180° a destra)
        # 1. Settore Centrale (Frontale)
        #    Corrisponde agli angoli vicini a 0 gradi. Es: -30 a +30 gradi
        # 2. Settore Destro
        #    Corrisponde agli angoli negativi. Es: -90 a -30 gradi
        # 3. Settore Sinistro
        #    Corrisponde agli angoli positivi. Es: +30 a +90 gradi
        
        # Se 720 punti: 720 / 360 = 2 punti per grado
        # Frontale (60 gradi): 60 * 2 = 120 punti. Dagli indici (720/2 - 60) a (720/2 + 60)
        
        # Poiché il tuo codice originale usava solo il minimo, useremo una semplice suddivisione in 3 zone
        # che funziona per la maggior parte delle configurazioni standard.
        
        sector_size = num_ranges // 3
        
        if start_angle == 'left': # 1/3 dei dati (es. indici da 0 a sector_size)
            sector_ranges = ranges[:sector_size]
        elif start_angle == 'center': # 1/3 dei dati (es. indici da sector_size a 2*sector_size)
            sector_ranges = ranges[sector_size:2 * sector_size]
        elif start_angle == 'right': # 1/3 dei dati (es. indici da 2*sector_size a num_ranges)
            sector_ranges = ranges[2 * sector_size:]
        else:
            return float('inf') # Caso non previsto
            
        # Filtra i valori 'inf' (fuori portata o non validi) prima di calcolare il minimo
        filtered_ranges = [r for r in sector_ranges if r < msg.range_max and r > msg.range_min]
        
        if not filtered_ranges:
            return float('inf')
            
        return min(filtered_ranges)

    def lidar_callback(self, msg: LaserScan):
        # 1. Suddividi lo scan in 3 settori (Destra, Centro, Sinistra)
        # La logica esatta dipende dall'orientamento e dal range angolare del tuo LiDAR.
        # Qui useremo una suddivisione per indici, che è più generica (ma meno precisa se lo scan
        # non è allineato perfettamente).
        
        ranges = msg.ranges
        num_ranges = len(ranges)
        
        # Per i LiDAR a 360 gradi (es. Turtlebot, ecc.), le misure frontali sono divise tra la fine e l'inizio dell'array.
        # Per semplificare l'esempio e mantenerlo vicino al tuo originale, uso 3 settori sequenziali:
        
        # Zona 1: Centro-Destra (es. Indici 0 - 240)
        # Zona 2: Centro-Frontale (es. Indici 240 - 480)
        # Zona 3: Centro-Sinistra (es. Indici 480 - 720)
        
        # Nota: La configurazione corretta per la maggior parte dei LiDAR ROS a 360°
        # con 0 gradi in avanti è:
        # * Sinistra: Indici 0 -> N/4
        # * Centro: Indici (3N/8 -> N/2) + (N/2 -> 5N/8) -- Molto complicato da implementare semplicemente
        # * Destra: Indici 3N/4 -> N

        # Per un robot che guarda in avanti (0 gradi), supponiamo che i range siano ordinati 
        # con 0 al centro, -180 a destra e +180 a sinistra.
        
        # Assumiamo una suddivisione più intuitiva (che richiede di conoscere l'indice centrale):
        center_idx = num_ranges // 2
        sector_width_idx = num_ranges // 6 # 30 gradi circa per settore (se 360 gradi totali)

        # Settore Centrale:
        center_ranges = ranges[center_idx - sector_width_idx : center_idx + sector_width_idx]
        
        # Settore Destro (angoli negativi):
        right_ranges = ranges[center_idx - 2*sector_width_idx : center_idx - sector_width_idx]
        
        # Settore Sinistro (angoli positivi):
        left_ranges = ranges[center_idx + sector_width_idx : center_idx + 2*sector_width_idx]


        # Calcola le distanze minime nei settori, ignorando i valori 'inf'
        min_center = self._get_min_range(center_ranges, msg.range_max, msg.range_min)
        min_right = self._get_min_range(right_ranges, msg.range_max, msg.range_min)
        min_left = self._get_min_range(left_ranges, msg.range_max, msg.range_min)
        
        # 2. Logica di Decisone (FSM Semplice)
        twist_msg = Twist()
        
        # Priorità: Evita Ostacolo Frontale
        if min_center < self.safe_distance:
            # Ostacolo in centro
            twist_msg.linear.x = 0.0
            
            # Decidi dove girare: Gira dove c'è più spazio (o meno pericolo)
            if min_right > min_left:
                # Più spazio a destra: Gira a destra (angolare negativo)
                twist_msg.angular.z = -self.turn_speed
                self.get_logger().warn('Obstacle Frontal: Turning Right')
            else:
                # Più spazio a sinistra: Gira a sinistra (angolare positivo)
                twist_msg.angular.z = self.turn_speed
                self.get_logger().warn('Obstacle Frontal: Turning Left')

        # Controllo laterale mentre si procede
        elif min_right < self.safe_distance:
             # Ostacolo a destra: Sterza leggermente a sinistra
            twist_msg.linear.x = self.forward_speed / 2
            twist_msg.angular.z = self.turn_speed / 2
            self.get_logger().info('Obstacle Right: Steering Left')

        elif min_left < self.safe_distance:
            # Ostacolo a sinistra: Sterza leggermente a destra
            twist_msg.linear.x = self.forward_speed / 2
            twist_msg.angular.z = -self.turn_speed / 2
            self.get_logger().info('Obstacle Left: Steering Right')
            
        else:
            # Nessun ostacolo rilevante: Prosegui dritto
            twist_msg.linear.x = self.forward_speed
            twist_msg.angular.z = 0.0
            self.get_logger().info('Path Clear: Moving Forward')

        # 3. Pubblica il messaggio
        self.publisher.publish(twist_msg)

    def _get_min_range(self, ranges_slice, range_max, range_min):
        """Funzione helper per trovare il minimo ignorando i valori non validi."""
        # numpy.clip limita i valori a range_max; poi filtra i valori 'inf' originali
        filtered_ranges = [r for r in ranges_slice if r < range_max and r > range_min]
        return min(filtered_ranges) if filtered_ranges else float('inf')


def main(args=None):
    rclpy.init(args=args)
    node = ImprovedObjectAvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()