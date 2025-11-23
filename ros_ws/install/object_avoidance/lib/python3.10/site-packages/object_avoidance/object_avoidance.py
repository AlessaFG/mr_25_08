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

        # Parametri 
        self.safe_distance = 0.5  # Distanza di sicurezza 
        self.forward_speed = 0.2  # Velocità lineare quando si procede
        self.turn_speed = 0.5     # Velocità angolare di rotazione

        self.get_logger().info('Improved Object Avoidance Node Started')
    
    def get_sector_min(self, ranges, start_angle, end_angle, angle_increment):
        """Calcola la distanza minima in un settore dello scan."""
        num_ranges = len(ranges)
        
 
        
        sector_size = num_ranges // 3
        
        if start_angle == 'left': # 1/3 dei dati 
            sector_ranges = ranges[:sector_size]
        elif start_angle == 'center': # 1/3 dei dati 
            sector_ranges = ranges[sector_size:2 * sector_size]
        elif start_angle == 'right': # 1/3 dei dati 
            sector_ranges = ranges[2 * sector_size:]
        else:
            return float('inf') 
            
        filtered_ranges = [r for r in sector_ranges if r < msg.range_max and r > msg.range_min]
        
        if not filtered_ranges:
            return float('inf')
            
        return min(filtered_ranges)

    def lidar_callback(self, msg: LaserScan):
        # 1. Suddivido lo scan in 3 settori (Destra, Centro, Sinistra)
    
        
        ranges = msg.ranges
        num_ranges = len(ranges)
        
        
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
        
        self.get_logger().info(f'Distanze minime: Centro={min_center:.2f}m, Destra={min_right:.2f}m, Sinistra={min_left:.2f}m')
     
        twist_msg = Twist()
        
        # Priorità: Evita Ostacolo Frontale
        if min_center < self.safe_distance:
            # Ostacolo in centro
            twist_msg.linear.x = 0.0
            
            # Decidi dove girare: Gira dove c'è più spazio (o meno pericolo)
            if min_right > min_left:
                # Più spazio a destra: Gira a destra (angolare negativo)
                twist_msg.angular.z = -self.turn_speed
                self.get_logger().warn('🚨 OST. FRONTALE: Gira a DESTRA')
            else:
                # Più spazio a sinistra: Gira a sinistra (angolare positivo)
                twist_msg.angular.z = self.turn_speed
                self.get_logger().warn('🚨 OST. FRONTALE: Gira a SINISTRA')

        # Controllo laterale mentre si procede
        elif min_right < self.safe_distance:
             # Ostacolo a destra: Sterza leggermente a sinistra
            twist_msg.linear.x = self.forward_speed / 2
            twist_msg.angular.z = self.turn_speed / 2
            self.get_logger().info('⚠️ Ost. DESTRA: Sterza a sinistra')

        elif min_left < self.safe_distance:
            # Ostacolo a sinistra: Sterza leggermente a destra
            twist_msg.linear.x = self.forward_speed / 2
            twist_msg.angular.z = -self.turn_speed / 2
            self.get_logger().info('⚠️ Ost. SINISTRA: Sterza a destra')
            
        else:
            # Nessun ostacolo rilevante: Prosegui dritto
            twist_msg.linear.x = self.forward_speed
            twist_msg.angular.z = 0.0
            self.get_logger().info('✅ Percorso Libero: Avanti')

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