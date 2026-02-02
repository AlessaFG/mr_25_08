#!/usr/bin/env python3

"""
Librerie necessarie
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class ImprovedObjectAvoidanceNode(Node):

    def __init__(self):
        super().__init__('improved_object_avoidance_node')

        """
        Sottoscrizione al topic /scan
        """
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10
        )

        """
        Publisher per il topic /cmd_vel
        """
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Parametri 
        self.safe_distance = 0.5  # Distanza di sicurezza 
        self.forward_speed = 0.22  # Velocità lineare quando si procede
        self.turn_speed = 0.5     # Velocità angolare di rotazione

        self.get_logger().info('Improved Object Avoidance Node Started')

    def lidar_callback(self, msg: LaserScan):
        # Suddivido lo scan in 3 settori (Destra, Centro, Sinistra)
    
        ranges = msg.ranges
        num_ranges = len(ranges)
        
    
        center_idx = num_ranges // 2
        sector_width_idx = num_ranges // 6 # 30 gradi circa per settore

        center_ranges = ranges[center_idx - sector_width_idx : center_idx + sector_width_idx]
        right_ranges = ranges[center_idx - 2*sector_width_idx : center_idx - sector_width_idx]
        left_ranges = ranges[center_idx + sector_width_idx : center_idx + 2*sector_width_idx]


        # Calcola le distanze minime nei settori ignorando i valori 'inf'
        min_center = self._get_min_range(center_ranges, msg.range_max, msg.range_min)
        min_right = self._get_min_range(right_ranges, msg.range_max, msg.range_min)
        min_left = self._get_min_range(left_ranges, msg.range_max, msg.range_min)
        
        self.get_logger().info(f'Distanze minime: Centro={min_center:.2f}m, Destra={min_right:.2f}m, Sinistra={min_left:.2f}m')
     
        twist_msg = Twist()
        
        # Evita Ostacolo Frontale
        if min_center < self.safe_distance:

            twist_msg.linear.x = 0.0
            
            # Gira dove c'è più spazio
            if min_right > min_left:
                # Più spazio a destra -> Gira a destra
                twist_msg.angular.z = -self.turn_speed
                self.get_logger().warn('🚨 OST. FRONTALE: Gira a DESTRA')
            else:
                # Più spazio a sinistra -> Gira a sinistra
                twist_msg.angular.z = self.turn_speed
                self.get_logger().warn('🚨 OST. FRONTALE: Gira a SINISTRA')

        # Controllo laterale
        elif min_right < self.safe_distance:
             # Ostacolo a destra -> Sterza leggermente a sinistra
            twist_msg.linear.x = self.forward_speed / 2
            twist_msg.angular.z = self.turn_speed / 2
            self.get_logger().info('⚠️ Ost. DESTRA: Sterza a sinistra')

        elif min_left < self.safe_distance:
            # Ostacolo a sinistra -> Sterza leggermente a destra
            twist_msg.linear.x = self.forward_speed / 2
            twist_msg.angular.z = -self.turn_speed / 2
            self.get_logger().info('⚠️ Ost. SINISTRA: Sterza a destra')
            
        else:
            # Nessun ostacolo rilevante -> Prosegui dritto
            twist_msg.linear.x = self.forward_speed
            twist_msg.angular.z = 0.0
            self.get_logger().info('✅ Percorso Libero: Avanti')

        self.publisher.publish(twist_msg)

    def _get_min_range(self, ranges_slice, range_max, range_min):
        """Funzione per trovare il minimo ignorando i valori non validi."""
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