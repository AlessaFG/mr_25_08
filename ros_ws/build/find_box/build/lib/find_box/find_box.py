#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import threading
import sys
import math
import time
import csv  # <--- Per il log
import os

from tf2_ros import Buffer, TransformListener

class BoxListener(Node):
    def __init__(self):
        super().__init__('box_listener_node')
        self.available_boxes = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(String, '/detected_box_info', self.info_callback, 10)

    def info_callback(self, msg):
        try:
            data = msg.data.split(',')
            box_id = int(data[0]); x = float(data[1]); y = float(data[2])
            if box_id not in self.available_boxes:
                sys.stdout.write(f"\n✨ [NUOVA BOX] ID {box_id} trovata a ({x:.1f}, {y:.1f})\nComando > ")
                sys.stdout.flush()
            self.available_boxes[box_id] = (x, y)
        except Exception: pass

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return trans.transform.translation.x, trans.transform.translation.y
        except Exception: return None, None

def main():
    rclpy.init()
    
    # --- SETUP LOGGER ---
    filename = "nav_log.csv"
    # Modalità 'a' (append) così se fai più viaggi li salva tutti, 
    # ma scriviamo l'header solo se il file è nuovo
    file_exists = os.path.isfile(filename)
    log_file = open(filename, mode='a', newline='')
    writer = csv.writer(log_file)
    
    if not file_exists:
        # Time: Tempo trascorso dall'inizio del viaggio
        # Distance_Remaining: Quanto manca al traguardo (Goal)
        writer.writerow(["Time", "Distance_Remaining"])
    
    print(f"📝 Logging attivo su: {filename}")
    # --------------------

    listener_node = BoxListener()
    executor = MultiThreadedExecutor()
    executor.add_node(listener_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    navigator = BasicNavigator()
    
    print("\n--- NAVIGAZIONE MANUALE ---")

    while rclpy.ok():
        try:
            user_input = input("Comando (Inserisci ID Box): ")
            if not user_input.strip().isdigit(): continue
            target_id = int(user_input)

            if target_id not in listener_node.available_boxes:
                print(f"❌ Box {target_id} non trovata!")
                continue

            box_x, box_y = listener_node.available_boxes[target_id]
            robot_x, robot_y = listener_node.get_robot_pose()
            if robot_x is None: continue

            # Punto di arrivo (1.2m prima)
            stop_distance = 1.20 
            dx = box_x - robot_x; dy = box_y - robot_y
            angolo = math.atan2(dy, dx)
            target_x = box_x - (stop_distance * math.cos(angolo))
            target_y = box_y - (stop_distance * math.sin(angolo))

            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = navigator.get_clock().now().to_msg()
            goal_pose.pose.position.x = target_x; goal_pose.pose.position.y = target_y
            goal_pose.pose.orientation.z = math.sin(angolo / 2.0)
            goal_pose.pose.orientation.w = math.cos(angolo / 2.0)

            print(f"🚀 Partenza verso BOX {target_id}...")
            navigator.goToPose(goal_pose)

            # --- PREPARAZIONE TIMER LOG ---
            start_trip_time = time.time()
            
            # --- LOOP ATTESA ---
            while not navigator.isTaskComplete():
                feedback = navigator.getFeedback()
                if feedback:
                    dist_rimanente = feedback.distance_remaining
                    
                    # --- LOGGING ---
                    # Tempo relativo a QUESTO viaggio
                    t_now = time.time() - start_trip_time
                    writer.writerow([f"{t_now:.3f}", f"{dist_rimanente:.4f}"])
                    # ---------------

                    print(f"   Mancano: {dist_rimanente:.2f}m", end="\r")

                time.sleep(0.1)

            result = navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                print(f"\n✅ ARRIVATO!")
            else:
                print(f"\n⚠️ Stop/Fallito.")

        except KeyboardInterrupt:
            break
        except Exception:
            pass

    log_file.close()
    navigator.lifecycleShutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()