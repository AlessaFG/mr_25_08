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

# Import per sapere dove si trova il robot (TF)
from tf2_ros import Buffer, TransformListener

class BoxListener(Node):
    """
    Questo nodo serve SOLO ad ascoltare la telecamera e tenere aggiornata la lista.
    Gira in background.
    """
    def __init__(self):
        super().__init__('box_listener_node')
        
        self.available_boxes = {}
        
        # TF Buffer per sapere dove siamo
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.create_subscription(String, '/detected_box_info', self.info_callback, 10)
        self.get_logger().info("👂 Listener avviato in background...")

    def info_callback(self, msg):
        try:
            data = msg.data.split(',')
            box_id = int(data[0])
            x = float(data[1])
            y = float(data[2])
            
            # Aggiorna il dizionario (Thread-safe in Python per operazioni atomiche)
            if box_id not in self.available_boxes:
                # Usiamo print speciali per non rompere il prompt dell'input
                sys.stdout.write(f"\n✨ [NUOVA BOX] ID {box_id} trovata a ({x:.1f}, {y:.1f})\nComando > ")
                sys.stdout.flush()
            
            self.available_boxes[box_id] = (x, y)
        except Exception:
            pass

    def get_robot_pose(self):
        """ Ottiene la posizione X, Y attuale del robot """
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return trans.transform.translation.x, trans.transform.translation.y
        except Exception:
            return None, None

def main():
    rclpy.init()

    # 1. Avvia il Listener in un Thread Separato
    listener_node = BoxListener()
    executor = MultiThreadedExecutor()
    executor.add_node(listener_node)

    # Thread che gestisce SOLO l'ascolto dei topic
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 2. Inizializza il Navigatore nel Main Thread
    navigator = BasicNavigator()
    
    # Aspetta che Nav2 sia attivo (importante!)
    # print("⏳ Attesa attivazione Nav2...")
    # navigator.waitUntilNav2Active()
    # print("✅ Nav2 Pronto!")

    print("\n-------------------------------------------------")
    print("  SISTEMA DI NAVIGAZIONE MANUALE BOX (V2)  ")
    print("  1. Il robot esplora o tu lo muovi.")
    print("  2. Appena appare una box, scrivi l'ID.")
    print("  3. Il robot ci va, poi ti chiede il prossimo.")
    print("-------------------------------------------------\n")

    # 3. Loop Principale (Input + Navigazione)
  # 3. Loop Principale (Input + Navigazione)
    while rclpy.ok():
        try:
            user_input = input("Comando (Inserisci ID Box): ")

            if not user_input.strip().isdigit():
                print("⚠️ Inserisci un numero intero.")
                continue

            target_id = int(user_input)

            # Controlliamo se la box esiste nella memoria del Listener
            if target_id not in listener_node.available_boxes:
                print(f"❌ Errore: La Box {target_id} non è ancora stata vista!")
                continue

            # --- CALCOLO PERCORSO ---
            box_x, box_y = listener_node.available_boxes[target_id]
            robot_x, robot_y = listener_node.get_robot_pose()

            if robot_x is None:
                print("⚠️ Errore TF: Non so dove sia il robot.")
                continue

            # Calcolo punto di approccio (50cm prima)
            dx = box_x - robot_x
            dy = box_y - robot_y
            angolo = math.atan2(dy, dx)
            stop_distance = 1.20 

            target_x = box_x - (stop_distance * math.cos(angolo))
            target_y = box_y - (stop_distance * math.sin(angolo))

            # Creazione Goal
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = navigator.get_clock().now().to_msg()
            goal_pose.pose.position.x = target_x
            goal_pose.pose.position.y = target_y
            goal_pose.pose.orientation.z = math.sin(angolo / 2.0)
            goal_pose.pose.orientation.w = math.cos(angolo / 2.0)

            print(f" Partenza verso BOX {target_id}...")

            # --- AZIONE DI NAVIGAZIONE ---
            navigator.goToPose(goal_pose)

            # Loop di attesa con controllo distanza ferma
            last_distance = None
            stuck_start_time = None

            while not navigator.isTaskComplete():
                feedback = navigator.getFeedback()
                if feedback:
                    dist_rimanente = feedback.distance_remaining
                    print(f"   In viaggio... {dist_rimanente:.2f}m mancanti", end="\r")

                    # Controllo se la distanza è ferma
                    if last_distance is not None:
                        if abs(dist_rimanente - last_distance) < 0.01:  # distanza praticamente invariata
                            if stuck_start_time is None:
                                stuck_start_time = time.time()
                            elif time.time() - stuck_start_time >= 8:  # 8 secondi fermi
                                print(f"\n⚠️ Distanza ferma da 8s. Considero il robot arrivato.")
                                navigator.cancelTask()  # Interrompe Nav2
                                break
                        else:
                            stuck_start_time = None  # reset se la distanza cambia
                    last_distance = dist_rimanente

                time.sleep(0.5)

            # Risultato finale
            result = navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                print(f"\n✅ ARRIVATO alla Box {target_id}! Pronto per il prossimo ordine.")
            elif result == TaskResult.CANCELED:
                print(f"\n⚠️ Navigazione terminata (arrivo forzato o interruzione).")
            elif result == TaskResult.FAILED:
                print(f"\n❌ Navigazione fallita (ostacolo?).")

        except KeyboardInterrupt:
            print("\nSpegnimento...")
            break
        except Exception as e:
            print(f"Errore generico: {e}")


    # Pulizia
    navigator.lifecycleShutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()