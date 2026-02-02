#!/usr/bin/env python3
"""
Libreria necessaria per unire una camera RGB e le inferenze YOLOv8
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import math
from std_msgs.msg import String 
import csv  
import time 
import os

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from visualization_msgs.msg import Marker, MarkerArray
from yolov8_msgs.msg import InferenceResult, Yolov8Inference

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        # Parametri della camera
        self.CAMERA_FRAME_ID = "base_link" 
        self.H_FOV = 120.8
        # Parametri di filtro e storage
        self.ema_alpha = 0.3
        self.min_distance = 1.2  
        self.boxes_found = []        
        self.box_points_history = {} 
        self.ema_history_depth = {}  

    
        # Questo crea il file dove salveremo i dati
        self.csv_filename = "marker.csv"
        self.log_file = open(self.csv_filename, mode='w', newline='')
        self.writer = csv.writer(self.log_file)
        
        # Time: Secondi passati dall'avvio
        # BoxID: Quale scatola è
        # Marker_X, Marker_Y: La posizione filtrata
        self.writer.writerow(["Time", "BoxID", "Marker_X", "Marker_Y"])
        # Struttura: tempo | ID scatola | x | y

        
        self.start_time = time.time()
        self.get_logger().info(f"📍 LOGGING ATTIVO: Sto registrando la posizione del marker in '{self.csv_filename}'")

        # Modello Segmentation
        self.model = YOLO("yolov8n-seg.pt") 
        
        # Ricevi depth
        self.br = CvBridge()
        self.latest_depth_img = None
        self.tf_buffer = Buffer()
        # Trasformi coordinate
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # base_link -> map

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Image, '/rgb', self.camera_callback, 10)
        self.create_subscription(Image, '/depth', self.depth_callback, qos_profile)
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "/box_markers", 10)
        self.box_info_pub = self.create_publisher(String, '/detected_box_info', 10)

        self.get_logger().info(f"✅ Nodo Trova la scatola aperta!")

    def depth_callback(self, data):
        try:
            self.latest_depth_img = self.br.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except Exception:
            pass

    def get_map_coordinates(self, u, v, depth, img_width):
        # Angolo orizzontale del pixe
        center_x = img_width / 2.0
        pixel_offset = center_x - u
        rad_per_pixel = math.radians(self.H_FOV) / img_width
        angle_x = pixel_offset * rad_per_pixel
        # Ricostruisci punto 3D nel frame camera
        x_local = depth 
        y_local = depth * math.tan(angle_x) 
        z_local = 0.0
        # Creazione PointStamped (punto tridimensionale nello spazio)
        point_local = PointStamped()
        point_local.header.frame_id = self.CAMERA_FRAME_ID
        point_local.header.stamp = self.get_clock().now().to_msg()
        point_local.point.x = float(x_local)
        point_local.point.y = float(y_local)
        point_local.point.z = float(z_local)

        try:
            transform = self.tf_buffer.lookup_transform("map", self.CAMERA_FRAME_ID, rclpy.time.Time())
            # Trasforma il punto nel frame "map"
            pt = do_transform_point(point_local, transform)
            return pt.point.x, pt.point.y
        except Exception:
            return None, None
    # Aggiornamento delle scatole. L'ho già vista oppure no?
    def update_storage_smart(self, new_x, new_y):
        found_idx = -1
        
        for i, (saved_x, saved_y) in enumerate(self.boxes_found):
            # Calcola la distanza euclidea
            dist = math.sqrt((new_x - saved_x)**2 + (new_y - saved_y)**2)
            # Se è abbastanza distante, nuova scatola
            if dist < self.min_distance:
                found_idx = i
                break
        
        is_new = False
        if found_idx == -1:
            self.boxes_found.append((new_x, new_y))
            found_idx = len(self.boxes_found) - 1
            # Inizializzi lo storico delle osservazioni di questa scatola
            self.box_points_history[found_idx] = [] 
            is_new = True

        
        # Stimare centro robusto
        self.box_points_history[found_idx].append([new_x, new_y])
        # Mantieni solo le ultime 500 osservazioni
        if len(self.box_points_history[found_idx]) > 500: 
            self.box_points_history[found_idx].pop(0)
        # Trasformi la lista in array numpy
        points = np.array(self.box_points_history[found_idx], dtype=np.float32)

        if len(points) >= 5:
            # Robutosto
            rect = cv2.minAreaRect(points)
            (raw_center_x, raw_center_y) = rect[0]
        else:
            # Fammi la media semplice
            raw_center_x = float(np.mean(points[:, 0]))
            raw_center_y = float(np.mean(points[:, 1]))

        prev_center_x, prev_center_y = self.boxes_found[found_idx]
        
        # EMA Smoothing
        if is_new:
            fx, fy = raw_center_x, raw_center_y
        else:
            fx = (self.ema_alpha * raw_center_x) + ((1.0 - self.ema_alpha) * prev_center_x)
            fy = (self.ema_alpha * raw_center_y) + ((1.0 - self.ema_alpha) * prev_center_y)
        
        self.boxes_found[found_idx] = (fx, fy)
        return found_idx, is_new, fx, fy

    def publish_markers(self):
        marker_array = MarkerArray()
        for i, (x, y) in enumerate(self.boxes_found):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "boxes_center"
            m.id = i
            m.type = Marker.SPHERE 
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.15
            m.scale.x = 1.0; m.scale.y = 1.0; m.scale.z = 1.0
            m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0 
            marker_array.markers.append(m)
            
            if i in self.box_points_history:
                p_marker = Marker()
                p_marker.header.frame_id = "map"
                p_marker.header.stamp = self.get_clock().now().to_msg()
                p_marker.ns = f"box_points_{i}"
                p_marker.id = i + 2000
                p_marker.type = Marker.POINTS
                p_marker.action = Marker.ADD
                p_marker.scale.x = 0.02
                p_marker.scale.y = 0.02
                p_marker.color.r = 0.0; p_marker.color.g = 0.5; p_marker.color.b = 1.0; p_marker.color.a = 0.6
                
                for pt in self.box_points_history[i]:
                    p = PointStamped().point
                    p.x = float(pt[0])
                    p.y = float(pt[1])
                    p.z = 0.1
                    p_marker.points.append(p)
                marker_array.markers.append(p_marker)
            
            t = Marker()
            t.header.frame_id = "map"
            t.header.stamp = self.get_clock().now().to_msg()
            t.ns = "boxes_text"
            t.id = i + 1000
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = x
            t.pose.position.y = y
            t.pose.position.z = 0.5
            t.text = f"BOX {i}"
            t.scale.z = 0.2
            t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0; t.color.a = 1.0
            marker_array.markers.append(t)
            
        self.marker_pub.publish(marker_array)

    def camera_callback(self, data):
        try:
            img = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except Exception:
            return
        # Parte della segmentazione di YOLO
        h, w = img.shape[:2]
        results = self.model.track(img, persist=True, verbose=False, conf=0.87, retina_masks=True)
        annotated = results[0].plot(labels=False, conf=False, boxes=False)
        
        yolov8_msg = Yolov8Inference()
        yolov8_msg.header = data.header
        update_rviz = False  # Pubblica marker su Rviz solo se troviamo nuove scatole

        if results[0].masks is not None:
            for box, mask in zip(results[0].boxes, results[0].masks):
                track_id = int(box.id.item()) if box.id is not None else -1
                dist_surf = 0.0

                if self.latest_depth_img is not None:
                    # Creazione maschera binaria
                    contour = mask.xy[0].astype(np.int32)
                    mask_img = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask_img, [contour], 255)
                    masked_depth = self.latest_depth_img[mask_img == 255]
                    # Filtro valori validi
                    valid = masked_depth[~np.isnan(masked_depth) & ~np.isinf(masked_depth)]
                    
                    if len(valid) > 0:
                        # Stima robusta distanza
                        raw_dist = float(np.median(valid))
                        raw_dist = np.clip(raw_dist, 0.2, 10.0)
                        
                        if track_id != -1:
                            # Filtro EMA sulla profondità
                            prev = self.ema_history_depth.get(track_id, raw_dist)
                            dist_surf = (self.ema_alpha * raw_dist) + ((1.0 - self.ema_alpha) * prev)
                            self.ema_history_depth[track_id] = dist_surf
                        else:
                            dist_surf = raw_dist
                # Calcolo i momenti della maschera
                M = cv2.moments(mask.xy[0])
                cx = int(M['m10']/M['m00']) if M['m00']!=0 else int((box.xyxy[0][0]+box.xyxy[0][2])//2)
                cy = int(M['m01']/M['m00']) if M['m00']!=0 else int((box.xyxy[0][1]+box.xyxy[0][3])//2)

                if dist_surf > 0.0:
                    # PROIEZIONE 2D -> MAP
                    mx, my = self.get_map_coordinates(cx, cy, dist_surf, w)
                    
                    if mx is not None:
                        # Qui otteniamo fx e fy -> LE COORDINATE DEL MARKER FILTRATE
                        bid, new, fx, fy = self.update_storage_smart(mx, my)
                        update_rviz = True
                        if new: self.get_logger().info(f"🆕 BOX {bid}")
                        
                        # Salvataggio su dei dati per il plot
                        current_time = time.time() - self.start_time
                        # Salviamo: Tempo, ID, Marker_X, Marker_Y
                        self.writer.writerow([f"{current_time:.3f}", bid, f"{fx:.4f}", f"{fy:.4f}"])

                        msg = String(); msg.data = f"{bid},{fx},{fy}"; self.box_info_pub.publish(msg)
                # Disegno bounding box e centroide
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.circle(annotated, (cx,cy), 5, (0,0,255), -1)

        if update_rviz: self.publish_markers()
        self.img_pub.publish(self.br.cv2_to_imgmsg(annotated, "bgr8"))
        self.yolov8_pub.publish(yolov8_msg)


    def destroy_node(self):
        self.log_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()