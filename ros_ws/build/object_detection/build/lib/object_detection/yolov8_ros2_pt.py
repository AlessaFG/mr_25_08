#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import math
from std_msgs.msg import String 

# TF e Geometria
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from visualization_msgs.msg import Marker, MarkerArray

# Messaggi Custom
from yolov8_msgs.msg import InferenceResult
from yolov8_msgs.msg import Yolov8Inference

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.CAMERA_FRAME_ID = "base_link" 
        self.H_FOV = 120.8 
        self.ema_alpha = 0.15
        self.ema_history = {} 

        self.boxes_found = []    
        self.min_distance = 1.6  

        self.model = YOLO("best_87_.pt")
        self.br = CvBridge()
        self.latest_depth_img = None
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(Image, '/rgb', self.camera_callback, 10)
        self.create_subscription(Image, '/depth', self.depth_callback, qos_profile)
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "/box_markers", 10)
        
        # Pubblica stringhe tipo "0,2.5,1.2" (ID, X, Y)
        self.box_info_pub = self.create_publisher(String, '/detected_box_info', 10)

        self.get_logger().info(f"Nodo pronto. Informazioni box su '/detected_box_info'")

    def depth_callback(self, data):
        try:
            self.latest_depth_img = self.br.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except Exception:
            pass

    def get_map_coordinates(self, u, v, depth, img_width):
        center_x = img_width / 2.0
        pixel_offset = center_x - u
        rad_per_pixel = math.radians(self.H_FOV) / img_width
        angle_x = pixel_offset * rad_per_pixel
        
        x_local = depth # Avanti
        y_local = depth * math.tan(angle_x) # Lato
        z_local = 0.0

        point_local = PointStamped()
        point_local.header.frame_id = self.CAMERA_FRAME_ID
        point_local.header.stamp = self.get_clock().now().to_msg()
        point_local.point.x = float(x_local)
        point_local.point.y = float(y_local)
        point_local.point.z = float(z_local)

        try:
            transform = self.tf_buffer.lookup_transform("map", self.CAMERA_FRAME_ID, rclpy.time.Time())
            point_map = do_transform_point(point_local, transform)
            return point_map.point.x, point_map.point.y
        except Exception:
            return None, None

    def update_storage(self, new_x, new_y):
        """
        Ritorna l'INDICE della scatola (ID) e True/False se è nuova
        """
        for i, (saved_x, saved_y) in enumerate(self.boxes_found):
            dist = math.sqrt((new_x - saved_x)**2 + (new_y - saved_y)**2)
            if dist < self.min_distance:
                self.boxes_found[i] = (new_x, new_y)
                return i, False  # Ritorna ID esistente e False
        
        self.boxes_found.append((new_x, new_y))
        return len(self.boxes_found) - 1, True # Ritorna nuovo ID e True

    def publish_markers(self):
        marker_array = MarkerArray()
        for i, (x, y) in enumerate(self.boxes_found):
            point_marker = Marker()
            point_marker.header.frame_id = "map"
            point_marker.header.stamp = self.get_clock().now().to_msg()
            point_marker.ns = "boxes"
            point_marker.id = i
            
            # --- MODIFICA QUI: USA SPHERE (SFERA/PUNTO) INVECE DI CUBE ---
            point_marker.type = Marker.SPHERE 
            point_marker.action = Marker.ADD
            
            point_marker.pose.position.x = x
            point_marker.pose.position.y = y
            point_marker.pose.position.z = 0.1 # Basso, vicino a terra
            
            point_marker.pose.orientation.w = 1.0
            
            # --- DIMENSIONI PICCOLE (per sembrare un punto) ---
            point_marker.scale.x = 0.2
            point_marker.scale.y = 0.2
            point_marker.scale.z = 0.2
            
            # Colore (Arancione)
            point_marker.color.r = 1.0
            point_marker.color.g = 0.5 
            point_marker.color.b = 0.0
            point_marker.color.a = 0.5

            marker_array.markers.append(point_marker)
        
        self.marker_pub.publish(marker_array)

    def camera_callback(self, data):
        try:
            img = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except Exception:
            return

        h_img, w_img, _ = img.shape
        results = self.model.track(img, persist=True, verbose=False, conf=0.915)
        annotated_frame = results[0].plot(labels=False, conf=False)
        
        yolov8_msg = Yolov8Inference()
        yolov8_msg.header = data.header
        
        update_rviz = False

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 
                class_name = self.model.names[int(box.cls)]
                track_id = int(box.id.item()) if box.id is not None else -1

                distanza_finale = 0.0
                if self.latest_depth_img is not None:
                    if 0 <= cx < w_img and 0 <= cy < h_img:
                        raw_dist = self.latest_depth_img[cy, cx]
                        if not np.isnan(raw_dist) and not np.isinf(raw_dist):
                            raw_dist = float(raw_dist)
                            # clamp per evitare spike
                            raw_dist = np.clip(raw_dist, 0.2, 10.0)

                            if track_id != -1:
                                if track_id in self.ema_history:
                                    prev = self.ema_history[track_id]
                                    smoothed = (self.ema_alpha * raw_dist) + ((1.0 - self.ema_alpha) * prev)
                                else:
                                    smoothed = raw_dist
                                self.ema_history[track_id] = smoothed
                                distanza_finale = smoothed
                            else:
                                distanza_finale = raw_dist

                coord_text = ""
                box_color = (0, 255, 0)
                
                if distanza_finale > 0.0:
                    map_x, map_y = self.get_map_coordinates(cx, cy, distanza_finale, w_img)
                    
                    if map_x is not None:
                        coord_text = f"Map: [{map_x:.1f}, {map_y:.1f}]"
                        
                        box_id, is_new = self.update_storage(map_x, map_y)
                        
                        update_rviz = True

                        if is_new:
                            self.get_logger().info(f"🆕 NUOVA BOX ID {box_id}!")
                            box_color = (0, 0, 255) 
                        
                        msg_str = String()
                        msg_str.data = f"{box_id},{map_x},{map_y}"
                        self.box_info_pub.publish(msg_str)

                label = f"{class_name} {distanza_finale:.2f}m"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                if coord_text:
                    cv2.putText(annotated_frame, f"TOT: {len(self.boxes_found)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    cv2.putText(annotated_frame, coord_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                inf_res = InferenceResult()
                inf_res.class_name = class_name
                yolov8_msg.yolov8_inference.append(inf_res)

        if update_rviz:
            self.publish_markers()

        self.img_pub.publish(self.br.cv2_to_imgmsg(annotated_frame, "bgr8"))
        self.yolov8_pub.publish(yolov8_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()