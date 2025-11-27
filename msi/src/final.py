#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_msgs.msg import String

from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator, TurtleBot4Directions

import numpy as np
import cv2
import threading
import math
from ultralytics import YOLO
from message_filters import Subscriber, ApproximateTimeSynchronizer


class ObjectApproachNode(Node):
    def __init__(self):
        super().__init__('object_approach_node')
        
        # ================================
        # 설정 파라미터
        # ================================
        self.TARGET_OBJECT = 'car'  # 탐지할 목표 객체 (YOLO 클래스 이름)
        self.TARGET_DISTANCE = 0.05  # 목표 거리 (미터)
        self.DISTANCE_THRESHOLD = 0.05  # 거리 허용 오차
        self.MIN_CONFIDENCE = 0.5  # 최소 신뢰도
        self.MODEL_PATH = '/home/rokey/rokey_ws/my_best.pt'
        
        # ================================
        # 초기화
        # ================================
        self.bridge = CvBridge()
        self.K = None  # 카메라 내부 파라미터
        self.lock = threading.Lock()
        
        # 네임스페이스 설정
        ns = self.get_namespace()
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'
        self.rgb_topic = f'{ns}/oakd/rgb/image_raw/compressed'
        self.info_topic = f'{ns}/oakd/rgb/camera_info'
        
        # 상태 변수
        self.rgb_image = None
        self.depth_image = None
        self.camera_frame = None
        self.target_detected = False
        self.approaching = False
        self.navigation_in_progress = False
        
        # YOLO 모델 로드
        self.get_logger().info(f'Loading YOLO model from {self.MODEL_PATH}...')
        self.model = YOLO(self.MODEL_PATH)
        self.get_logger().info('YOLO model loaded successfully.')
        
        # TF2 버퍼 초기화
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # TurtleBot4 Navigator 초기화
        self.navigator = TurtleBot4Navigator()
        
        # 초기 포즈 설정
        if not self.navigator.getDockedStatus():
            self.get_logger().info('Docking before initializing pose')
            self.navigator.dock()
        
        initial_pose = self.navigator.getPoseStamped([-2.441, -0.8078], TurtleBot4Directions.NORTH)
        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()
        self.navigator.undock()
        
        # 카메라 정보 구독
        self.create_subscription(CameraInfo, self.info_topic, self.camera_info_callback, 1)
        
        # 시간 동기화된 RGB + Depth 구독자
        self.rgb_sub = Subscriber(self, CompressedImage, self.rgb_topic)
        self.depth_sub = Subscriber(self, CompressedImage, self.depth_topic)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)
        
        # 결과 발행자
        self.result_publisher = self.create_publisher(String, '/object_detection_result', 10)
        self.annotated_image_publisher = self.create_publisher(CompressedImage, '/annotated_image_with_depth', 10)
        
        # TF 안정화 대기
        self.get_logger().info("Waiting for TF tree to stabilize (5 seconds)...")
        self.start_timer = self.create_timer(5.0, self.start_detection)
        
    def start_detection(self):
        self.get_logger().info("TF tree stabilized. Starting object detection and navigation.")
        self.detection_timer = self.create_timer(0.2, self.process_detection)
        self.start_timer.cancel()
    
    def camera_info_callback(self, msg):
        """카메라 내부 파라미터 저장"""
        with self.lock:
            if self.K is None:
                self.K = np.array(msg.k).reshape(3, 3)
                self.get_logger().info(
                    f"Camera intrinsics: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, "
                    f"cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
                )
    
    def synced_callback(self, rgb_msg, depth_msg):
        """시간 동기화된 RGB와 Depth 이미지 콜백"""
        try:
            with self.lock:
                # RGB 이미지 변환
                rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
                if rgb is not None and rgb.size > 0:
                    self.rgb_image = rgb
                
                # Depth 이미지 변환
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                if depth is not None and depth.size > 0:
                    self.depth_image = depth
                    self.camera_frame = depth_msg.header.frame_id
                    
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
    
    def process_detection(self):
        """객체 탐지 및 접근 처리"""
        with self.lock:
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None
            depth = self.depth_image.copy() if self.depth_image is not None else None
            frame_id = self.camera_frame
            K = self.K
        
        if rgb is None or depth is None or K is None or frame_id is None:
            return
        
        if self.navigation_in_progress:
            return
        
        try:
            # YOLO 추론
            results = self.model.predict(rgb, verbose=False, conf=self.MIN_CONFIDENCE)
            
            annotated_image = rgb.copy()
            detection_summary = []
            target_found = False
            closest_target = None
            closest_distance = float('inf')
            
            # 결과 처리
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]
                    conf = round(box.conf[0].item(), 2)
                    cls_id = int(box.cls[0].item())
                    cls_name = self.model.names[cls_id]
                    
                    # 바운딩 박스 중심 계산
                    xc = int((x1 + x2) / 2)
                    yc = int((y1 + y2) / 2)
                    
                    # Depth 값 추출
                    distance_m = np.nan
                    if 0 <= yc < depth.shape[0] and 0 <= xc < depth.shape[1]:
                        depth_patch = depth[max(0, yc-1):min(depth.shape[0], yc+2), 
                                           max(0, xc-1):min(depth.shape[1], xc+2)]
                        valid_depths = depth_patch[depth_patch > 0]
                        
                        if valid_depths.size > 0:
                            distance_mm = np.mean(valid_depths)
                            distance_m = distance_mm / 1000.0
                    
                    # 시각화
                    color = (0, 255, 0) if cls_name != self.TARGET_OBJECT else (0, 0, 255)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_image, f'{cls_name} {conf}', (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(annotated_image, (xc, yc), 4, (0, 0, 255), -1)
                    
                    if not np.isnan(distance_m):
                        distance_text = f'{distance_m:.2f}m'
                        cv2.putText(annotated_image, distance_text, (x1, y2 + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        detection_summary.append(f'{cls_name} ({conf}, {distance_text})')
                        
                        # 목표 객체 확인
                        if cls_name == self.TARGET_OBJECT and distance_m < closest_distance:
                            target_found = True
                            closest_distance = distance_m
                            closest_target = {
                                'xc': xc,
                                'yc': yc,
                                'distance': distance_m,
                                'conf': conf
                            }
                    else:
                        cv2.putText(annotated_image, 'No Depth', (x1, y2 + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        detection_summary.append(f'{cls_name} ({conf}, No Depth)')
            
            # Annotated 이미지 발행
            try:
                img_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                self.annotated_image_publisher.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish annotated image: {e}')
            
            # 결과 발행
            result_msg = String()
            result_msg.data = " | ".join(detection_summary) if detection_summary else "No objects detected"
            self.result_publisher.publish(result_msg)
            
            # 목표 객체 접근 처리
            if target_found and closest_target:
                current_dist = closest_target['distance']
                
                if abs(current_dist - self.TARGET_DISTANCE) > self.DISTANCE_THRESHOLD:
                    self.get_logger().info(
                        f"Target '{self.TARGET_OBJECT}' detected at {current_dist:.2f}m "
                        f"(conf: {closest_target['conf']:.2f}). Approaching..."
                    )
                    self.approach_target(closest_target, frame_id, K, depth)
                else:
                    self.get_logger().info(
                        f"Target '{self.TARGET_OBJECT}' within target distance "
                        f"({current_dist:.2f}m ≈ {self.TARGET_DISTANCE}m). Holding position."
                    )
                    
        except Exception as e:
            self.get_logger().error(f"Detection processing error: {e}")
    
    def approach_target(self, target_info, frame_id, K, depth):
        """목표 객체에 접근"""
        try:
            xc, yc = target_info['xc'], target_info['yc']
            z = target_info['distance']
            
            if z <= 0.2 or z >= 5.0:
                self.get_logger().warn(f"Invalid depth value: {z:.2f}m")
                return
            
            # 카메라 좌표계에서 3D 포인트 계산
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            X = (xc - cx) * z / fx
            Y = (yc - cy) * z / fy
            Z = z
            
            # PointStamped 생성
            pt_camera = PointStamped()
            pt_camera.header.stamp = Time().to_msg()
            pt_camera.header.frame_id = frame_id
            pt_camera.point.x = X
            pt_camera.point.y = Y
            pt_camera.point.z = Z
            
            # 맵 좌표계로 변환
            pt_map = self.tf_buffer.transform(pt_camera, 'map', timeout=Duration(seconds=1.0))
            
            self.get_logger().info(
                f"Target map coordinate: ({pt_map.point.x:.2f}, {pt_map.point.y:.2f}, {pt_map.point.z:.2f})"
            )
            
            # 목표 거리만큼 떨어진 위치 계산
            current_distance = z
            approach_distance = current_distance - self.TARGET_DISTANCE
            
            # 목표 포즈 생성
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            
            # 로봇에서 목표까지의 방향 벡터 계산
            dx = pt_map.point.x
            dy = pt_map.point.y
            distance_xy = math.sqrt(dx**2 + dy**2)
            
            if distance_xy > 0:
                # 목표 거리만큼 떨어진 지점 계산
                ratio = approach_distance / distance_xy
                goal_pose.pose.position.x = dx * ratio
                goal_pose.pose.position.y = dy * ratio
                goal_pose.pose.position.z = 0.0
                
                # 목표를 향한 방향 계산
                yaw = math.atan2(dy, dx)
                qz = math.sin(yaw / 2.0)
                qw = math.cos(yaw / 2.0)
                goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
                
                # 네비게이션 시작
                self.navigation_in_progress = True
                self.navigator.goToPose(goal_pose)
                self.get_logger().info(f"Navigation goal sent to ({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})")
                
                # 네비게이션 완료 대기 (비동기)
                self.create_timer(1.0, self.check_navigation_status)
                
        except Exception as e:
            self.get_logger().error(f"Approach target failed: {e}")
            self.navigation_in_progress = False
    
    def check_navigation_status(self):
        """네비게이션 상태 확인"""
        if not self.navigator.isTaskComplete():
            return
        
        self.navigation_in_progress = False
        self.get_logger().info("Navigation completed.")
    
    def destroy_node(self):
        """노드 종료 시 도킹"""
        self.get_logger().info("Shutting down. Docking robot...")
        self.navigator.dock()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectApproachNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()