#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo
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


class CarFollowingNode(Node):
    def __init__(self):
        super().__init__('car_following_node')
        
        # ================================
        # 설정 파라미터
        # ================================
        self.TARGET_OBJECT = 'car'  # 탐지할 목표 객체
        self.MIN_CONFIDENCE = 0.6  # 최소 신뢰도
        self.MODEL_PATH = '/home/rokey/rokey_ws/my_best.pt'
        self.APPROACH_DISTANCE = 1.5  # 차량에 접근할 거리 (미터)
        
        # 목표 위치 설정 (원하는 위치로 수정)
        self.DESIRED_POSITION = [-3.58, -1.16]  # [x, y]
        self.DESIRED_DIRECTION = TurtleBot4Directions.NORTH
        
        # ================================
        # 초기화
        # ================================
        self.bridge = CvBridge()
        self.K = None  # 카메라 내부 파라미터
        self.lock = threading.Lock()
        
        # 네임스페이스 설정
        ns = self.get_namespace()
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'
        self.rgb_topic = f'{ns}/oakd/rgb/image_raw'
        self.info_topic = f'{ns}/oakd/rgb/camera_info'
        
        # 상태 변수
        self.rgb_image = None
        self.depth_image = None
        self.camera_frame = None
        self.navigation_complete = False
        self.detection_active = False
        self.car_detected = False
        self.navigation_in_progress = False
        
        # 프로젝트 단계
        self.stage = "INITIALIZING"  # INITIALIZING -> NAVIGATING -> DETECTING -> APPROACHING
        
        # YOLO 모델 로드
        self.get_logger().info(f'[STAGE: {self.stage}] Loading YOLO model...')
        self.model = YOLO(self.MODEL_PATH)
        self.get_logger().info('YOLO model loaded successfully.')
        
        # TF2 버퍼 초기화
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # TurtleBot4 Navigator 초기화
        self.navigator = TurtleBot4Navigator()
        
        # 1단계: Undock & 초기 위치로 이동
        self.initialize_robot()
        
    def initialize_robot(self):
        """1단계: 로봇 초기화 및 목표 위치로 이동"""
        self.get_logger().info('[STAGE 1] Initializing robot...')
        
        # 도킹되어 있으면 언도킹
        if self.navigator.getDockedStatus():
            self.get_logger().info('Robot is docked. Undocking...')
            self.navigator.undock()
        
        # 초기 포즈 설정
        initial_pose = self.navigator.getPoseStamped([-2.441, -0.8078], TurtleBot4Directions.NORTH)
        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()
        
        self.get_logger().info(f'[STAGE 1] Navigating to desired position: {self.DESIRED_POSITION}')
        self.stage = "NAVIGATING"
        
        # 목표 위치로 이동
        goal_pose = self.navigator.getPoseStamped(self.DESIRED_POSITION, self.DESIRED_DIRECTION)
        self.navigator.goToPose(goal_pose)
        
        # 네비게이션 완료 대기 타이머
        self.nav_check_timer = self.create_timer(1.0, self.check_initial_navigation)
        
    def check_initial_navigation(self):
        """초기 네비게이션 완료 확인"""
        if self.navigator.isTaskComplete():
            self.nav_check_timer.cancel()
            self.navigation_complete = True
            self.get_logger().info('[STAGE 1] Arrived at desired position!')
            self.get_logger().info('[STAGE 2] Starting car detection...')
            self.stage = "DETECTING"
            
            # 카메라 및 탐지 시스템 시작
            self.setup_camera_system()
    
    def setup_camera_system(self):
        """2단계: 카메라 및 탐지 시스템 설정"""
        # 카메라 정보 구독
        self.create_subscription(CameraInfo, self.info_topic, self.camera_info_callback, 1)
        
        # 시간 동기화된 RGB + Depth 구독자
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)
        
        # 결과 발행자
        self.result_publisher = self.create_publisher(String, '/car_detection_result', 10)
        self.annotated_image_publisher = self.create_publisher(Image, '/annotated_image', 10)
        
        # TF 안정화 대기 후 탐지 시작
        self.get_logger().info("Waiting for TF tree to stabilize (3 seconds)...")
        self.start_timer = self.create_timer(3.0, self.start_detection)
        
    def start_detection(self):
        """탐지 시작"""
        self.get_logger().info("[STAGE 2] Detection system active. Looking for cars...")
        self.detection_active = True
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
        if not self.detection_active:
            return
            
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
        """3단계: 차량 탐지 및 접근"""
        if not self.detection_active or self.navigation_in_progress:
            return
            
        with self.lock:
            rgb = self.rgb_image.copy() if self.rgb_image is not None else None
            depth = self.depth_image.copy() if self.depth_image is not None else None
            frame_id = self.camera_frame
            K = self.K
        
        if rgb is None or depth is None or K is None or frame_id is None:
            return
        
        try:
            # YOLO 추론
            results = self.model.predict(rgb, verbose=False, conf=self.MIN_CONFIDENCE)
            
            annotated_image = rgb.copy()
            detection_summary = []
            car_found = False
            best_car = None
            max_area = 0
            
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
                    
                    # 차량 감지 여부 확인
                    is_car = cls_name == self.TARGET_OBJECT
                    color = (0, 0, 255) if is_car else (0, 255, 0)
                    
                    # 시각화
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3 if is_car else 2)
                    label = f'{cls_name} {conf}'
                    cv2.putText(annotated_image, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 바운딩 박스 중심에 표시
                    cv2.circle(annotated_image, (xc, yc), 5, (255, 0, 255), -1)
                    cv2.drawMarker(annotated_image, (xc, yc), (255, 0, 255), 
                                  cv2.MARKER_CROSS, 20, 2)
                    
                    if not np.isnan(distance_m):
                        distance_text = f'{distance_m:.2f}m'
                        cv2.putText(annotated_image, distance_text, (x1, y2 + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        detection_summary.append(f'{cls_name} ({conf}, {distance_text})')
                        
                        # 가장 큰 차량 선택 (가장 가까운 차량)
                        if is_car:
                            box_area = (x2 - x1) * (y2 - y1)
                            if box_area > max_area:
                                max_area = box_area
                                car_found = True
                                best_car = {
                                    'xc': xc,
                                    'yc': yc,
                                    'distance': distance_m,
                                    'conf': conf,
                                    'bbox': (x1, y1, x2, y2)
                                }
                    else:
                        cv2.putText(annotated_image, 'No Depth', (x1, y2 + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        detection_summary.append(f'{cls_name} ({conf}, No Depth)')
            
            # 이미지 상태 표시
            status_text = f"[{self.stage}] Looking for cars..."
            if car_found:
                status_text = f"[{self.stage}] CAR DETECTED! Distance: {best_car['distance']:.2f}m"
            cv2.putText(annotated_image, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Annotated 이미지 발행
            try:
                img_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                self.annotated_image_publisher.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish annotated image: {e}')
            
            # 결과 발행
            result_msg = String()
            if car_found:
                result_msg.data = f"CAR DETECTED! {detection_summary[0]}"
            else:
                result_msg.data = " | ".join(detection_summary) if detection_summary else "No objects detected"
            self.result_publisher.publish(result_msg)
            
            # 차량 발견 시 접근
            if car_found and best_car and not self.car_detected:
                self.car_detected = True
                self.stage = "APPROACHING"
                self.get_logger().info(
                    f"[STAGE 3] Car detected! Distance: {best_car['distance']:.2f}m, "
                    f"Confidence: {best_car['conf']:.2f}"
                )
                self.get_logger().info(
                    f"[STAGE 3] Approaching car center at pixel ({best_car['xc']}, {best_car['yc']})"
                )
                self.approach_car(best_car, frame_id, K, depth)
                    
        except Exception as e:
            self.get_logger().error(f"Detection processing error: {e}")
    
    def approach_car(self, car_info, frame_id, K, depth):
        """3단계: 차량의 바운딩 박스 중심을 향해 접근"""
        try:
            xc, yc = car_info['xc'], car_info['yc']
            z = car_info['distance']
            
            if z <= 0.2 or z >= 10.0:
                self.get_logger().warn(f"Invalid depth value: {z:.2f}m. Retrying...")
                self.car_detected = False
                return
            
            # 카메라 좌표계에서 3D 포인트 계산
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            X = (xc - cx) * z / fx
            Y = (yc - cy) * z / fy
            Z = z
            
            self.get_logger().info(f"Camera frame 3D point: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")
            
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
                f"Car position in map: ({pt_map.point.x:.2f}, {pt_map.point.y:.2f}, {pt_map.point.z:.2f})"
            )
            
            # 로봇의 현재 위치 가져오기 (간단화를 위해 목표 지점으로 직접 이동)
            # 실제로는 로봇에서 차량까지의 방향을 계산해야 함
            
            # 접근 거리를 고려한 목표 위치 계산
            dx = pt_map.point.x
            dy = pt_map.point.y
            distance_xy = math.sqrt(dx**2 + dy**2)
            
            if distance_xy > self.APPROACH_DISTANCE:
                # 목표 거리만큼 떨어진 지점 계산
                ratio = (distance_xy - self.APPROACH_DISTANCE) / distance_xy
                goal_x = dx * ratio
                goal_y = dy * ratio
            else:
                # 이미 충분히 가까움
                self.get_logger().info(f"Already close enough to car ({distance_xy:.2f}m)")
                self.navigation_in_progress = False
                return
            
            # 목표 포즈 생성
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = goal_x
            goal_pose.pose.position.y = goal_y
            goal_pose.pose.position.z = 0.0
            
            # 차량을 향한 방향 계산
            yaw = math.atan2(dy, dx)
            qz = math.sin(yaw / 2.0)
            qw = math.cos(yaw / 2.0)
            goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            
            # 네비게이션 시작
            self.navigation_in_progress = True
            self.detection_active = False  # 접근 중에는 탐지 중지
            
            self.get_logger().info(
                f"[STAGE 3] Moving towards car bounding box center..."
            )
            self.get_logger().info(
                f"Goal position: ({goal_x:.2f}, {goal_y:.2f}), "
                f"Target distance from car: {self.APPROACH_DISTANCE:.2f}m"
            )
            
            self.navigator.goToPose(goal_pose)
            
            # 네비게이션 완료 대기
            self.nav_complete_timer = self.create_timer(1.0, self.check_approach_navigation)
                
        except Exception as e:
            self.get_logger().error(f"Approach car failed: {e}")
            self.navigation_in_progress = False
            self.car_detected = False
    
    def check_approach_navigation(self):
        """차량 접근 네비게이션 완료 확인"""
        if not self.navigator.isTaskComplete():
            return
        
        self.nav_complete_timer.cancel()
        self.navigation_in_progress = False
        
        self.get_logger().info("[STAGE 3] Reached target position near car!")
        self.get_logger().info("Mission completed. Robot will stay at current position.")
        
        # 미션 완료 후 계속 탐지할지 결정
        # 여기서는 탐지를 계속하지 않음
        # 만약 계속 탐지하려면 다음 줄의 주석을 해제
        # self.detection_active = True
        # self.car_detected = False
    
    def destroy_node(self):
        """노드 종료"""
        self.get_logger().info("Shutting down car following node...")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CarFollowingNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected.")
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()