import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import message_filters # ROS 메시지 동기화를 위한 라이브러리

# ================================
# 설정 상수
# ================================
# 실제 OAK-D 토픽 이름으로 변경
RGB_TOPIC = '/robot2/oakd/rgb/image_raw'
DEPTH_TOPIC = '/robot2/oakd/stereo/image_raw' 
# ================================

class YoloDepth(Node):
    def __init__(self):
        super().__init__('yolo_depth_fusion_node')
        
        # 1. 모델 초기화
        self.get_logger().info('YOLOv8 모델을 로드 중...')
        self.model = YOLO('/home/rokey/rokey_ws/my_best.pt') 
        self.get_logger().info('YOLOv8 모델 로드 완료.')
        
        # 2. CV Bridge 초기화
        self.bridge = CvBridge()

        # 3. 구독자(Subscriber) 설정: Time Synchronizer 사용
        self.sub_rgb = message_filters.Subscriber(self, Image, RGB_TOPIC)
        self.sub_depth = message_filters.Subscriber(self, Image, DEPTH_TOPIC)

        # ApproximateTimeSynchronizer 대신 TimeSynchronizer를 사용합니다 (시간차가 작을 경우)
        # 큐 사이즈 10, 허용되는 시간차 0.1초 (튜닝 필요)
        self.ts = message_filters.TimeSynchronizer([self.sub_rgb, self.sub_depth], 10)
        self.ts.registerCallback(self.image_depth_callback)

        # 4. 발행자(Publisher) 설정
        self.image_publisher = self.create_publisher(Image, '/oakd/annotated_image_with_depth', 10)


    def image_depth_callback(self, rgb_msg, depth_msg):
        """
        RGB 이미지와 Depth 이미지가 동기화되어 수신될 때 호출되는 콜백 함수
        """
        try:
            # 1. ROS Image -> OpenCV Image 변환
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            # Depth 이미지: uint16 (mm) 또는 float32 (m) 형태이며, 'passthrough' 인코딩 사용
            depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Depth 데이터의 유효성/형식 확인
            if depth_mm.dtype not in [np.uint16, np.float32]:
                 self.get_logger().error(f"Depth 이미지 예상치 못한 dtype: {depth_mm.dtype}. uint16 (mm) 또는 float32 (m) 예상.")
                 return
                 
        except Exception as e:
            self.get_logger().error(f'이미지 변환 실패: {e}')
            return

        detection_summary = []
        
        # 2. YOLO 추론 수행
        # cv_image (RGB) 기반으로 추론
        results = self.model.predict(cv_image, verbose=False)
        
        # 3. 추론 결과 처리, Depth 측정 및 이미지에 바운딩 박스 그리기
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]
                conf = round(box.conf[0].item(), 2)
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]
                
                # 객체 중심 좌표 계산
                xc = int((x1 + x2) / 2)
                yc = int((y1 + y2) / 2)
                
                distance_m = np.nan # 기본값
                
                # 4. Depth 값 추출 (Depth 이미지는 RGB 이미지와 정렬되어야 함)
                # xc, yc 좌표가 Depth 이미지 범위 내에 있는지 확인
                if 0 <= yc < depth_mm.shape[0] and 0 <= xc < depth_mm.shape[1]:
                    # 중심 주변 3x3 픽셀 영역의 깊이 평균 사용 (노이즈 감소)
                    depth_patch = depth_mm[yc-1:yc+2, xc-1:xc+2]
                    valid_depths = depth_patch[depth_patch > 0] # 0은 무효한 깊이로 가정

                    if valid_depths.size > 0:
                        distance_mm = np.mean(valid_depths)
                        distance_m = distance_mm / 1000.0 # 밀리미터(mm)를 미터(m)로 변환

                # 5. 시각화 및 결과 저장
                color = (0, 255, 0) # 초록색
                
                # 바운딩 박스 및 클래스/확률 표시
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(cv_image, f'{cls_name} {conf}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                
                # 중심점 표시
                cv2.circle(cv_image, (xc, yc), 4, (0, 0, 255), -1)

                # 깊이(거리) 표시
                if distance_m is not np.nan:
                    distance_text = f'{distance_m:.2f}m'
                    cv2.putText(cv_image, distance_text, (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    detection_summary.append(f'{cls_name} ({conf}, {distance_text})')
                else:
                    cv2.putText(cv_image, 'No Depth', (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    detection_summary.append(f'{cls_name} ({conf}, No Depth)')
        
        
        # 7. 시각화된 이미지 ROS 토픽 발행
        try:
            annotated_img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            
            # 원본 메시지의 헤더(타임스탬프, 프레임 ID)를 복사하여 시간 동기화 유지
            annotated_img_msg.header = rgb_msg.header 
            
            self.image_publisher.publish(annotated_img_msg)
            self.get_logger().info(f'최종 이미지 발행 완료. (탐지 개수: {len(detection_summary)})')
            
        except Exception as e:
            self.get_logger().error(f'최종 이미지 발행 실패: {e}')


def main(args=None):
    rclpy.init(args=args)
    yolo_depth_node = YoloDepth()
    
    rclpy.spin(yolo_depth_node) 

    yolo_depth_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()