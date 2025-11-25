import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String # 간단한 결과 메시지 발행을 위해 String 사용
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloInferenceNode(Node):
    """
    ROS 2에서 이미지 토픽을 구독하고, YOLO 추론 후 결과를 다시 발행하는 노드
    """
    def __init__(self):
        super().__init__('yolo_inference_node')
        
        # 1. 모델 초기화
        self.get_logger().info('YOLOv8 모델을 로드 중...')
        # 'yolov8n.pt'는 예시이며, 실제 사용할 .pt 파일 경로로 변경해야 합니다.
        self.model = YOLO('yolov8n.pt') 
        self.get_logger().info('YOLOv8 모델 로드 완료.')
        
        # 2. CV Bridge 초기화 (ROS <-> OpenCV 이미지 변환)
        self.bridge = CvBridge()

        # 3. 구독자(Subscriber) 설정: 카메라 이미지 토픽 구독
        # /image_raw 토픽에서 sensor_msgs/Image 메시지를 받습니다.
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # 실제 카메라 이미지 토픽 이름으로 변경 필요
            self.image_callback,
            10) # QoS 큐 사이즈 10

        # 4. 발행자(Publisher) 설정: 추론 결과 발행 (간단하게 String 메시지 사용)
        self.publisher_ = self.create_publisher(
            String, 
            '/yolo_detection_results', # 발행할 토픽 이름
            10)
            
        # 5. 시각화 이미지를 다시 발행하고 싶다면 이 주석을 해제하세요.
        # self.image_publisher = self.create_publisher(Image, '/annotated_image', 10)


    def image_callback(self, msg):
        """
        새로운 이미지 메시지가 수신될 때마다 호출되는 콜백 함수
        """
        try:
            # 1. ROS Image -> OpenCV Image (NumPy Array) 변환
            # cv_image는 이후 추론과 드로잉에 사용됩니다.
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'cv_bridge 변환 실패: {e}')
            return

        # 2. YOLO 추론 수행
        results = self.model.predict(cv_image, verbose=False)
        
        detection_summary = []
        
        # 3. 추론 결과 처리 및 이미지에 바운딩 박스 그리기
        for r in results:
            # 바운딩 박스 정보를 직접 시각화에 사용
            for box in r.boxes:
                # 바운딩 박스 좌표 (xyxy 형식)
                x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]
                conf = round(box.conf[0].item(), 2)
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]
                
                detection_summary.append(f'{cls_name} ({conf})')

                # 🖍️ OpenCV를 사용하여 이미지에 바운딩 박스 그리기
                color = (0, 255, 0) # 초록색
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(cv_image, f'{cls_name} {conf}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        
        
        # 4. (선택 사항) 텍스트 결과 ROS 토픽 발행
        result_msg = String()
        result_msg.data = "Detected: " + " | ".join(detection_summary) if detection_summary else "No objects detected."
        self.publisher_.publish(result_msg)
        
        
        # 5. 🖼️ 시각화된 이미지 ROS 토픽 발행 (핵심!)
        try:
            # 시각화된 OpenCV 이미지를 ROS Image 메시지로 변환
            annotated_img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            
            # 원본 메시지의 헤더(타임스탬프, 프레임 ID)를 복사하여 시간 동기화 유지
            annotated_img_msg.header = msg.header 
            
            # 발행
            self.image_publisher.publish(annotated_img_msg)
            self.get_logger().info(f'시각화 이미지 발행 완료. (탐지 개수: {len(detection_summary)})')
            
        except Exception as e:
            self.get_logger().error(f'이미지 발행 실패: {e}')


def main(args=None):
    rclpy.init(args=args)
    yolo_inference_node = YoloInferenceNode()
    
    # 노드 실행 (이미지 콜백이 주기적으로 호출됨)
    rclpy.spin(yolo_inference_node) 

    # 노드 종료 시 자원 해제
    yolo_inference_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()