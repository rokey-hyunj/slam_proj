import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/robot0/oakd/rgb/preview/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.frame = None

    def listener_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def main():
    capture_interval = 0.1  # 1초에 10장씩

    rclpy.init()
    sub_node = CameraSubscriber()
    last_capture_time = time.time()

    try:
        while rclpy.ok():
            rclpy.spin(sub_node, timeout_sec=0.5)

            if sub_node.frame is not None:
                img = sub_node.frame    # subscriber에서 전달받은 이미지 (모델로 전달)
                now = time.time()
                if now - last_capture_time >= capture_interval:
                    last_capture_time = now

    except KeyboardInterrupt:
        pass
    finally:
        sub_node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()