import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class ResultPublisher(Node):
    def __init__(self):
        super().__init__('result_publisher')
        self.publisher_ = self.create_publisher(Image, '/robot2/model_output', 10)
        timer_period = 0.1  # Publish at 10 Hz
        self.timer = self.create_timer(timer_period, self.publish_image)
        self.bridge = CvBridge()

    def publish_image(self):
        frame = ""    # subscribed image
        if frame:
            # Convert OpenCV image (BGR) to ROS Image message
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
            self.get_logger().info('Published image')
        else:
            self.get_logger().warn('Failed to capture image')

    def destroy_node(self):
        self.cap.release()  # Release the camera when shutting down
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = ResultPublisher()

    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()