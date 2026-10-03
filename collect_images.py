import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os


class StereoCollector(Node):
    def __init__(self):
        super().__init__('stereo_collector')
        self.bridge = CvBridge()
        self.left_img = None
        self.right_img = None
        self.img_count = 0

        #tworzenie folderów na zdjęcia
        os.makedirs('calib_data/left', exist_ok=True)
        os.makedirs('calib_data/right', exist_ok=True)

        self.sub_left = self.create_subscription(Image, '/left/image_raw', self.left_callback, 10)
        self.sub_right = self.create_subscription(Image, '/right/image_raw', self.right_callback, 10)

        self.timer = self.create_timer(0.05, self.display_loop)
        self.get_logger().info("Gotowe do zapisu. Wcisnij SPACJE, aby zapisac pare, ESC aby wyjsc.")

    def left_callback(self, msg):
        self.left_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def right_callback(self, msg):
        self.right_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def display_loop(self):
        if self.left_img is not None and self.right_img is not None:
            preview = cv2.hconcat([self.left_img, self.right_img])
            preview_resized = cv2.resize(preview, (1280, 360))

            cv2.imshow("Kalibracja - Spacja: Zapisz | ESC: Wyjdz", preview_resized)

            key = cv2.waitKey(1)
            if key == 32:  # Klawisz SPACJA
                l_path = f"calib_data/left/left_{self.img_count:02d}.png"
                r_path = f"calib_data/right/right_{self.img_count:02d}.png"

                cv2.imwrite(l_path, self.left_img)
                cv2.imwrite(r_path, self.right_img)

                self.get_logger().info(f"Zapisano parę zdjęć nr {self.img_count}")
                self.img_count += 1

            elif key == 27:  # Klawisz ESC
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = StereoCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()


if __name__ == '__main__':
    main()
