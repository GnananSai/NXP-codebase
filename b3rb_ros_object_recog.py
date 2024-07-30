# Copyright 2024 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from synapse_msgs.msg import TrafficStatus
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import threading

QOS_PROFILE_DEFAULT = 10

class ObjectRecognizer(Node):
    def __init__(self):
        super().__init__('object_recognizer')
        
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)
        
        self.publisher_traffic = self.create_publisher(
            TrafficStatus,
            '/traffic_status',
            QOS_PROFILE_DEFAULT)
        
        self.image_queue = []
        self.lock = threading.Lock()
        self.stop_sign_detected = False
        
        self.detection_thread = threading.Thread(target=self.process_images)
        self.detection_thread.start()

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        with self.lock:
            self.image_queue.append(image)
    
    def process_images(self):
        while rclpy.ok():
            with self.lock:
                if self.image_queue:
                    image = self.image_queue.pop(0)
                else:
                    image = None
            
            if image is not None:
                if self.detect_stop_sign(image):
                    self.stop_sign_detected = True
                    traffic_status_message = TrafficStatus()
                    traffic_status_message.stop_sign = True
                    self.publisher_traffic.publish(traffic_status_message)

    def detect_stop_sign(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 8:
                area = cv2.contourArea(contour)
                if area > 30:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = image[y:y+h, x:x+w]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    
                    lower_red1 = np.array([0, 70, 50])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 70, 50])
                    upper_red2 = np.array([180, 255, 255])
                    
                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    
                    mask = cv2.bitwise_or(mask1, mask2)
                    red_percentage = (cv2.countNonZero(mask) / (w * h)) * 100
                    
                    if red_percentage > 50:
                        return True
        
        return False

def main(args=None):
    rclpy.init(args=args)
    object_recognizer = ObjectRecognizer()
    rclpy.spin(object_recognizer)
    object_recognizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
