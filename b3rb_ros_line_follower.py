import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
import math

from synapse_msgs.msg import EdgeVectors
from synapse_msgs.msg import TrafficStatus
from sensor_msgs.msg import LaserScan

QOS_PROFILE_DEFAULT = 10

PI = math.pi

LEFT_TURN = +1.0
RIGHT_TURN = -1.0

TURN_MIN = 0.0
TURN_MAX = 1.0
SPEED_MIN = 0.0
SPEED_MAX = 1.0
SPEED_25_PERCENT = SPEED_MAX / 4
SPEED_50_PERCENT = SPEED_25_PERCENT * 2
SPEED_75_PERCENT = SPEED_25_PERCENT * 3

THRESHOLD_OBSTACLE_VERTICAL = 1
THRESHOLD_OBSTACLE_HORIZONTAL = 0.25
THRESHOLD_VECTOR_DISTANCE = 300  # Adjust this based on your requirements
SHARP_TURN_THRESHOLD = 0.62  # Adjust this threshold to identify sharp turns

class LineFollower(Node):
    """ Initializes line follower node with the required publishers and subscriptions.

        Returns:
            None
    """
    def __init__(self):
        super().__init__('line_follower')

        # Subscription for edge vectors.
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        # Publisher for joy (for moving the rover in manual mode).
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Subscription for traffic status.
        self.subscription_traffic = self.create_subscription(
            TrafficStatus,
            '/traffic_status',
            self.traffic_status_callback,
            QOS_PROFILE_DEFAULT)

        # Subscription for LIDAR data.
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        self.traffic_status = TrafficStatus()

        self.obstacle_detected = False

        self.ramp_detected = False
        self.ramp_timer = 0  # Timer to keep ramp detection status
        self.ramp_timer_threshold = 180  # Number of iterations to persist ramp detection after last detection

        # Create a timer to decrement the ramp timer
        self.create_timer(0.1, self.decrement_ramp_timer)

        # Turn history for smoothing
        self.turn_history = []
        self.turn_history_size = 5

        # Previous vectors for vector prediction
        self.previous_vectors = None

    def decrement_ramp_timer(self):
        if self.ramp_timer > 0:
            self.ramp_timer -= 1
        else:
            self.ramp_detected = False

    """ Operates the rover in manual mode by publishing on /cerebri/in/joy.

        Args:
            speed: the speed of the car in float. Range = [-1.0, +1.0];
                Direction: forward for positive, reverse for negative.
            turn: steer value of the car in float. Range = [-1.0, +1.0];
                Direction: left turn for positive, right turn for negative.

        Returns:
            None
    """
    def rover_move_manual_mode(self, speed, turn):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, speed, 0.0, turn]
        self.publisher_joy.publish(msg)

    """ Analyzes edge vectors received from /edge_vectors to achieve line follower application.
        It checks for existence of ramps & obstacles on the track through instance members.
            These instance members are updated by the lidar_callback using LIDAR data.
        The speed and turn are calculated to move the rover using rover_move_manual_mode.

        Args:
            message: "~/cognipilot/cranium/src/synapse_msgs/msg/EdgeVectors.msg"

        Returns:
            None
    """
    def edge_vectors_callback(self, message):
        speed = SPEED_MAX
        turn = 0.0

        vectors = message
        half_width = vectors.image_width / 2

        def vector_distance(vector):
            return math.sqrt((vector[1].x - vector[0].x)**2 + (vector[1].y - vector[0].y)**2)

        # Filter vectors based on distance threshold
        filtered_vectors = []
        for i in range(vectors.vector_count):
            vector = getattr(vectors, f'vector_{i+1}')
            if vector_distance(vector) <= THRESHOLD_VECTOR_DISTANCE:
                filtered_vectors.append(vector)

        # Use previous vectors if no vectors detected
        if len(filtered_vectors) == 0 and self.previous_vectors is not None:
            filtered_vectors = self.previous_vectors
        else:
            self.previous_vectors = filtered_vectors

        # Calculate deviation for the filtered vectors
        if len(filtered_vectors) == 0:
            turn = 0.0
        elif len(filtered_vectors) == 1:
            # Handle the case of one edge vector (turn)
            vector = filtered_vectors[0]
            deviation = vector[1].x - vector[0].x
            turn = deviation / vectors.image_width
        elif len(filtered_vectors) == 2:
            # Handle the case of two edge vectors (straight line)
            deviations = []
            for vector in filtered_vectors:
                deviations.append(vector[1].x - vector[0].x)
            average_deviation = sum(deviations) / len(deviations)
            turn = average_deviation / vectors.image_width

        if self.traffic_status.stop_sign is True:
            speed = SPEED_MIN
            print("stop sign detected")

        if self.ramp_detected is True:
            # Maintain reduced speed on the ramp.
            speed = 0.45
            print("ramp/bridge detected")

        if self.obstacle_detected is True:
            # Reduce speed for obstacles.
            print("obstacle detected")
        
        # Clamp the turn value
        turn = max(min(turn, TURN_MAX), -TURN_MAX)

        # Adjust speed based on turn severity
        if abs(turn) > SHARP_TURN_THRESHOLD:
            speed *= 0.29

        

        # Add current turn to history and keep history size constant
        self.turn_history.append(turn)
        if len(self.turn_history) > self.turn_history_size:
            self.turn_history.pop(0)

        # Calculate smoothed turn value
        smoothed_turn = sum(self.turn_history) / len(self.turn_history)

        self.rover_move_manual_mode(speed, smoothed_turn)


    """ Updates instance member with traffic status message received from /traffic_status.

        Args:
            message: "~/cognipilot/cranium/src/synapse_msgs/msg/TrafficStatus.msg"

        Returns:
            None
    """
    def traffic_status_callback(self, message):
        self.traffic_status = message

    """ Analyzes LIDAR data received from /scan topic for detecting ramps/bridges & obstacles.

        Args:
            message: "docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html"

        Returns:
            None
    """
    def lidar_callback(self, message):
        shield_vertical = 4
        shield_horizontal = 1
        theta = math.atan(shield_vertical / shield_horizontal)

        # Get the middle half of the ranges array returned by the LIDAR.
        length = float(len(message.ranges))
        ranges = message.ranges[int(length / 4): int(3 * length / 4)]

        # Separate the ranges into the part in the front and the part on the sides.
        length = float(len(ranges))
        front_ranges = ranges[int(length * theta / PI): int(length * (PI - theta) / PI)]
        side_ranges_right = ranges[0: int(length * theta / PI)]
        side_ranges_left = ranges[int(length * (PI - theta) / PI):]

        # Process front ranges for obstacles.
        angle = theta - PI / 2
        for i in range(len(front_ranges)):
            if front_ranges[i] < THRESHOLD_OBSTACLE_VERTICAL:
                self.obstacle_detected = True
                return
            angle += message.angle_increment

        # Process side ranges for obstacles.
        side_ranges_left.reverse()
        for side_ranges in [side_ranges_left, side_ranges_right]:
            angle = 0.0
            for i in range(len(side_ranges)):
                if side_ranges[i] < THRESHOLD_OBSTACLE_HORIZONTAL:
                    self.obstacle_detected = True
                    return
                angle += message.angle_increment

        self.obstacle_detected = False

        # Ramp detection based on distance and slope
        ramp_detection_distance_threshold = 2.0  # Distance threshold in meters to consider ramps
        ramp_threshold = 0.4  # Adjust this threshold based on your requirements
        valid_slopes = []
        
        # Filter front ranges based on distance threshold
        filtered_front_ranges = [r for r in front_ranges if r < ramp_detection_distance_threshold]
        
        # Calculate slopes from filtered front ranges
        for i in range(len(filtered_front_ranges) - 1):
            height_diff = filtered_front_ranges[i + 1] - filtered_front_ranges[i]
            angle_diff = message.angle_increment
            # Calculate slope based on the change in height and the angle difference.
            slope = height_diff / math.sin(angle_diff)
            if not math.isnan(slope) and not math.isinf(slope):
                valid_slopes.append(slope)

        # Consider a ramp detected if a significant portion of the slopes exceed the threshold
        ramp_detected = any(slope > ramp_threshold for slope in valid_slopes)
        if ramp_detected:
            self.ramp_detected = True
            self.ramp_timer = self.ramp_timer_threshold

def main(args=None):
    rclpy.init(args=args)

    line_follower = LineFollower()

    rclpy.spin(line_follower)

    line_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()