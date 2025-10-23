import numpy as np
from config import Config


class Environment:
    def __init__(self, width=15.0, height=15.0, resolution=0.2):
        self.width = width
        self.height = height
        self.resolution = resolution

        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        # Initialize occupancy grid with zeros (empty)
        self.occupancy_grid = np.zeros((self.grid_width, self.grid_height))

        self.obstacles = []

    def add_obstacle(self, x, y, width, height, theta=0):
        """Add rectangular obstacle"""
        self.obstacles.append({
            'x': x, 'y': y, 'width': width, 'height': height, 'theta': theta
        })

        # Update occupancy grid
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                # Convert grid index to world coordinates
                gx = i * self.resolution
                gy = j * self.resolution

                # Calculate coordinates relative to the obstacle center
                dx = gx - x
                dy = gy - y
                # Rotate the relative coordinates back by -theta (the obstacle's rotation)
                rx = dx * np.cos(-theta) - dy * np.sin(-theta)
                ry = dx * np.sin(-theta) + dy * np.cos(-theta)

                # Check if the point is inside the rotated rectangle
                if abs(rx) <= width / 2 and abs(ry) <= height / 2:
                    self.occupancy_grid[i, j] = 1  # Mark as occupied

    def create_parallel_parking_scenario(self, scs=1.6):
        # Reset the environment
        self.occupancy_grid = np.zeros((self.grid_width, self.grid_height))
        self.obstacles = []

        vehicle_y = 5.0  # The Y-coordinate of the parked vehicles (and the goal)
        parking_length = scs * Config.E_l

        # Front car (obstacle)
        front_x = 6.0
        self.add_obstacle(front_x, vehicle_y, Config.E_l, Config.E_w)

        # Rear car (obstacle)
        rear_x = front_x + Config.E_l + parking_length
        self.add_obstacle(rear_x, vehicle_y, Config.E_l, Config.E_w)

        # Start and Goal positions
        init_x = 2.0
        init_y = 9.0
        init_theta = 0.0

        # Goal is centered in the parking spot
        goal_x = front_x + Config.E_l / 2 + parking_length / 2
        goal_y = vehicle_y
        goal_theta = 0.0  # Parallel to the obstacles

        # Return start state, goal state, and True indicating parallel parking
        return (init_x, init_y, init_theta), (goal_x, goal_y, goal_theta), True

    def create_perpendicular_parking_scenario(self, scs=1.4):

        # Reset the environment
        self.occupancy_grid = np.zeros((self.grid_width, self.grid_height))
        self.obstacles = []

        parking_width = scs * Config.E_w

        parking_area_x = 8.0  # X-coordinate center of the parking area

        # Left adjacent car (parked perpendicularly)
        left_car_x = parking_area_x - parking_width / 2 - Config.E_w / 2
        left_car_y = 4.0
        self.add_obstacle(left_car_x, left_car_y, Config.E_w, Config.E_l, theta=np.pi / 2)

        # Right adjacent car (parked perpendicularly)
        right_car_x = parking_area_x + parking_width / 2 + Config.E_w / 2
        right_car_y = 4.0
        self.add_obstacle(right_car_x, right_car_y, Config.E_w, Config.E_l, theta=np.pi / 2)

        # Initial position (start)
        init_x = parking_area_x
        init_y = 10.0
        init_theta = 0.0

        # Calculate the Y-coordinate of the rear boundary of the adjacent obstacles
        obstacle_rear_y = left_car_y - Config.E_l / 2

        safety_margin = 0.3
        # The front of the goal vehicle should be here
        goal_vehicle_front_y = obstacle_rear_y - safety_margin

        # Goal vehicle center Y
        goal_y = goal_vehicle_front_y - Config.E_l / 2
        goal_x = parking_area_x
        goal_theta = np.pi / 2  # Perpendicular parking orientation

        # Boundary check and adjustment for environment height (min_y_boundary)
        min_y_boundary = 1.0
        if goal_y - Config.E_l / 2 < min_y_boundary:
            # Calculate required adjustment to move the entire setup up
            adjustment = min_y_boundary - (goal_y - Config.E_l / 2) + 0.5

            # Re-initialize/reset the environment
            self.occupancy_grid = np.zeros((self.grid_width, self.grid_height))
            self.obstacles = []

            # Apply adjustment to y-coordinates
            left_car_y += adjustment
            right_car_y += adjustment
            goal_y += adjustment

            # Re-add adjusted obstacles
            self.add_obstacle(left_car_x, left_car_y, Config.E_w, Config.E_l, theta=np.pi / 2)
            self.add_obstacle(right_car_x, right_car_y, Config.E_w, Config.E_l, theta=np.pi / 2)

            # Re-calculate obstacle_rear_y with the new left_car_y
            obstacle_rear_y = left_car_y - Config.E_l / 2  # Update for printout

        # Printout of the scenario configuration
        print(f"\n  [Perpendicular Parking Configuration v2]")
        print(f"    Parking Spot Center X: {parking_area_x:.2f}m")
        print(f"    Parking Spot Width: {parking_width:.2f}m (SCS={scs:.2f})")
        print(f"    Left Vehicle: ({left_car_x:.2f}, {left_car_y:.2f})")
        print(f"    Right Vehicle: ({right_car_x:.2f}, {right_car_y:.2f})")
        print(f"    Obstacle Rear Y: {obstacle_rear_y:.2f}m")
        print(f"    Goal Position: ({goal_x:.2f}, {goal_y:.2f}, {np.degrees(goal_theta):.0f}°)")
        print(f"    Goal Vehicle Rear Y: {goal_y - Config.E_l / 2:.2f}m")
        print(f"    Goal Vehicle Front Y: {goal_y + Config.E_l / 2:.2f}m")

        # Return start state, goal state, and False indicating not parallel (i.e., perpendicular) parking
        return (init_x, init_y, init_theta), (goal_x, goal_y, goal_theta), False