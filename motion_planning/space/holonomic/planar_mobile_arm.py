import time

import numpy as np
from shapely import LineString, affinity
from sklearn.metrics import pairwise_distances

from motion_planning.space import HolonomicRobot
from motion_planning.tools import NumpyState
from motion_planning.utils import create_rectangle_geometry, issue_warning


class PlanarMobileArm(HolonomicRobot):
    def __init__(
            self, 
            edge_validity_delta: float = 0.5, 
            base_width: float = 2,
            base_length: float = 0.1,
            num_links: int = 3,
            arm_lengths: list[int] | None = None
    ):
        super().__init__()
        self.base_width = base_width
        self.base_length = base_length

        self.x_range = [-10, 10]
        self.y_range = [-10, 10]
        self.theta1_range = [0, np.pi]
        self.theta_range = [0, 2*np.pi]

        self.edge_validity_delta = edge_validity_delta

        assert (num_links > 0), "Num Links Must Be Greater Than Zero"
        
        self.num_links = num_links
        self.arm_lengths = np.array([1] * self.num_links)

        if arm_lengths is not None:
            assert (num_links == len(arm_lengths)), "Num Links must be equal to the list size of arm lengths"
            self.arm_lengths = np.array(arm_lengths)
        
        self.obstacle_check_time = 0
        self.generate_state_time = 0

    def dist(self, state1: NumpyState, state2: NumpyState) -> float:
        state1 = self.get_state_value(state1)
        state2 = self.get_state_value(state2)
        return np.linalg.norm(state1 - state2)
    
    def make_state(self, state: np.ndarray) -> NumpyState:
        return NumpyState(value=state)
    
    def sample_point(self) -> NumpyState:
        x = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
        y = np.random.uniform(low=self.y_range[0], high=self.y_range[1])
        theta1 = np.random.uniform(low=self.theta1_range[0], high=self.theta1_range[1])
        link_thetas = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1], size=(self.num_links-1,))
        return self.make_state(np.array([x, y, theta1, *link_thetas]))
    
    # TODO: Convert to return only an np.ndarray
    def create_end_effector_representation(self, base_point: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        x, y = base_point

        ee_lengths = [0.5, 0.2]
        ee_angles = [np.pi/6, 2*np.pi/3]
        # cumulative_ee_angles = np.cumsum(ee_angles)

        r_joint_pos1 = np.array([
            ee_lengths[0] * np.cos(ee_angles[0]),
            ee_lengths[0] * np.sin(ee_angles[0]),
        ]) + base_point

        r_joint_pos2 = r_joint_pos1 + np.array([
            ee_lengths[1] * np.cos(ee_angles[0] + ee_angles[1]),
            ee_lengths[1] * np.sin(ee_angles[0] + ee_angles[1]),
        ])

        p1 = np.pi - ee_angles[0]
        p2 = 2*np.pi - ee_angles[1]

        l_joint_pos1 = np.array([
            ee_lengths[0] * np.cos(p1),
            ee_lengths[0] * np.sin(p1),
        ]) + base_point

        l_joint_pos2 = l_joint_pos1 + np.array([
            ee_lengths[1] * np.cos(p1+p2),
            ee_lengths[1] * np.sin(p1+p2),
        ])

        ee_point_pairs = [(base_point, r_joint_pos1),
                          (r_joint_pos1, r_joint_pos2),
                          (base_point, l_joint_pos1),
                          (l_joint_pos1, l_joint_pos2)]
        
        return ee_point_pairs
    
    def get_arm_base_position(self, state) -> np.ndarray:
        state = self.get_state_value(state)
        x, y, *_ = state
        return np.array([x, y+self.base_length/2])

    def forward_kinematics(self, state) -> np.ndarray:
        state = self.get_state_value(state)
        x, y, *thetas = state

        thetas = np.array(thetas)
        thetas[1:] = thetas[1:] - np.pi # Hack to treat angles properly

        arm_base = self.get_arm_base_position(state)
        link_thetas = np.cumsum(thetas)

        cos = np.cos(link_thetas)
        sin = np.sin(link_thetas)
        normalized_link_points = np.vstack((cos, sin)).T

        point_der = self.arm_lengths.reshape(self.num_links, 1) * normalized_link_points
        joint_pos = np.cumsum(point_der, axis=0) + arm_base
        return np.vstack((arm_base, joint_pos))

    def generate_robot_representation(self, state):
        state = self.get_state_value(state)
        x, y, *thetas = state

        robot = create_rectangle_geometry(x_loc=x, y_loc=y, x_width=self.base_width, y_length=self.base_length)

        rotation_offset = np.pi/2 if self.num_links % 2 == 0 else -np.pi/2
        arms = []

        joint_positions = self.forward_kinematics(state)

        for i in range(len(joint_positions) - 1):
            arm = LineString([joint_positions[i], joint_positions[i+1]])
            arms.append(arm)

        end_effector_point_pairs = self.create_end_effector_representation(joint_positions[-1])

        ee = []
        for i in range(len(end_effector_point_pairs)):
            ee_link = LineString(end_effector_point_pairs[i])
            ee_link = affinity.rotate(ee_link, angle=(np.sum(thetas)+rotation_offset), use_radians=True, origin=list(joint_positions[-1]))
            ee.append(ee_link)

        return robot, arms, ee

    def draw_state(self, ax, state, color='red'):
        robot, arms, ee = self.generate_robot_representation(state)
        ax.plot(*robot.exterior.xy, color=color)

        for arm in arms:
            ax.plot(*arm.xy, color=color)
        
        for ee_link in ee:
            ax.plot(*ee_link.xy, color=color)

    def collides_with_self(self, robot, arms, ee) -> bool:
        for i in range(len(arms)):
            for j in range(i+2, len(arms)):
                if arms[i].intersects(arms[j]):
                    return True

            if i > 0:
                if robot.intersects(arms[i]):
                    return True 
        return False 
    
    def is_valid(self, state) -> bool:
        start_time = time.time()
        self.num_collision_checks += 1
        robot, arms, ee = self.generate_robot_representation(state)
        self.generate_state_time += (time.time() - start_time)
        # if self.collides_with_self(robot, arms, ee):
        #     return False
        start_time = time.time()
        for obs in self.obstacles:
            if obs.intersects(robot):
                return False
            for arm in arms:
                if obs.intersects(arm):
                    return False
            for ee_link in ee:
                if obs.intersects(ee_link):
                    return False
        self.obstacle_check_time += (time.time() - start_time)

        return True

    def inverse_kinematics(self, target_ee_position):
        # NEED TO UPDATE TO STOP SAMPLING FROM A RANDOM POINT
        q = self.sample_valid_point().value

        # q_start = np.array([self.sample_valid_point().value for _ in range(10000)])

        d = q.shape[0]

        num_steps = 1000
        sample_size = 10

        scale = 0.1
        tolerance = 0.1

        # q_start = np.full((sample_size, d), )

        for i in range(num_steps):
            noise = np.random.normal(scale=scale, size=(sample_size, d))
            q_near = q.reshape(1, -1) + noise
            # validities = np.array([self.is_valid(self.make_state(q_val)) for q_val in q_near])
            # q_near = q_near[validities]

            positions = self.batch_forward_kinematics(q_near)
            ee_positions = positions[:, 3, :]

            dist_mat = pairwise_distances(target_ee_position.reshape(1, -1), ee_positions)
            dist_array = dist_mat[0]
            best_idx = np.argmin(dist_array)
            error = np.min(dist_array)
            q = q_near[best_idx]

            if error < tolerance:
                return self.make_state(q)

            # scale = scale * 0.99

        # return self.make_state(q)
        return None

    def batch_sample_point(self, num_points) -> np.ndarray:
        issue_warning(True, "Batch Sample Points is hardcoded for planar mobile arm", 'warning')
        return np.random.uniform(low=np.array([-10,-10,0,0,0]), high=np.array([10,10,2*np.pi,2*np.pi,2*np.pi]), size=(num_points, 5))

    def sample_configs_ee_target(self, target_ee_position) -> np.ndarray:
        tolerance = 0.1
        # q_start = np.array([self.sample_valid_point().value for _ in range(20000)])
        q_start = self.batch_sample_point(100000)
        positions = self.batch_forward_kinematics(q_start)
        ee_positions = positions[:, 3, :]
        dist_mat = pairwise_distances(target_ee_position.reshape(1, -1), ee_positions)
        dist_array = dist_mat[0]
        mask = dist_array < tolerance
        final_qs = q_start[mask]
        return final_qs

    ## ---- Batched Methods ---- ##
    def batch_forward_kinematics(self, states : np.ndarray) -> np.ndarray:
        states = np.copy(states)
        states[:, 3:] -= np.pi # Hack to treat angles properly
        arm_bases = np.vstack((states[:, 0], states[:, 1] + self.base_length/2)).T # (B, 2)
        link_thetas = np.cumsum(states[:, 2:], axis=1) # (B, num_links)
        link_cosines = np.cos(link_thetas) # (B, num_links)
        link_sines = np.sin(link_thetas) # (B, num_links)
        normalized_link_points = np.stack((link_cosines, link_sines), axis=2) # (B, num_links, 2)
        point_der = self.arm_lengths.reshape(1, self.num_links, 1) * normalized_link_points # (B, num_links, 2)
        joint_pos = np.cumsum(point_der, axis=1) + arm_bases.reshape(-1, 1, 2) # (B, num_links, 2)
        return np.concatenate((arm_bases.reshape(-1, 1, 2), joint_pos), axis=1)
    
    def batch_create_end_effector_segments(self, base_points: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """
        base_points: (N, 2) batch of N points from the end_effector joint
        """
        N, _ = base_points.shape

        xs = base_points[:, 0]
        ys = base_points[:, 1]

        ee_lengths = [0.5, 0.2]
        ee_angles = [np.pi/6, 2*np.pi/3]

        r_joint_pos1 = np.array([
            ee_lengths[0] * np.cos(ee_angles[0]),
            ee_lengths[0] * np.sin(ee_angles[0]),
        ]).reshape(1, 2) + base_points

        r_joint_pos2 = r_joint_pos1 + np.array([
            ee_lengths[1] * np.cos(ee_angles[0] + ee_angles[1]),
            ee_lengths[1] * np.sin(ee_angles[0] + ee_angles[1]),
        ])

        p1 = np.pi - ee_angles[0]
        p2 = 2*np.pi - ee_angles[1]

        l_joint_pos1 = np.array([
            ee_lengths[0] * np.cos(p1),
            ee_lengths[0] * np.sin(p1),
        ]).reshape(1, 2) + base_points

        l_joint_pos2 = l_joint_pos1 + np.array([
            ee_lengths[1] * np.cos(p1+p2),
            ee_lengths[1] * np.sin(p1+p2),
        ])

        ee_arm_1 = np.stack((base_points, l_joint_pos1), axis=2).transpose(0, 2, 1)
        ee_arm_2 = np.stack((l_joint_pos1, l_joint_pos2), axis=2).transpose(0, 2, 1)
        ee_arm_3 = np.stack((base_points, r_joint_pos1), axis=2).transpose(0, 2, 1)
        ee_arm_4 = np.stack((r_joint_pos1, r_joint_pos2), axis=2).transpose(0, 2, 1)

        batch_ee_points = np.concatenate((ee_arm_1, ee_arm_2, ee_arm_3, ee_arm_4), axis=0)

        cos = np.cos(thetas)
        sin = np.sin(thetas)

        rotation_mats = np.array([[cos, -sin, -xs*cos+ys*sin+xs],
                                  [sin, cos, -xs*sin-ys*cos+ys],
                                  [np.zeros_like(cos), np.zeros_like(cos), np.ones_like(cos)]])
        
        batch_ee_points_homogeneous = np.concatenate((batch_ee_points, np.ones(shape=(len(base_points)*4, 2, 1))), axis=2).reshape(-1, 4, 2, 3)

        rotation_mats = rotation_mats.transpose(2, 0, 1)

        batch_ee_points_rotated = rotation_mats @ batch_ee_points_homogeneous.reshape(-1, 8, 3).transpose(0, 2, 1)

        batch_ee_points_rotated = batch_ee_points_rotated.transpose(0, 2, 1).reshape(-1, 4, 2, 3)
        batch_ee_points_rotated = batch_ee_points_rotated[:, :, :, :2]
        batch_ee_points_rotated = batch_ee_points_rotated.reshape(-1, 2, 2)

        return batch_ee_points_rotated
    
    def batch_get_robot_representations(self, states: np.ndarray) -> dict:
        rectangles = np.stack((states[:, 0], states[:, 1], np.ones((states.shape[0])) * self.base_width, np.ones((states.shape[0])) * self.base_length), axis=1)
        segment_points = self.batch_forward_kinematics(states)#.reshape(-1, 2, 2)
        start_points = segment_points[:, :-1, :]
        end_points = segment_points[:, 1:, :]
        segments = np.concatenate((start_points, end_points), axis=2).reshape(-1, 4).reshape(-1, 2, 2)

        ## Handling End Effector Lines -- BEGIN
        """
        While this does work, in practice it is not the most efficient. 

        This is due to the fact that the segments of the robot arms and the ee lines are 
        not the same lengths. ApproximationSpace is better optimized for lines of the same length. 

        Since adding the end effector lines breaks this same length status, ApproximationSpace must use a for loop for some verification, 
        causing the inefficiency.

        Therefore, these segments will be commented out unless desired by the end user
        """

        rotation_offset = np.pi/2 if self.num_links % 2 == 0 else -np.pi/2
        summed_thetas = np.sum(states[:, 2:], axis=1) + rotation_offset
        ee_segments = self.batch_create_end_effector_segments(end_points[:, -1, :], summed_thetas)
        segments = np.concatenate((segments, ee_segments), axis=0)

        ## Handling End Effector Lines -- END

        return {
            'rectangles': rectangles,
            'segments': segments, 
            'points': np.empty((0, 2)),
            'points_radius': 0.0,
            'segments_radii': 0.1, 
        }

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = PlanarMobileArm()
    state = env.sample_point()

    env.draw_environment(plt.gca())
    env.draw_state(plt.gca(), state)
    plt.show()