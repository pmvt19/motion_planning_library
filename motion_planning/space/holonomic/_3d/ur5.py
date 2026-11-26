import numpy as np
import rerun as rr
from sklearn.metrics import pairwise_distances

from motion_planning.space import HolonomicRobot, RobotSpace3D
from motion_planning.tools import NumpyState, AngularNumpyState
from motion_planning.utils import numpystate_distance


class UR5(RobotSpace3D, HolonomicRobot):
    def __init__(self):
        super().__init__()

        self.edge_validity_delta = 0.1

        self.x_range = [-10,10]
        self.y_range = [-10,10]
        self.z_range = [-10,10]

        self.theta_range = [0, 2*np.pi]
        self.angular_dims_start = 0

        self.robot_radius = 0.5

        self.obstacles = []

        self.do_boundary_check = True

        self.num_collision_checks = 0

        self.state_dims = 4
        self.c_space_lower_bound = np.array([0.0, 0.0, 0.0, 0.0])
        self.c_space_upper_bound = np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
    
    def make_state(self, state: np.ndarray) -> AngularNumpyState:
        return AngularNumpyState(
            value=state, 
            angular_dims_start=self.angular_dims_start
        )

    def sample_point(self) -> AngularNumpyState:
        theta_0 = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        theta_1 = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        theta_2 = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        theta_3 = np.random.uniform(low=self.theta_range[0], high=self.theta_range[1])
        return self.make_state(np.array([theta_0, theta_1, theta_2, theta_3]))
    
    def generate_robot_representation(self, state: AngularNumpyState):
        raise NotImplementedError

    def dist(self, state1: AngularNumpyState, state2: AngularNumpyState) -> float:
        return numpystate_distance(state1, state2)

    def is_valid(self, state: AngularNumpyState) -> bool:
        raise NotImplementedError

    def draw_state(self, ax, state, method="rerun"):
        rr.init("ur5 drawing", spawn=True)

        ees = self.forward_kinematics(state)
        cylinder_centers = np.mean(ees, axis=1) # (N, 3)
        direction_vectors = np.diff(ees[:, :, :], axis=1).squeeze() # (N, 3)
        direction_vectors_norms = np.linalg.norm(direction_vectors, axis=1)
        unit_direction_vectors = direction_vectors / direction_vectors_norms # (N, 3)

        z = np.array([0.0, 0.0, 1.0])
        
        rotation_axis = np.cross(z, unit_direction_vectors) # (N, 3)

        # Dot Product Only Preserves the Z Component
        dot_prods = unit_direction_vectors[:, 2]
        
        rotation_magnitude = np.arccos(dot_prods) # (N,)

        rotation_axis_angles=[
            rr.RotationAxisAngle(
                rotation_axis[i],
                rr.Angle(rad=rotation_magnitude[i]),
            )
            for i in range(len(rotation_axis))
        ]

        rr.log(
            "links",
            rr.Cylinders3D(
                lengths=1.0,
                radii=0.1,
                centers=cylinder_centers,
                colors=[[255, 0, 0]],
                rotation_axis_angles=rotation_axis_angles,
                fill_mode=rr.components.FillMode.Solid
            )
        )
        rr.log(
            "ees",
            rr.Points3D(
                ees.reshape(-1, 3),
                colors=[255, 255, 0],
                radii=0.05
            )
        )

    def forward_kinematics(self, state: NumpyState) -> np.ndarray:
        theta0, theta1, theta2, theta3, *_ = state.value
        H_10 = np.array([[np.cos(theta0), -np.sin(theta0), 0.0, 0.0],
                         [np.sin(theta0), np.cos(theta0),  0.0, 0.0],
                         [0.0,            0.0,             1.0, 0.0],
                         [0.0,            0.0,             0.0, 1.0]])

        H_21 = np.array([[np.cos(theta1),  0.0, np.sin(theta1), 0.0],
                         [0.0,             1.0, 0.0,            0.0],
                         [-np.sin(theta1), 0.0, np.cos(theta1), 0.0],
                         [0.0,             0.0, 0.0,            1.0]])

        H_32 = np.array([[np.cos(theta2),  0.0, np.sin(theta2), 1.0],
                         [0.0,             1.0, 0.0,            0.0],
                         [-np.sin(theta2), 0.0, np.cos(theta2), 0.0],
                         [0.0,             0.0, 0.0,            1.0]])

        H_43 = np.array([[np.cos(theta3),  0.0, np.sin(theta3), 1.0],
                         [0.0,             1.0, 0.0,            0.0],
                         [-np.sin(theta3), 0.0, np.cos(theta3), 0.0],
                         [0.0,             0.0, 0.0,            1.0]])

        H_54 = np.array([[1.0, 0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

        homogenous_origin = np.array([0.0, 0.0, 0.0, 1.0])
        transformation_matricies = [H_10, H_21, H_32, H_43, H_54]

        # Start the Cumulative Matrix as the identity matrix
        cum_mat = np.eye(4)

        cum_transformation_matricies: list[np.ndarray] = []

        for i in range(len(transformation_matricies)):
            cum_mat = cum_mat @ transformation_matricies[i]

            # Copy the Cumulative Matrix to avoid overwriting appended results
            cum_transformation_matricies.append(cum_mat.copy())

        cum_transformation_matricies = np.array(cum_transformation_matricies)

        links = cum_transformation_matricies @ homogenous_origin

        # Convert From Homogenous Coordinates
        links = links / links[:, 3:]
        links = links[:, :3]

        # Remove First Link Since it's just a Rotation and will always have endpoints
        # at the origin
        links = links[1:]
        link_ees = np.stack((links[:-1], links[1:]), axis=1)

        return link_ees
    
    def batch_forward_kinematics(
        self, states: np.ndarray, use_optimization: bool = True
    ) -> np.ndarray:
        # states: (N, m)
        # returns: (N, m-1, 2, 3)
        N, m = states.shape

        
        # TODO: Deprecate Unoptimized Version
        if use_optimization:
            thetas0 = states[:, 0]
            thetas1 = states[:, 1]
            thetas2 = states[:, 2]
            thetas3 = states[:, 3]

            zeros = np.zeros(N)
            ones = np.ones(N)
            batch_H_10 = np.array(
                [
                    [np.cos(thetas0), -np.sin(thetas0), zeros, zeros],
                    [np.sin(thetas0), np.cos(thetas0),  zeros, zeros],
                    [zeros,           zeros,            ones,  zeros],
                    [zeros,           zeros,            zeros, ones]
                ]
            )
            batch_H_10 = batch_H_10.transpose(2, 0, 1)

            batch_H_21 = np.array(
                [
                    [np.cos(thetas1),  zeros, np.sin(thetas1), zeros],
                    [zeros,            ones,            zeros, zeros],
                    [-np.sin(thetas1), zeros, np.cos(thetas1), zeros],
                    [zeros,            zeros,           zeros, ones]
                ]
            )
            batch_H_21 = batch_H_21.transpose(2, 0, 1)

            batch_H_32 = np.array(
                [
                    [np.cos(thetas2),  zeros, np.sin(thetas2), ones],
                    [zeros,            ones,            zeros, zeros],
                    [-np.sin(thetas2), zeros, np.cos(thetas2), zeros],
                    [zeros,            zeros,           zeros, ones]
                ]
            )
            batch_H_32 = batch_H_32.transpose(2, 0, 1)

            batch_H_43 = np.array(
                [
                    [np.cos(thetas3),  zeros, np.sin(thetas3), ones],
                    [zeros,            ones,            zeros, zeros],
                    [-np.sin(thetas3), zeros, np.cos(thetas3), zeros],
                    [zeros,            zeros,           zeros, ones]
                ]
            )
            batch_H_43 = batch_H_43.transpose(2, 0, 1)

            batch_H_54 = np.array(
                [
                    [ones,  zeros, zeros, ones],
                    [zeros, ones,  zeros, zeros],
                    [zeros, zeros, ones,  zeros],
                    [zeros, zeros, zeros, ones]
                ]
            )
            batch_H_54 = batch_H_54.transpose(2, 0, 1)

            transformation_matricies = [
                batch_H_10, batch_H_21, batch_H_32, batch_H_43, batch_H_54
            ]
            
            # Start the Cumulative Matrix as the identity matrix
            cum_mat = np.eye(4)
    
            cum_transformation_matricies: list[np.ndarray] = []
    
            for i in range(len(transformation_matricies)):
                cum_mat = cum_mat @ transformation_matricies[i]
    
                # Copy the Cumulative Matrix to avoid overwriting appended results
                cum_transformation_matricies.append(cum_mat.copy())
    
            cum_transformation_matricies = np.array(cum_transformation_matricies)
            cum_transformation_matricies = cum_transformation_matricies.transpose(
                1, 0, 2, 3
            )

            homogenous_origin = np.array([0.0, 0.0, 0.0, 1.0])

            batch_ees = cum_transformation_matricies @ homogenous_origin

            # Convert from homogeneous coordinates
            batch_ees = batch_ees / batch_ees[:, :, 3:]

            # Remove Scaling Factor
            batch_ees = batch_ees[:, :, :3]

            # Remove First Link Since it's just a Rotation and will always have
            # endpoints at the origin
            batch_ees = batch_ees[:, 1:, :]
            batch_link_ees = np.stack((batch_ees[:, :-1], batch_ees[:, 1:]), axis=2)
            return batch_link_ees
        else:
            fks = []
            for state in states:
                fks.append(self.forward_kinematics(self.make_state(state)))
            
            return np.stack(fks, axis=0)

    def batch_get_robot_representations(
        self, states: np.ndarray
    ) -> dict[str, np.ndarray]:
        cylinder_endpoints = self.batch_forward_kinematics(states)

        return {
            'prisms' : np.empty((0, 6)),
            'cylinders' : cylinder_endpoints, 
            'cylinder_radii' : 0.1,
            'points' : np.empty((0, 3)),
            'points_radii' : self.robot_radius
        }

    def sample_configs_ee_target(self, target_ee_position: np.ndarray) -> np.ndarray:
        tolerance = 0.5
        points = self.batch_sample_points(1000000) # (B, n)
        ees = self.batch_forward_kinematics(points) # (B, 3, 2, 3)
        ee_positions = ees[:, -1, -1] # (B, 3)
        
        dist_mat = pairwise_distances(target_ee_position.reshape(1, -1), ee_positions)
        dist_array = dist_mat[0]
        mask = dist_array < tolerance
        final_qs = points[mask]
        return final_qs
    
    def batch_sample_points_around_target(self, targets):
        raise NotImplementedError

if __name__ == "__main__":
    from motion_planning.obstacle_sets import SingleCup
    env = UR5()
    env.set_obstacles(SingleCup())
    env.draw_state(None, env.sample_point())
    env.draw_environment(None)