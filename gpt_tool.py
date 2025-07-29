import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import LineString, Polygon as ShapelyPolygon

# Robot parameters
link_lengths = [1.0, 1.0]  # Lengths of the two links

# Define obstacles as a list of polygons (list of (x, y) tuples)
obstacles = [
    [(-0.5, 1.0), (0.0, 1.5), (0.5, 1.0), (0.0, 0.5)],
    [(1.5, -0.5), (2.0, 0.0), (1.5, 0.5), (1.0, 0.0)]
]

def forward_kinematics(theta1, theta2):
    """Compute the (x, y) positions of each joint and end effector."""
    x0, y0 = 0, 0
    x1 = x0 + link_lengths[0] * np.cos(theta1)
    y1 = y0 + link_lengths[0] * np.sin(theta1)
    x2 = x1 + link_lengths[1] * np.cos(theta1 + theta2)
    y2 = y1 + link_lengths[1] * np.sin(theta1 + theta2)
    return [(x0, y0), (x1, y1), (x2, y2)]

def check_collision(joint_positions, obstacles):
    """Check if the robot's links collide with any obstacles."""
    link1 = LineString([joint_positions[0], joint_positions[1]])
    link2 = LineString([joint_positions[1], joint_positions[2]])
    for obs in obstacles:
        poly = ShapelyPolygon(obs)
        if link1.intersects(poly) or link2.intersects(poly):
            return True
    return False

def plot_workspace(theta1, theta2, obstacles):
    """Plot the robot in the workspace."""
    joint_positions = forward_kinematics(theta1, theta2)
    collision = check_collision(joint_positions, obstacles)

    fig, ax = plt.subplots()
    # Plot links
    ax.plot([joint_positions[0][0], joint_positions[1][0]],
            [joint_positions[0][1], joint_positions[1][1]], 'b-', linewidth=5)
    ax.plot([joint_positions[1][0], joint_positions[2][0]],
            [joint_positions[1][1], joint_positions[2][1]], 'r-', linewidth=5)
    # Plot joints
    ax.plot(joint_positions[0][0], joint_positions[0][1], 'ko')  # Base
    ax.plot(joint_positions[1][0], joint_positions[1][1], 'ko')  # Joint
    ax.plot(joint_positions[2][0], joint_positions[2][1], 'go')  # End effector
    # Plot obstacles
    for obs in obstacles:
        polygon = Polygon(obs, closed=True, color='gray', alpha=0.5)
        ax.add_patch(polygon)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Workspace - ' + ('Collision' if collision else 'No Collision'))
    plt.grid(True)
    plt.show()

def plot_cspace(obstacles, resolution=100):
    """Plot the configuration space."""
    theta1_vals = np.linspace(-np.pi, np.pi, resolution)
    theta2_vals = np.linspace(-np.pi, np.pi, resolution)
    cspace = np.zeros((resolution, resolution))

    for i, t1 in enumerate(theta1_vals):
        for j, t2 in enumerate(theta2_vals):
            joints = forward_kinematics(t1, t2)
            if check_collision(joints, obstacles):
                cspace[j, i] = 1  # Note: rows correspond to theta2, columns to theta1

    fig, ax = plt.subplots()
    ax.imshow(cspace, extent=[-np.pi, np.pi, -np.pi, np.pi], origin='lower', cmap='Greys')
    ax.set_xlabel('Theta1')
    ax.set_ylabel('Theta2')
    ax.set_title('Configuration Space')
    plt.grid(True)
    plt.show()

# Example usage
theta1 = np.deg2rad(45)
theta2 = np.deg2rad(30)
plot_workspace(theta1, theta2, obstacles)
plot_cspace(obstacles)