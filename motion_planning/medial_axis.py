import matplotlib.pyplot as plt
import numpy as np
import cv2

from io import BytesIO
from PIL import Image
from skimage.morphology import medial_axis

from space import PointRobot, PlanarMobileArm, PolygonalRobot
from obstacle_sets import BiasedPassage, RandomSamplePassage, TestSet, ParkingSpace
from circle_approximation import ApproximationSpace
from utils import interpolate_path, smooth_path
from prm import PRM
from biased_rrt import BiasedSamplingRRT
from rrt import RRT


# Medial Axis Can Either Guide PRM (Use as starting vertices) or RRT (Biased Sampling)

def compute_medial_axis_points(env : ApproximationSpace, percent_kept=0.1):
    plt.axis('off')
    env.space.draw_environment(plt.gca())

    data_obj = BytesIO()
    plt.savefig(data_obj, format='png', bbox_inches='tight', pad_inches=0)
    data_obj.seek(0)

    img = Image.open(data_obj).convert("L")
    img = np.array(img)

    img[:,0] = False
    img[:,-1] = False
    img[0,:] = False
    img[-1,:] = False

    ylen, xlen = img.shape

    ma_img_unflipped = medial_axis(img, return_distance=False)

    ma_img = ma_img_unflipped[::-1, :]
    ys, xs = np.where(ma_img == True)

    # HACK: HARDCODED
    env.x_range = [0,40]
    env.y_range = [0,10]
    # HACK: HARDCODED

    scaled_ys = ((ys / ylen) * (env.y_range[1] - env.y_range[0]) + env.y_range[0])
    scaled_xs = (xs / xlen) * (env.x_range[1] - env.x_range[0]) + env.x_range[0]

    xs = scaled_xs
    ys = scaled_ys

    ma_coords = np.hstack((xs.reshape(-1,1), ys.reshape(-1,1)))

    ma_circles = np.hstack((ma_coords, np.zeros((xs.shape[0],1)))).reshape(1, -1, 3)
    validities = env.circles_to_indiv_validity(env.obstacle_circles, ma_circles)

    validities = np.all(validities, axis=2)[0]

    ma_circles = ma_circles[0][validities]

    ma_points = ma_circles[:, :2]
    idxes = np.random.choice(len(ma_points), size=(int(percent_kept*len(ma_points))))
    ma_points = ma_points[idxes]
    print(f"Num MedialAxis Points: {len(ma_points)}")
    plt.close()
    return ma_points, ma_img

class MedialAxisRRT(BiasedSamplingRRT):
    def __init__(self, env, points_bias=0.4, delta=0.5):
        assert (isinstance(env, ApproximationSpace)), "MedialAxisRRT is only compatible with ApproximationSpace Environments"
        
        # COMPUTE MEDIAL AXIS POINTS WORKSPACE HERE
        ma_points, ma_img = compute_medial_axis_points(env, percent_kept=0.1)

        self.ma_img = ma_img
        self.ma_points = ma_points
        # MEDIAL AXIS POINTS TO ENV SPECFIC CONFIGURATIONS

        configs = env.batch_sample_points_around_target(ma_points) # TODO: IMPLEMENT THIS FUNCTION
        # configs = ma_points
        super().__init__(env=env, biased_points=configs, points_bias=points_bias, delta=delta)
    
    def show_medial_axis(self):
        plt.clf()
        plt.imshow(self.ma_img)
        plt.gca().invert_yaxis()
        plt.show()

class MedialAxisPRM(PRM):
    def __init__(self, env, num_samples=1000, num_neighbors=10, edge_dist_radius=None):
        assert (isinstance(env, ApproximationSpace)), "MedialAxisPRM is only compatible with ApproximationSpace Environments"

        # COMPUTE MEDIAL AXIS POINTS WORKSPACE HERE
        ma_points, ma_img = compute_medial_axis_points(env, percent_kept=0.1)

        self.ma_img = ma_img
        self.ma_points = ma_points

        # MEDIAL AXIS POINTS TO ENV SPECFIC CONFIGURATIONS
        configs = env.batch_sample_points_around_target(ma_points) # TODO: IMPLEMENT THIS FUNCTION

        self.starting_configs = configs
        
        super().__init__(env=env, num_samples=num_samples, num_neighbors=num_neighbors, edge_dist_radius=edge_dist_radius, validate_edges=True)

        # print("Creating Graph In Initialization")
        # self.create_graph(starting_samples=configs)
    def create_graph(self, starting_samples=[]):
        assert (len(starting_samples) == 0), "Starting Samples Doesn't Work Right Now"
        super().create_graph(starting_samples=self.starting_configs)

if __name__ == '__main__':
    seed = np.random.randint(0, 100)
    seed = 53
    print(f"Setting Seed: {seed}")
    np.random.seed(seed)
    # env = PointRobot()
    env = PolygonalRobot()

    # env = PlanarMobileArm()
    # env.set_obstacles(BiasedPassage(num_walls=2))
    # env.set_obstacles(ParkingSpace())

    # env.set_obstacles(RandomSamplePassage(gap_width=0.09))
    env.set_obstacles(RandomSamplePassage(gap_width=1.5))
    env = ApproximationSpace(env, do_overapproximation=False)

    # start, target = env.sample_valid_point(), env.sample_valid_point()
    # start, target = env.sample_valid_point(), env.sample_valid_point()
    print("Start and Target")
    start = env.make_state(np.array([5.0, 5.0, 0.0]))
    target = env.make_state(np.array([35.0, 5.0, 0.0]))
    print(start.value, target.value)
    # print(start.value, target.value)
    # rrt = MedialAxisRRT(env, prm_starting_configs, points_bias=0.7)
    # rrt = MedialAxisRRT(env, points_bias=0.7)
    # # rrt = MedialAxisRRT(env, points_bias=0.8)
    
    # # rrt = RRT(env)
    # path = rrt.search(start, target, max_steps=20000, goal_bias=0.1)
    # rrt.draw_tree(plt.gca(), path=path)
    # env.space.draw_environment(plt.gca())
    # plt.show()

    # rrt.show_medial_axis()

    # env.animate_path(path, frame_delay=0.1)

    prm = MedialAxisPRM(env)
    prm.create_graph()
    path = prm.search(start, target)
    path = smooth_path(env, path)
    path = interpolate_path(path, env, 0.1)
    env.space.animate_path(path, frame_delay=0.001)
