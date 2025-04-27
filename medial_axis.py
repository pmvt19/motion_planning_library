# from skimage.morphology import medial_axis

from space import PointRobot, PlanarMobileArm
from obstacle_sets import BiasedPassage, RandomSamplePassage, TestSet, ParkingSpace
import matplotlib.pyplot as plt
import numpy as np
import cv2
from circle_approximation import ApproximationSpace

from io import BytesIO
from skimage.morphology import medial_axis
from utils import interpolate_path
from prm import PRM

from biased_rrt import BiasedSamplingRRT
from rrt import RRT
from PIL import Image

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

    ma_img = medial_axis(img, return_distance=False)

    ma_img = ma_img[::-1, :]
    ys, xs = np.where(ma_img == True)

    env.x_range = [0,40]
    env.y_range = [0,10]

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

    return ma_points

class MedialAxisRRT(BiasedSamplingRRT):
    def __init__(self, env, points_bias=0.4, delta=0.5):
        
        # COMPUTE MEDIAL AXIS POINTS WORKSPACE HERE
        ma_points = compute_medial_axis_points(env, percent_kept=0.1)
        # MEDIAL AXIS POINTS TO ENV SPECFIC CONFIGURATIONS

        configs = env.batch_sample_points_around_target(ma_points) # TODO: IMPLEMENT THIS FUNCTION
        super().__init__(env=env, biased_points=configs, points_bias=points_bias, delta=delta)

class MedialAxisPRM(PRM):
    def __init__(self, env, num_samples=1000, num_neighbors=10, edge_dist_radius=None):

        # COMPUTE MEDIAL AXIS POINTS WORKSPACE HERE
        ma_points = compute_medial_axis_points(env, percent_kept=0.1)

        # MEDIAL AXIS POINTS TO ENV SPECFIC CONFIGURATIONS
        configs = env.batch_sample_points_around_target(ma_points) # TODO: IMPLEMENT THIS FUNCTION
        
        super().__init__(env=env, num_samples=num_samples, num_neighbors=num_neighbors, edge_dist_radius=edge_dist_radius, validate_edges=True)

        print("Creating Graph In Initialization")
        self.create_graph(starting_samples=configs)

if __name__ == '__main__':
    seed = np.random.randint(0, 100)
    seed = 53
    print(f"Setting Seed: {seed}")
    np.random.seed(seed)
    env = PointRobot()
    # env = PlanarMobileArm()
    # env.set_obstacles(BiasedPassage(num_walls=2))
    # env.set_obstacles(ParkingSpace())
    env.set_obstacles(RandomSamplePassage(gap_width=0.09))
    # env.set_obstacles(TestSet())
    

    plt.axis('off')
    env.draw_environment(plt.gca())

    fig = plt.gcf()
    width, height = fig.canvas.get_width_height()
    # plt.savefig("imgs/env.png", bbox_inches='tight', pad_inches=0)
    # exit()

    data_obj = BytesIO()
    plt.savefig(data_obj, format='png', bbox_inches='tight', pad_inches=0)
    data_obj.seek(0)

    # img = np.frombuffer()
    # img = np.frombuffer(data_obj).reshape(height, width, 3)
    # img = plt.imread(data_obj)#.reshape(height, width, 3)
    img = Image.open(data_obj).convert("L")
    img = np.array(img)

    # print(data_obj)
    # plt.clf()
    # plt.imshow(img)
    # plt.show()
    # exit()

    env = ApproximationSpace(env, do_overapproximation=False)
    # img = cv2.imread("imgs/env.png", cv2.IMREAD_GRAYSCALE)
    img[:,0] = False
    img[:,-1] = False
    img[0,:] = False
    img[-1,:] = False
    # plt.imshow(img)
    # plt.show()
    print(type(img), img.shape)
    ylen, xlen = img.shape

    ma_img = medial_axis(img, return_distance=False)
    # print(np.unique(ma_img))
    plt.imshow(ma_img)
    plt.show()
    # cv2.imshow('frame', ma_img)

    # xs, ys = np.where(ma_img == True)
    ma_img = ma_img[::-1, :]
    ys, xs = np.where(ma_img == True)

    env.x_range = [0,40]
    # env.x_range = [0,30]
    env.y_range = [0,10]

    # env.x_range = [-15,15]
    # env.y_range = [-15,15]

    scaled_ys = ((ys / ylen) * (env.y_range[1] - env.y_range[0]) + env.y_range[0])
    scaled_xs = (xs / xlen) * (env.x_range[1] - env.x_range[0]) + env.x_range[0]

    xs = scaled_xs
    ys = scaled_ys

    ma_coords = np.hstack((xs.reshape(-1,1), ys.reshape(-1,1)))

    ma_circles = np.hstack((ma_coords, np.zeros((xs.shape[0],1)))).reshape(1, -1, 3)
    validities = env.circles_to_indiv_validity(env.obstacle_circles, ma_circles)

    validities = np.all(validities, axis=2)[0]

    ma_circles = ma_circles[0][validities]

    # env.draw_environment(plt.gca())
    # env.space.draw_environment(plt.gca())
    # plt.scatter(ma_circles[:, 0], ma_circles[:, 1])
    # plt.show()

    ma_points = ma_circles[:, :2]
    idxes = np.random.choice(len(ma_points), size=(int(0.1*len(ma_points))))
    ma_points = ma_points[idxes]

    env.draw_environment(plt.gca())
    env.space.draw_environment(plt.gca())
    plt.scatter(ma_points[:, 0], ma_points[:, 1])
    plt.show()


    print(f"Num Points: {len(ma_points)}")

    # starting_points = []
    # for i in range(len(ma_points)):
    #     print(f"Point: {i}", end='\r')
    #     configs = env.space.sample_configs_ee_target(ma_points[i])
    #     starting_points.append(configs)
    #     # print(configs.shape)
    #     # exit()

    #     # if configs.shape[0] > 0:
    #     #     print("Found Points")

    # prm_starting_configs = np.concatenate(starting_points, axis=0)
    # print(prm_starting_configs.shape)

    prm_starting_configs = ma_points


    start, target = env.sample_valid_point(), env.sample_valid_point()
    start, target = env.sample_valid_point(), env.sample_valid_point()
    print("Start and Target")
    target = env.make_state(np.array([35.0, 5.0]))
    print(start.value, target.value)
    # print(start.value, target.value)
    rrt = MedialAxisRRT(env, prm_starting_configs, points_bias=0.7)
    # rrt = RRT(env)
    path = rrt.search(start, target, max_steps=20000, goal_bias=0.1)
    rrt.draw_tree(plt.gca(), path=path)
    plt.show()

    # plt.clf()
    # env.draw_environment(plt.gca())
    # env.space.draw_environment(plt.gca())

    # for config in prm_starting_configs:
    #     print(config)
    #     env.draw_state(plt.gca(), env.make_state(config))
    # # plt.scatter(ma_points[:, 0], ma_points[:, 1])
    # plt.show()

    # env.edge_validity_delta = 0.005
    # prm = PRM(env=env, num_samples=20, num_neighbors=10, validate_edges=True)
    # prm.create_graph(starting_samples=prm_starting_configs)
    # start, target = env.sample_valid_point(), env.sample_valid_point()

    # path = prm.search(start, target)
    # env.draw_environment(plt.gca())
    # prm.draw(plt.gca())
    # plt.show()

    # path = interpolate_path(path, env, 0.1)
    # env.space.animate_path(path, frame_delay=0.001)