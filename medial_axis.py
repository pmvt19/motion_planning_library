# from skimage.morphology import medial_axis

from space import PointRobot
from obstacle_sets import BiasedPassage, RandomSamplePassage, TestSet
import matplotlib.pyplot as plt
import numpy as np
import cv2
from circle_approximation import ApproximationSpace

from io import BytesIO
from skimage.morphology import medial_axis

if __name__ == '__main__':
    np.random.seed(0)
    env = PointRobot()
    # env.set_obstacles(BiasedPassage(num_walls=2))
    env.set_obstacles(RandomSamplePassage())
    # env.set_obstacles(TestSet())
    env = ApproximationSpace(env, do_overapproximation=True)

    # plt.axis('off')
    # env.draw_environment(plt.gca())
    # plt.savefig("imgs/env.png", bbox_inches='tight', pad_inches=0)

    # data_obj = BytesIO()
    # plt.savefig(data_obj, format='png')


    img = cv2.imread("imgs/env.png", cv2.IMREAD_GRAYSCALE)
    img[:,0] = False
    img[:,-1] = False
    img[0,:] = False
    img[-1,:] = False
    # plt.imshow(img)
    # plt.show()
    ylen, xlen = img.shape

    ma_img = medial_axis(img, return_distance=False)
    # print(np.unique(ma_img))
    plt.imshow(ma_img)
    plt.show()
    # cv2.imshow('frame', ma_img)

    # TODO: REMOVE POINTS WITHIN OBSTACLES (USE CIRCLE APPROX SPACE TECHNIQUES)

    # xs, ys = np.where(ma_img == True)
    ys, xs = np.where(ma_img == True)

    env.x_range = [0,40]
    env.y_range = [0,10]

    # HACK: NEED TO PROPERLY FORM THIS MATHEMATICALLY
    scaled_ys = env.y_range[1] - ((ys / ylen) * (env.y_range[1] - env.y_range[0]) + env.y_range[0])
    scaled_xs = (xs / xlen) * (env.x_range[1] - env.x_range[0]) + env.x_range[0]

    xs = scaled_xs
    ys = scaled_ys

    ma_coords = np.hstack((xs.reshape(-1,1), ys.reshape(-1,1)))

    ma_circles = np.hstack((ma_coords, np.zeros((xs.shape[0],1)))).reshape(1, -1, 3)
    validities = env.circles_to_indiv_validity(env.obstacle_circles, ma_circles)

    validities = np.all(validities, axis=2)[0]

    ma_circles = ma_circles[0][validities]

    env.draw_environment(plt.gca())
    env.space.draw_environment(plt.gca())
    plt.scatter(ma_circles[:, 0], ma_circles[:, 1])
    plt.show()