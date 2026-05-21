import unittest

import numpy as np

from motion_planning.tools import NumpyState, AngularNumpyState
from motion_planning.utils import numpystate_distance, euclidean_distance, angular_distance

class TestUtils(unittest.TestCase):
    def test_euclidean_distance(self):
        start = np.array([32.42, 23.42])
        end = np.array([74.35, 24.53])

        distance = euclidean_distance(start, end)
        self.assertTrue(isinstance(distance, float))
        self.assertAlmostEqual(distance, 41.94468, places=4)

    def test_angular_distance(self):
        start = np.array([np.pi/2, np.pi])
        end = np.array([np.pi/4, np.pi/3])

        distance = angular_distance(start, end)

        self.assertTrue(isinstance(distance, np.ndarray))
        self.assertTrue(np.all(np.isclose(distance, np.array([0.78539816, 2.0943951]))))

    # def test_angular_distance_wrap_around(self):
    #     assert (False), "Ensure angular wrap around is tested here" # TODO: See Message in Assertion    

    def test_numpy_state_distance(self):
        start = NumpyState(np.array([3.2, 5.6]))
        end = NumpyState(np.array([7.8, 2.6]))

        distance = numpystate_distance(start, end)

        self.assertTrue(isinstance(distance, float))
        self.assertAlmostEqual(distance, 5.49181, places=4)

    def test_angular_numpy_state_distance(self):
        # AngularNumpyState Tests
        start = AngularNumpyState(np.array([3.2, 5.6, 0.2]), angular_dims_start=2)
        end = AngularNumpyState(np.array([7.8, 2.6, 5.6]), angular_dims_start=2)

        distance = numpystate_distance(start, end)
        self.assertTrue(isinstance(distance, float))
        self.assertAlmostEqual(distance, 5.49181, places=4)


def test_numpystate_distance():
    # NumpyState Tests
    start = NumpyState(np.array([3.2, 5.6]))
    end = NumpyState(np.array([7.8, 2.6]))

    distance = numpystate_distance(start, end)
    print(distance)
    assert (isinstance(distance, float))
    assert (np.isclose(distance, 5.49181))

    # AngularNumpyState Tests
    start = AngularNumpyState(np.array([3.2, 5.6, 0.2]), angular_dims_start=2)
    end = AngularNumpyState(np.array([7.8, 2.6, 5.6]), angular_dims_start=2)
    # start = AngularNumpyState(np.array([3.2, 5.6, 0.2, 0.0]), angular_dims_start=2)
    # end = AngularNumpyState(np.array([7.8, 2.6, 5.6, 0.0]), angular_dims_start=2)

    distance = numpystate_distance(start, end)
    print(distance)
    assert (isinstance(distance, float))
    assert (np.isclose(distance, 5.49181))

# test_euclidean_distance()
# test_angular_distance()
# test_numpystate_distance()