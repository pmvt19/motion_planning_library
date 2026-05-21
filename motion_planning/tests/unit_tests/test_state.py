import unittest

import numpy as np

from motion_planning.tools import AngularNumpyState, NumpyState


class TestNumpyState(unittest.TestCase):
    def test_numpy_state_value_fixed_state(self):
        raw_state = np.array([4.3, 5.7])
        numpy_state = NumpyState(raw_state.copy())
        np.testing.assert_equal(numpy_state.value, raw_state)

    def test_numpy_state_value_random_state(self):
        raw_state = np.random.random(size=(4,))
        numpy_state = NumpyState(raw_state.copy())
        np.testing.assert_equal(numpy_state.value, raw_state)


class TestAngularNumpyState(unittest.TestCase):
    def test_angular_numpy_state_value_fixed_state(self):
        raw_state = np.array([3.3, 20.3, 2.9])
        angular_numpy_state = AngularNumpyState(raw_state.copy(), angular_dims_start=2)
        np.testing.assert_equal(angular_numpy_state.value, raw_state)

    def test_angular_numpy_state_value_random_state(self):
        raw_state = np.random.random(size=(5,))
        angular_numpy_state = AngularNumpyState(raw_state.copy(), angular_dims_start=2)
        np.testing.assert_equal(angular_numpy_state.value, raw_state)

    def test_angular_numpy_state_wrap_around(self):
        raw_state = np.array([5.0, 5.3, 7.5])
        angular_numpy_state = AngularNumpyState(raw_state.copy(), angular_dims_start=2)
        wrapped_raw_state = np.array([5.0, 5.3, 1.2168])
        np.testing.assert_array_almost_equal(
            angular_numpy_state.value, wrapped_raw_state, decimal=4
        )
