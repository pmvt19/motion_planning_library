import unittest

import numpy as np

from motion_planning.tools import KinodynamicPath, NumpyState, Path


class TestPath(unittest.TestCase):
    def test_path_only_accepts_list(self):
        # Path Cannot Be Instantiated with an Integer
        with self.assertRaises(AssertionError):
            Path(path=4)

        # Path Cannot Be Instantiated with a float
        with self.assertRaises(AssertionError):
            Path(path=4.0)

        # Path Cannot Be Instantiated with a NumpyState
        with self.assertRaises(AssertionError):
            Path(path=NumpyState(np.array([0.0, 0.0])))

        path = [NumpyState(np.array([0.0, 0.0])), NumpyState(np.array([1.0, 1.0]))]

        Path(path=path)

    def test_path_can_be_empty(self):
        Path(path=[])


class TestKinodynamicPath(unittest.TestCase):
    def test_kinodynamic_path_accepts_valid_path_controls_and_dt(self):
        path = [
            NumpyState(np.array([0.0, 0.0, 0.0, 0.0])),
            NumpyState(np.array([1.0, 1.0, 0.5, -0.2])),
        ]
        controls = [np.array([0.1, 0.0]), np.array([0.0, -0.1])]
        dt = 0.05

        kinodynamic_path = KinodynamicPath(path=path, controls=controls, dt=dt)

        self.assertEqual(len(kinodynamic_path), 2)
        self.assertIs(kinodynamic_path.path, path)
        self.assertIs(kinodynamic_path.controls, controls)
        self.assertEqual(kinodynamic_path.dt, dt)
        self.assertTrue(np.array_equal(kinodynamic_path[0].value, path[0].value))

    def test_kinodynamic_path_defaults_controls_and_dt(self):
        path = [NumpyState(np.array([0.0, 0.0, 0.0, 0.0]))]
        kinodynamic_path = KinodynamicPath(path=path)

        self.assertEqual(len(kinodynamic_path), 1)
        self.assertEqual(kinodynamic_path.controls, [])
        self.assertIsNone(kinodynamic_path.dt)

    def test_kinodynamic_path_requires_path_to_be_list(self):
        with self.assertRaises(AssertionError):
            KinodynamicPath(path=NumpyState(np.array([0.0, 0.0, 0.0, 0.0])))
