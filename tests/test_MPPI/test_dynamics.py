from unittest import TestCase
import numpy as np

from gym_pybullet_drones.control.MPPI.dynamics import DroneDynamics
from gym_pybullet_drones.utils.enums import DroneModel


class TestDynamics(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_urdf_parameter_prop(self):
        model = DroneModel.CF2X
        dynamics = DroneDynamics(model)

        epsilon = 1e-5
        self.assertTrue(
            np.linalg.norm(dynamics._get_urdf_parameter('prop0_body_xyz') - np.array([0.028, -0.028, 0.0])) < epsilon)
        self.assertTrue(
            np.linalg.norm(dynamics._get_urdf_parameter('prop1_body_xyz') - np.array([-0.028, -0.028, 0.0])) < epsilon)
        self.assertTrue(
            np.linalg.norm(dynamics._get_urdf_parameter('prop2_body_xyz') - np.array([-0.028, 0.028, 0.0])) < epsilon)
        self.assertTrue(
            np.linalg.norm(dynamics._get_urdf_parameter('prop3_body_xyz') - np.array([0.028, 0.028, 0.0])) < epsilon)

        self.assertTrue(
            np.linalg.norm(dynamics.prop_body_coords[0, :] - np.array([0.028, -0.028, 0.0])) < epsilon)
        self.assertTrue(
            np.linalg.norm(dynamics.prop_body_coords[1, :] - np.array([-0.028, -0.028, 0.0])) < epsilon)
        self.assertTrue(
            np.linalg.norm(dynamics.prop_body_coords[2, :] - np.array([-0.028, 0.028, 0.0])) < epsilon)
        self.assertTrue(
            np.linalg.norm(dynamics.prop_body_coords[3, :] - np.array([0.028, 0.028, 0.0])) < epsilon)
