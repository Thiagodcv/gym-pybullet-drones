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

    def test_quaternion_to_rotation_matrix(self):
        model = DroneModel.CF2X
        dynamics = DroneDynamics(model)

        x = np.array([1., 0., 0.])
        y = np.array([0., 1., 0.])
        z = np.array([0., 0., 1.])
        q = np.zeros(4)

        # Rotate around z axis by pi/2
        theta = np.pi / 2
        q[0] = np.cos(theta/2)
        q[1:4] = np.sin(theta/2)*z
        R = dynamics.quaternion_to_rotation_matrix(q)

        epsilon = 1e-5
        self.assertTrue(np.linalg.norm(y - R @ x) < epsilon)
        self.assertTrue(np.linalg.norm(-x - R @ y) < epsilon)
        self.assertTrue(np.linalg.norm(z - R @ z) < epsilon)

        # Rotate around x axis by pi
        theta = np.pi
        q[0] = np.cos(theta / 2)
        q[1:4] = np.sin(theta / 2) * x
        R = dynamics.quaternion_to_rotation_matrix(q)
        self.assertTrue(np.linalg.norm(-y - R @ y) < epsilon)
        self.assertTrue(np.linalg.norm(-z - R @ z) < epsilon)
        self.assertTrue(np.linalg.norm(x - R @ x) < epsilon)

    def test_quaternion_mult(self):
        model = DroneModel.CF2X
        dynamics = DroneDynamics(model)

        x = np.array([1., 0., 0.])
        y = np.array([0., 1., 0.])
        z = np.array([0., 0., 1.])
        q1 = np.zeros(4)
        q2 = np.zeros(4)

        # Rotate around z axis by pi/2
        theta1 = np.pi / 2
        q1[0] = np.cos(theta1/2)
        q1[1:4] = np.sin(theta1/2) * z
        R1 = dynamics.quaternion_to_rotation_matrix(q1)

        # Rotate around x axis by pi
        theta2 = np.pi
        q2[0] = np.cos(theta2/2)
        q2[1:4] = np.sin(theta2/2) * x
        R2 = dynamics.quaternion_to_rotation_matrix(q2)

        q_comp = dynamics.quaternion_mult(q2, q1)
        R_comp = R2 @ R1

        R_comp_from_quat = dynamics.quaternion_to_rotation_matrix(q_comp)
        self.assertTrue(np.linalg.norm(R_comp - R_comp_from_quat) < 1e-5)

    def test_x_ddot(self):
        model = DroneModel.CF2X
        dynamics = DroneDynamics(model)

        x = np.array([1., 0., 0.])
        y = np.array([0., 1., 0.])
        z = np.array([0., 0., 1.])
        q = np.zeros(4)

        # Rotate around y axis (pitch downwards) by pi/2
        theta = np.pi / 2
        q[0] = np.cos(theta/2)
        q[1:4] = np.sin(theta/2) * y
        u = 1000*np.ones(4)

        x_ddot = dynamics.x_ddot(q, u)
        # print(x_ddot)
        self.assertTrue(x_ddot[0] > 0)
        self.assertTrue(x_ddot[1] == 0)
        self.assertTrue(x_ddot[2] == -9.8)

        # Rotate around x axis by -pi/2
        theta = -np.pi / 2
        q[0] = np.cos(theta / 2)
        q[1:4] = np.sin(theta / 2) * x
        u = 1000 * np.ones(4)

        x_ddot = dynamics.x_ddot(q, u)
        # print(x_ddot)
        self.assertTrue(x_ddot[0] == 0)
        self.assertTrue(x_ddot[1] > 0)
        self.assertTrue(x_ddot[2] == -9.8)

        # Rotate around z axis by pi
        theta = np.pi
        q[0] = np.cos(theta / 2)
        q[1:4] = np.sin(theta / 2) * z
        u = 1000 * np.ones(4)

        x_ddot = dynamics.x_ddot(q, u)
        # print(x_ddot)
        self.assertTrue(x_ddot[0] == 0)
        self.assertTrue(x_ddot[1] == 0)
        self.assertTrue(x_ddot[2] > -9.8 and x_ddot[2] < -9.7)

    def test_w_dot(self):
        """
        This test focuses on the torque generated along the xy-plane, while torque along z-axis is zero.
        """
        model = DroneModel.CF2X
        dynamics = DroneDynamics(model)

        # Compute quaternion which represents pi/2 rotation around z-axis.
        theta = np.pi / 2
        z = np.array([0., 0., 1.])
        q = np.zeros(4)
        q[0] = np.cos(theta/2)
        q[1:4] = np.sin(theta/2) * z

        # Angular velocity which represents drone spinning clockwise around z-axis.
        w = np.array([0., 0., -0.05])

        # Input is nonzero only for propellers 1 and 2 which causes drone to pitch downwards.
        u = 1000*np.array([0., 1., 1., 0.])

        # Angular acceleration should have negative x part and zero all other parts.
        w_dot = dynamics.w_dot(q, w, u)

        tol = 1e-10
        # print(w_dot)
        self.assertTrue(w_dot[0] < 0)
        self.assertTrue(np.abs(w_dot[1]) < tol)
        self.assertTrue(np.abs(w_dot[2]) < tol)
