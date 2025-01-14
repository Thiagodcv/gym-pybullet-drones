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

    def test_w_dot_xy_plane(self):
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

    def test_w_dot_zaxis(self):
        """
        This test focuses on the torque generated along the z-axis (yaw), while torque along the xy-plane is zero.
        """
        model = DroneModel.CF2X
        dynamics = DroneDynamics(model)

        # Compute quaternion which represents pi/2 rotation around z-axis.
        theta = -np.pi / 2
        z = np.array([0., 0., 1.])
        q = np.zeros(4)
        q[0] = np.cos(theta/2)
        q[1:4] = np.sin(theta/ 2) * z

        # Angular velocity which represents drone not spinning along any axis.
        w = np.array([0., 0., 0.])

        # Input is nonzero only for propellers 1 and 3 which causes drone to spin along z-axis (counter-clockwise)
        u = 1000 * np.array([0., 1., 0., 1.])

        # Angular acceleration should have positive z part and zero all other parts.
        w_dot = dynamics.w_dot(q, w, u)

        tol = 1e-10
        # print(w_dot)
        self.assertTrue(np.abs(w_dot[0]) < tol)
        self.assertTrue(np.abs(w_dot[1]) < tol)
        self.assertTrue(w_dot[2] > 0)

        # Input is nonzero only for propellers 0 and 2 which causes drone to spin along z-axis (clockwise)
        u = 1000 * np.array([1., 0., 1., 0.])

        # Angular acceleration should have negative z part and zero all other parts.
        w_dot = dynamics.w_dot(q, w, u)

        tol = 1e-10
        # print(w_dot)
        self.assertTrue(np.abs(w_dot[0]) < tol)
        self.assertTrue(np.abs(w_dot[1]) < tol)
        self.assertTrue(w_dot[2] < 0)

    def test_q_dot(self):
        """
        Ensures q_dot runs without crashing and leads to intuitive results.
        """
        model = DroneModel.CF2X
        dynamics = DroneDynamics(model)

        q = np.array([1., 0., 0., 0.])  # Body frame initially equals global frame
        w = np.array([0., 0., 1.])  # Rigid body spinning along its z-axis counter-clockwise
        q_dot = dynamics.q_dot(q, w)
        q2 = q + (1e-2)*q_dot  # Estimate quaternion after small interval of time
        q2 = q2 / np.linalg.norm(q2)

        R2 = dynamics.quaternion_to_rotation_matrix(q2)  # Rotation matrix at next timestep
        x = np.array([1., 0., 0.])
        rot_vec = R2 @ x

        # Ensure x_axis in body frame rotated as expected
        # print(rot_vec)
        self.assertTrue(rot_vec[0] > 0)
        self.assertTrue(rot_vec[1] > 0)
        self.assertTrue(np.abs(rot_vec[2]) < 1e-5)

    def test_compute_dynamics(self):
        """
        TODO: Come back to this test later if any issues crop up.
        Ensures compute_dynamics runs without crashing and leads to intuitive results.
        """
        model = DroneModel.CF2X
        dynamics = DroneDynamics(model)

        # Compute quaternion which represents pi/2 rotation around z-axis.
        theta = np.pi / 2
        z = np.array([0., 0., 1.])
        q = np.zeros(4)
        q[0] = np.cos(theta / 2)
        q[1:4] = np.sin(theta / 2) * z

        # Angular velocity which represents drone spinning counter-clockwise along z-axis
        w = np.array([0., 0., 1])

        # Only propellers 0 and 2 spin causing drone to rotate clockwise
        u = 1000 * np.array([1., 0., 1., 0.])

        # Current position at (1, 1, 1) in global coords
        x = np.ones(3)

        # Current velocty is upwards
        v = np.array([0., 0., 2.])

        # Get state_dot
        state_dot1 = dynamics.compute_dynamics(v, q, w, u)
        print(state_dot1)

        state = np.zeros(13)
        state[0:3] = x
        state[3:6] = v
        state[6:10] = q
        state[10:13] = w

        state_dot2 = dynamics.state_dot(state, u)
        print(state_dot2)

        tol = 1e-7
        self.assertTrue(np.linalg.norm(state_dot1 - state_dot2) < tol)

    def test_forward_euler(self):
        """
        Ensures forward_euler runs without crashing and leads to intuitive results.
        """
        model = DroneModel.CF2X
        dynamics = DroneDynamics(model)

        # Parameters for forward euler
        h = 0.1
        n = 10

        # Compute quaternion which represents pi/2 rotation around z-axis.
        theta = np.pi / 2
        z = np.array([0., 0., 1.])
        q = np.zeros(4)
        q[0] = np.cos(theta / 2)
        q[1:4] = np.sin(theta / 2) * z

        # Angular velocity which represents zero spin
        w = np.array([0., 0., 0.])

        # Only propellers 0 and 2 spin slightly faster causing drone to rotate clockwise
        P = np.sqrt(dynamics.m*dynamics.g/(4*dynamics.kf))
        u = np.array([P*1.001, P, P*1.001, P])
        inputs = np.vstack([u] * n)

        # Current position at (1,1,1)
        x = np.ones(3)

        # Current velocity is zero
        v = np.array([0., 0., 0.])

        # Create state_init
        state_init = np.zeros(13)
        state_init[0:3] = x
        state_init[3:6] = v
        state_init[6:10] = q
        state_init[10:13] = w

        state_traj = dynamics.forward_euler(state_init, inputs, n, h=h)
        x = np.array([1., 0., 0.])
        for i in range(n):
            print("------- iteration ", i, " -------")
            print("xyz: ", state_traj[i, 0:3])
            print("w: ", state_traj[i, 10:13])
            q = state_traj[i, 6:10]
            R = dynamics.quaternion_to_rotation_matrix(q)
            print("R@X: ", R@x)
        