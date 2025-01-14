import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel


class DroneDynamics(object):
    """
    Implements 1st order nonlinear ODE which describes dynamics of a quadcopter.
    First implementation will focus on cf2x drone, and might be extended to other models later.

    NOTE: in this class assume first element of any quaternion is its scalar.
    """

    def __init__(self, drone_model, g: float = 9.8):
        """
        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.
        """
        self.drone_model = drone_model
        """DroneModel: The type of drone to control."""
        self.m = self._get_urdf_parameter('m')
        """float: Mass of drone."""
        self.kf = self._get_urdf_parameter('kf')
        """float: The coefficient converting RPMs into thrust."""
        self.km = self._get_urdf_parameter('km')
        """float: The coefficient converting RPMs into torque."""

        self.g = g
        # Body coordinates of propellers
        self.prop_body_coords = np.zeros((4, 3))
        self.prop_body_coords[0, :] = self._get_urdf_parameter('prop0_body_xyz')
        self.prop_body_coords[1, :] = self._get_urdf_parameter('prop1_body_xyz')
        self.prop_body_coords[2, :] = self._get_urdf_parameter('prop2_body_xyz')
        self.prop_body_coords[3, :] = self._get_urdf_parameter('prop3_body_xyz')

        # Define inertia matrix
        ixx = self._get_urdf_parameter('ixx')
        iyy = self._get_urdf_parameter('iyy')
        izz = self._get_urdf_parameter('izz')
        self.I_body = np.zeros((3, 3))
        self.I_body[0, 0] = ixx
        self.I_body[1, 1] = iyy
        self.I_body[2, 2] = izz

    def _get_urdf_parameter(self, parameter_name: str):
        """
        Reads a parameter from a drone's URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to read.

        Returns
        -------
        float
            The value of the parameter.

        """
        # Get the XML tree of the drone model to control
        urdf = self.drone_model.value + ".urdf"
        path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+urdf)
        urdf_tree = etxml.parse(path).getroot()

        # Find and return the desired parameter
        if parameter_name == 'm':
            return float(urdf_tree[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(urdf_tree[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff' 'prop_radius', \
                                'drag_coeff_xy', 'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3']:
            return float(urdf_tree[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(urdf_tree[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            collision_shape_offsets = [float(s) for s in urdf_tree[1][2][0].attrib['xyz'].split(' ')]
            return collision_shape_offsets[2]
        elif parameter_name == 'prop0_body_xyz':
            return np.fromstring(urdf_tree[2][0][0].attrib['xyz'], sep=" ")
        elif parameter_name == 'prop1_body_xyz':
            return np.fromstring(urdf_tree[4][0][0].attrib['xyz'], sep=" ")
        elif parameter_name == 'prop2_body_xyz':
            return np.fromstring(urdf_tree[6][0][0].attrib['xyz'], sep=" ")
        elif parameter_name == 'prop3_body_xyz':
            return np.fromstring(urdf_tree[8][0][0].attrib['xyz'], sep=" ")

    def state_dot(self, state, u):
        """
        Computes the time derivative of the state vector using the dynamics function.

        Parameters
        ----------
        state : ndarray
            The 13-dimensional state vector.
        u : ndarray
            The 4-dimensional control input. Namely, the RPM of each propeller.

        Returns
        -------
        ndarray
            The time derivative of the state vector.
        """
        v = state[3:6]
        q = state[6:10]
        w = state[10:13]
        return self.compute_dynamics(v, q, w, u)

    def compute_dynamics(self, v, q, w, u):
        """
        Computes the continuous-time rigid body dynamics function evaluated at a (x, v, q, w) tuple.

        Parameters
        ----------
        v : ndarray
            The current velocity of the drone in the global frame.
        q : ndarray
            The unit quaternion representing the orientation of the drone's body frame relative to the global frame.
            Of the form q = [s, v] where s is the scalar, v is the vector.
        w : ndarray
            Angular momentum of the drone's body frame in the global frame.
        u : ndarray
            The current control input. Namely, the RPM of each propeller.

        Returns
        -------
        ndarray
            The time derivative of the state (13 dimensional).
        """
        state_dot = np.zeros(13)

        # Time derivative of global position
        state_dot[0:3] = v

        # Time derivative of global velocity
        state_dot[3:6] = self.x_ddot(q, u)

        # Time derivative of the quaternion (rotation from local to global frame)
        state_dot[6:10] = self.q_dot(q, w)

        # Time derivative of (global) angular velocity
        state_dot[10:13] = self.w_dot(q, w, u)

        return state_dot

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        Computes the rotation matrix corresponding to a given unit quaternion.

        Parameters
        ----------
        q : ndarray
            the unit quaternion representing the orientation of the drone's body frame relative to the global frame.
            Of the form q = [s, v] where s is the scalar, v is the vector.

        Returns
        -------
        ndarray
            the 3x3 rotation matrix.
        """
        s = q[0]
        v = q[1:4]

        R = np.zeros((3, 3))
        R[0, :] = 1 - 2*v[1]**2 - 2*v[2]**2, 2*v[0]*v[1] - 2*s*v[2], 2*v[0]*v[2] + 2*s*v[1]
        R[1, :] = 2*v[0]*v[1] + 2*s*v[2], 1 - 2*v[0]**2 - 2*v[2]**2, 2*v[1]*v[2] - 2*s*v[0]
        R[2, :] = 2*v[0]*v[2] - 2*s*v[1], 2*v[1]*v[2] + 2*s*v[0], 1 - 2*v[0]**2 - 2*v[1]**2
        return R

    @staticmethod
    def quaternion_mult(q1, q2):
        """
        Multiplies quaternions q1 and q2 like q1 * q2. Note that order matters here.

        Parameters
        ----------
        q1, q2 : ndarray
            two quaternions.

        Returns
        -------
        ndarray
            the resulting quaternion.
        """
        s1 = q1[0]
        s2 = q2[0]
        v1 = q1[1:4]
        v2 = q2[1:4]

        q_result = np.zeros(4)
        q_result[0] = s1*s2 - v1.T @ v2
        q_result[1:4] = s1*v2 + s2*v1 + np.cross(v1, v2)
        return q_result

    def x_ddot(self, q, u):
        """
        Compute linear acceleration of the rigid body in world space.

        Parameters
        ----------
        q : ndarray
            the unit quaternion representing the orientation of the drone's body frame relative to the global frame.
            Of the form q = [s, v] where s is the scalar, v is the vector.
        u : ndarray
            the current control input. Namely, the RPM of each propeller.

        Returns
        -------
        ndarray
        """
        f_vec = self.kf * u**2  # vector of forces (i.e. thrust) as a result of propeller RPM
        R = self.quaternion_to_rotation_matrix(q)

        total_force_local = np.array([0., 0., np.sum(f_vec)])
        total_force_global = R @ total_force_local - np.array([0., 0., self.m * self.g])

        return total_force_global / self.m

    def w_dot(self, q, w, u):
        """
        Compute the angular acceleration of the rigid body in world space.

        Parameters
        ----------
        q : ndarray
            the unit quaternion representing the orientation of the drone's body frame relative to the global frame.
            Of the form q = [s, v] where s is the scalar, v is the vector.
        w : ndarray
            angular momentum of the drone's body frame in the global frame.
        u : ndarray
            the current control input. Namely, the RPM of each propeller.

        Returns
        -------
        ndarray
            The time derivative of angular velocity w.
        """
        R = self.quaternion_to_rotation_matrix(q)

        # Compute torque along x-y plane (in body frame)
        torque_xy_local = np.zeros(3)
        for i in range(4):
            force_local = np.array([0., 0., self.kf * u[i]**2])
            torque_xy_local += np.cross(self.prop_body_coords[i, :], force_local)
        torque_xy_global = R @ torque_xy_local

        # Compute torque along z axis (in body frame)
        torque_z_local = np.zeros(3)
        for i in range(4):
            force_local = np.array([0., 0., self.kf * u[i]**2])
            torque_z_local += (-1)**(i+1) * force_local
        torque_z_global = R @ torque_z_local

        # Compute total torque
        torque_global = torque_xy_global + torque_z_global

        # Global inertial matrix (2-39)
        I = R @ self.I_body @ R.T
        w_dot = np.linalg.inv(I) @ (torque_global - np.cross(w, I @ w))
        return w_dot

    def q_dot(self, q, w):
        """
        Compute the time derivative of the quaternion vector.

        Parameters
        ----------
        q : ndarray
            The unit quaternion representing the orientation of the drone's body frame relative to the global frame.
            Of the form q = [s, v] where s is the scalar, v is the vector.
        w : ndarray
            Angular momentum of the drone's body frame in the global frame.

        Returns
        -------
        ndarray
            The time derivative of the quaternion vector.
        """
        w_quat = np.zeros(4)
        w_quat[1:4] = w
        q_dot = 0.5*self.quaternion_mult(w_quat, q)
        return q_dot

    def forward_euler(self, state_init, inputs, n, h=0.01):
        """
        Estimate the trajectory of the rigid body dynamics using a forward Euler approach.

        Parameters
        ---------
        state_init : ndarray
            The initial state of the system.
        inputs : ndarray
            The control input for each timestep t=0,1,...,n-1 in an (n+1, 4)-sized array.
        n : Int
            The number of time steps.
        h : float, optional
            The length of time between each time step.

        Returns
        -------
        ndarray
            The estimated trajectory of the system in an (n+1, 13)-sized array (the first row is state_init).
        """
        state_traj = np.zeros((n+1, 13))
        state_traj[0, :] = state_init

        for i in range(0, n):
            state_traj[i+1, :] = state_traj[i, :] + h*self.state_dot(state_traj[i, :], inputs[i, :])

        return state_traj

    def rk4(self, state_init, inputs, n, h=0.01):
        """
        Estimate the trajectory of the rigid body dynamics using a forward Euler approach.

        NOTE: here we assume u(t + h/2) = u(t).

        Parameters
        ---------
        state_init : ndarray
            The initial state of the system.
        inputs : ndarray
            The control input for each timestep t=0,1,...,n-1 in an (n+1, 4)-sized array.
        n : Int
            The number of time steps.
        h : float, optional
            The length of time between each time step.

        Returns
        -------
        ndarray
            The estimated trajectory of the system in an (n+1, 13)-sized array (the first row is state_init).
        """
        state_traj = np.zeros((n+1, 13))
        state_traj[0, :] = state_init

        for i in range(0, n):
            state = state_traj[i, :]
            u = inputs[i, :]

            k1 = state
            k2 = state + h/2*self.state_dot(k1, u)
            k3 = state + h/2*self.state_dot(k2, u)
            k4 = state + h*self.state_dot(k3, u)

            state_traj[i + 1, :] = state + h/6*(self.state_dot(k1, u) +
                                                2*self.state_dot(k2, u) +
                                                2*self.state_dot(k3, u) +
                                                self.state_dot(k4, u))
        return state_traj
