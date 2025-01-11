import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel


class DroneDynamics(object):
    """
    Implements 1st order nonlinear ODE which describes dynamics of a quadcopter.
    First implementation will focus on cf2x drone, and might be extended to other models later.
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

        # Body coordinates of propellers
        self.prop_body_coords = np.zeros((4, 3))
        self.prop_body_coords[0, :] = self._get_urdf_parameter('prop0_body_xyz')
        self.prop_body_coords[1, :] = self._get_urdf_parameter('prop1_body_xyz')
        self.prop_body_coords[2, :] = self._get_urdf_parameter('prop2_body_xyz')
        self.prop_body_coords[3, :] = self._get_urdf_parameter('prop3_body_xyz')

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

    def compute_dynamics(self, x, v, q, w):
        """
        Computes the continuous-time rigid body dynamics function evaluated at a (x, v, q, w) tuple.

        Parameters
        ----------
        x : ndarray
            the current position of the drone in the global frame.
        v : ndarray
            the current velocity of the drone in the global frame.
        q : ndarray
            the unit quaternion representing the orientation of the drone's body frame relative to the global frame.
        w : ndarray
            angular momentum of the drone's body frame in the global frame.
        """
        pass
