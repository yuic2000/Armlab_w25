"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm

D2R = np.pi / 180.0

def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    
    T = np.eye(4)
    for i in range(link):
        a = dh_params[i][0]
        alpha = dh_params[i][1]
        d = dh_params[i][2]
        theta = dh_params[i][3] + joint_angles[i]
    
        T_i = get_transform_from_dh([a, alpha, d, theta])
        T = T @ T_i  

    return T


def get_transform_from_dh(dh_param):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians    
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    T = np.array([
        [ np.cos(dh_param[3]), -np.sin(dh_param[3]) * np.cos(dh_param[1]),  np.sin(dh_param[3]) * np.sin(dh_param[1]), dh_param[0]*np.cos(dh_param[3])],
        [ np.sin(dh_param[3]),  np.cos(dh_param[3]) * np.cos(dh_param[1]), -np.cos(dh_param[3]) * np.sin(dh_param[1]), dh_param[0]*np.sin(dh_param[3])],
        [ 0,                    np.sin(dh_param[1]),                        np.cos(dh_param[1]),                       dh_param[2] ],
        [ 0,                    0,                                          0,                                         1 ]
    ])
    return T

def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    R = T[0:3, 0:3]
    # For ZYZ:
    #  theta = arccos(R[2,2])
    #  phi   = arctan2(R[1,2], R[0,2])
    #  psi   = arctan2(R[2,1], -R[2,0])

    eps = 1e-9
    # Handle potential singularities near R[2,2] = ±1
    if abs(R[2,2] - 1.0) < eps:
        # close to +1
        theta = 0.0
        phi   = 0.0
        psi   = np.arctan2(R[1,0], R[0,0])  # rotation about Z
    elif abs(R[2,2] + 1.0) < eps:
        # close to -1
        theta = np.pi
        phi   = 0.0
        psi   = np.arctan2(R[1,0], R[0,0])  # rotation about Z
    else:
        theta = np.arccos(R[2,2])
        phi   = np.arctan2(R[1,2], R[0,2])
        psi   = np.arctan2(R[2,1], -R[2,0])
    return (phi, theta, psi)

def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    (phi, theta, psi) = get_euler_angles_from_T(T)
    return [x*1000, y*1000, z*1000, phi, theta, psi]


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    pass
