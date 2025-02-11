"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


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
    x_d, y_d, z_d, theta_d, psi_d = pose[0], pose[1], pose[2], pose[4], pose[5]   # x, y, z, theta, psi
    # print("Desired pose: ", x_d, y_d, z_d, theta_d, psi_d)
    theta_d = theta_d - np.pi/2  # 90 degrees offset


    if (np.abs(x_d) < 1e-9 and np.abs(y_d) < 1e-9) or z_d < -10:
        print("Degenerate case: x,y nearly zero => infinite or unreachable solutions for base.")
        return [0, 0, 0, 0, 0]
    
    ## DH parameters
    base_height = abs(dh_params[0][2]) * 1000       # d1, base height, 103.91 mm
    l1 = abs(dh_params[1][0]) * 1000                # a2, link1 length, 205.73 mm
    l2 = abs(dh_params[2][0]) * 1000                # a3, link2 length, 200 mm
    l_wrist2ee = abs(dh_params[4][2]) * 1000        # d4, wrist to ee length, 174.15 mm

    base_offset = dh_params[0][3]                   # base offset, 90 degrees
    l1_tilted_offset = dh_params[1][3]              # 90 + 14 degrees for link_1 tilted angle
    q1_offset = np.pi - l1_tilted_offset            # 90 - 14 degrees 
    wrist_roll_offset = dh_params[3][3]


    ## base angle
    # q_base = clamp(np.arctan2(-x_d, y_d))                    # base motor, initial state x0 = 0 -> x_d/y_d
    q_base = clamp(np.arctan2(y_d, x_d) - base_offset)*R2D     # base motor, rad
    q_base = max(min(q_base, 120), -120)                       # geomatric limitation



    ## calculate wrist position (r_eff, z_eff, D_eff)
    r = np.sqrt(x_d**2 + y_d**2)                                # base radius

    x_w = x_d + l_wrist2ee * np.cos(theta_d) * np.sin(q_base)   # wrist x
    y_w = y_d - l_wrist2ee * np.cos(theta_d) * np.cos(q_base)   # wrist y
    z_w = z_d + l_wrist2ee * np.sin(theta_d)                    # wrist z

    r_w = np.sqrt(x_w**2 + y_w**2)                              # wrist radius
    z_eff = z_w - base_height                                   # wrist height - base height
    D_eff = np.sqrt(r_w**2 + z_eff**2)                          # straight line from shoulder to wrist


    # Check reach:
    if D_eff > (l1 + l2) or D_eff < abs(l1 - l2):
        print("Target out of reach. D =", D_eff, " L1+L2 =", l1 + l2)
        return [0, 0, 0, 0, 0]
    

    ## calculate elbow angle btw l1 and l2
    cos_elbow = (l1**2 + l2**2 - D_eff**2) / (2.0 * l1 * l2)
    cos_elbow = max(min(cos_elbow, 1.0), -1.0)              # numeric clamp
    
    elbow_candidates = [ np.arccos(cos_elbow), -np.arccos(cos_elbow) ]      # elbow can be "up" or "down"
    

    ## calculate shoulder, elbow, and wrist angles
    solutions = []
    for elbow_angle in elbow_candidates:
        
        # Shoulder angle:
        beta = np.arctan2(z_eff, r_w)
        gamma = np.arccos((l1**2 + D_eff**2 - l2**2) / (2 * l1 * D_eff))

        q_shoulder = clamp(q1_offset - (beta + gamma))* R2D                # shoulder motor, q1_offset = 90 - 14 degrees
        q_shoulder = max(min(q_shoulder, 100), -110)                       # geomatric limitation


        # Elbow angle:
        q_elbow = clamp(l1_tilted_offset - elbow_angle) * R2D              # elbow motor, l1_tilted_offset = 90 + 14 degrees
        q_elbow = max(min(q_elbow, 95), -105)                              # geomatric limitation

        
        # Wrist pitch angles:
        q_wrist_pitch = -clamp(theta_d - q_elbow - q_shoulder) * R2D        # wrist pitch motor
        q_wrist_pitch = max(min(q_wrist_pitch, 127), -100)                       # geomatric limitation

        
        # Wrist roll angle:
        q_wrist_roll = clamp(psi_d) * R2D                                  # wrist rotation angle, -180~180
        # q_wrist_roll = 0.0 * R2D

        sol = np.array([q_base, q_shoulder, q_elbow, q_wrist_pitch, q_wrist_roll])
        solutions.append(sol)
    
    print(solutions)
    return np.array(solutions)
