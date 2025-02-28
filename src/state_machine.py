"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]
                
        # ArmLab Checkpoint 1, Task 1.3: List for storing waypoints recorded as part of the "cycling" task
        # self.taught_waypts = [
        #     [-0.01227185,  0.01073787,  0.05675729,  0.00920388,  0.        ,  0.        ],
        #     [ 0.42798066, -0.03374758,  0.31139812,  1.35757303,  0.42184472,  0.        ],
        #     [0.42644668, 0.1043107,  0.49854377, 1.03697109, 0.43411657, 0.        ],
        #     [0.42644668, 0.1043107,  0.49854377, 1.03697109, 0.43411657, 1.        ],
        #     [ 0.40803891, -0.17794177,  0.44178647,  1.23178661,  0.42031074,  0.        ],
        #     [-1.28854394,  0.02147573,  0.17333983,  1.39592254,  0.18100974,  0.        ],
        #     [-1.26706815,  0.13499032,  0.44485444,  0.98788363,  0.18867964,  0.        ],
        #     [-1.26706815,  0.13499032,  0.44485444,  0.98788363,  0.18867964, -1.        ],
        #     [-1.27013612, -0.16260196,  0.42491269,  1.16122353,  0.19328159,  0.        ],
        #     [-0.40650493, -0.07516506,  0.37429133,  1.28087401, -0.47093213,  0.        ],
        #     [-0.40650493,  0.06135923,  0.55990303,  0.97100985, -0.46786416,  0.        ],
        #     [-0.40650493,  0.06135923,  0.55990303,  0.97100985, -0.46786416,  1.        ],
        #     [-0.4172428,  -0.17487381,  0.40957287,  1.20570898, -0.36048549,  0.        ],
        #     [ 0.41417482, -0.0322136,   0.3850292,   1.26706815,  0.40343696,  0.        ],
        #     [0.41417482, 0.12425245, 0.46479619, 1.07992256, 0.40650493, 0.        ],
        #     [ 0.41417482,  0.12425245,  0.46479619,  1.07992256,  0.40650493, -1.        ],
        #     [ 0.40190297, -0.15646605,  0.40190297,  1.23025262,  0.39730105,  0.        ],
        #     [-1.28547597,  0.02147573,  0.30526218,  1.23792255,  0.14726216,  0.        ],
        #     [-1.27473807,  0.14879614,  0.44025251,  0.9909516 ,  0.20095149,  0.        ],
        #     [-1.27473807,  0.14879614,  0.44025251,  0.9909516 ,  0.20095149,  1.        ],
        #     [-1.27934003, -0.09817477,  0.36968938,  1.15508759,  0.19021362,  0.        ],
        #     [-0.40957287, -0.09050487,  0.38196123,  1.26400018, -0.44485444,  0.        ],
        #     [-0.40343696,  0.06289321,  0.56910688,  0.93419433, -0.45099038,  0.        ],
        #     [-0.40343696,  0.06289321,  0.56910688,  0.93419433, -0.45099038, -1.        ],
        #     [-0.01227185,  0.01073787,  0.05675729,  0.00920388,  0.        ,  0.        ]]
        self.taught_waypts = []
        self.current_block = None
        
        self.bball_taught_waypts = [
            [-0.04908739,   -0.30372819,    0.3850292 ,    -0.12578642,   0.01227185,     0.,    2.0,     0.5        ],
            [ 0.        ,    0.34667966,    1.02930117,    -0.9219225 ,   3.07409763,     0.,    2.0,     0.5        ],
            [ 0.        ,    0.34667966,    1.02930117,    -0.9219225 ,   3.07409763,     1.,    2.0,     0.5        ],
            [-1.14434969,   -0.64580595,    0.41110685,     0.18254372,  -0.0076699,      0.,    2.0,     0.5        ],
            [-1.60454392,    0.59365058,    0.74704868,    -1.47568953,   0.01840777,     0.,    2.0,     0.5        ],
            [-1.60454392,    0.59365058,    0.74704868,    -1.47568953,   0.01840777,    -1.,    2.0,     0.5        ],
            [-1.93434978,    0.81607783,    0.44945639,    -1.73033035,   0.10124274,     0.,    2.0,     0.5        ],
            [-1.91594207,    0.57984477,    0.34821364,    -0.80840790,  -0.02914564,     0.,    2.0,     0.5        ],
            [-1.94968963,   -0.09817477,    0.86823314,    -1.82236922,   0.05982525,     0.,    2.0,     0.5        ],
            [-1.92821395,   -1.46188378,    1.58306825,    -1.64289343,   0.11351458,     0.,    2.0,     0.5        ] 
            # [-0.01227185,   -0.25157285,    0.84982538,    -0.50314569,   3.12778687,     0.,    2.0,     0.5        ],
            # [-0.01073787,    0.15339808,    0.88664091,    -0.32980588,   3.1247189 ,     0.,    2.0,     0.5        ],
            # [-0.01073787,    0.15339808,    0.88664091,    -0.32980588,   3.1247189 ,     1.,    2.0,     0.5        ],
            # [ 0.98174775,   -0.40803891,    0.64887387,    -0.2086214 ,   0.05368933,     0.,    2.0,     0.5        ],
            # [ 0.98174775,   -0.40803891,    0.64887387,    -0.2086214 ,   0.05368933,    -1.,    2.0,     0.5        ],
            # [ 0.98328173,   -0.15646605,    0.81147587,    -0.65500981,   0.0398835 ,     0.,    2.0,     0.5        ],
            # [ 1.02776718,   -0.3129321 ,    0.02454369,     0.64427197,   0.0966408 ,     0.,    0.3,     0.1        ],
            # [ 0.05675729,   -0.54916513,    0.6703496 ,    -0.18714567,   0.02300971,     0.,    2.0,     0.5        ],
            # [-0.00920388,   -1.78095174,    1.7165246 ,     0.64733994,   0.05982525,     0.,    2.0,     0.5        ]
        ]

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and functions as needed.
        """

        # IMPORTANT: This function runs in a loop. If you make a new state, it will be run every iteration.
        #            The function (and the state functions within) will continuously be called until the state changes.

        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "record_positions":
            self.record_positions()

        if self.next_state == "repeat_positions":
            self.repeat_positions()

        if self.next_state == "record_gripper_open":
            self.record_gripper_open()

        if self.next_state == "record_gripper_closed":
        #     [-0.01227185,  0.01073787,  0.05675729,  0.00920388,  0.        ,  0.        ],
        #     [ 0.42798066, -0.03374758,  0.31139812,  1.35757303,  0.42184472,  0.        ],
        #     [0.42644668, 0.1043107,  0.49854377, 1.03697109, 0.43411657, 0.        ],
        #     [0.42644668, 0.1043107,  0.49854377, 1.03697109, 0.43411657, 1.        ],
        #     [ 0.40803891, -0.17794177,  0.44178647,  1.23178661,  0.42031074,  0.        ],
        #     [-1.28854394,  0.02147573,  0.17333983,  1.39592254,  0.18100974,  0.        ],
        #     [-1.26706815,  0.13499032,  0.44485444,  0.98788363,  0.18867964,  0.        ],
        #     [-1.26706815,  0.13499032,  0.44485444,  0.98788363,  0.18867964, -1.        ],
        #     [-1.27013612, -0.16260196,  0.42491269,  1.16122353,  0.19328159,  0.        ],
        #     [-0.40650493, -0.07516506,  0.37429133,  1.28087401, -0.47093213,  0.        ],
        #     [-0.40650493,  0.06135923,  0.55990303,  0.97100985, -0.46786416,  0.        ],
        #     [-0.40650493,  0.06135923,  0.55990303,  0.97100985, -0.46786416,  1.        ],
        #     [-0.4172428,  -0.17487381,  0.40957287,  1.20570898, -0.36048549,  0.        ],
        #     [ 0.41417482, -0.0322136,   0.3850292,   1.26706815,  0.40343696,  0.        ],
        #     [0.41417482, 0.12425245, 0.46479619, 1.07992256, 0.40650493, 0.        ],
        #     [ 0.41417482,  0.12425245,  0.46479619,  1.07992256,  0.40650493, -1.        ],
        #     [ 0.40190297, -0.15646605,  0.40190297,  1.23025262,  0.39730105,  0.        ],
        #     [-1.28547597,  0.02147573,  0.30526218,  1.23792255,  0.14726216,  0.        ],
        #     [-1.27473807,  0.14879614,  0.44025251,  0.9909516 ,  0.20095149,  0.        ],
        #     [-1.27473807,  0.14879614,  0.44025251,  0.9909516 ,  0.20095149,  1.        ],
        #     [-1.27934003, -0.09817477,  0.36968938,  1.15508759,  0.19021362,  0.        ],
        #     [-0.40957287, -0.09050487,  0.38196123,  1.26400018, -0.44485444,  0.        ],
        #     [-0.40343696,  0.06289321,  0.56910688,  0.93419433, -0.45099038,  0.        ],
        #     [-0.40343696,  0.06289321,  0.56910688,  0.93419433, -0.45099038, -1.        ],
        #     [-0.01227185,  0.01073787,  0.05675729,  0.00920388,  0.        ,  0.        ]]
            self.record_gripper_closed()
            
        if self.next_state == "click_grab":
            self.click_grab()
            
        if self.next_state == "click_place":
            self.click_place()
        
        if self.next_state == 'sort_n_stack':
            self.sort_n_stack()
        
        if self.next_state == 'line_em_up':
            self.line_em_up()
        
        if self.next_state == 'to_the_sky':
            self.to_the_sky()
        
        if self.next_state == 'Bball_repeat_position':
            self.Bball_repeat_position()

    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """![
        #     [-0.01227185,  0.01073787,  0.05675729,  0.00920388,  0.        ,  0.        ],
        #     [ 0.42798066, -0.03374758,  0.31139812,  1.35757303,  0.42184472,  0.        ],
        #     [0.42644668, 0.1043107,  0.49854377, 1.03697109, 0.43411657, 0.        ],
        #     [0.42644668, 0.1043107,  0.49854377, 1.03697109, 0.43411657, 1.        ],
        #     [ 0.40803891, -0.17794177,  0.44178647,  1.23178661,  0.42031074,  0.        ],
        #     [-1.28854394,  0.02147573,  0.17333983,  1.39592254,  0.18100974,  0.        ],
        #     [-1.26706815,  0.13499032,  0.44485444,  0.98788363,  0.18867964,  0.        ],
        #     [-1.26706815,  0.13499032,  0.44485444,  0.98788363,  0.18867964, -1.        ],
        #     [-1.27013612, -0.16260196,  0.42491269,  1.16122353,  0.19328159,  0.        ],
        #     [-0.40650493, -0.07516506,  0.37429133,  1.28087401, -0.47093213,  0.        ],
        #     [-0.40650493,  0.06135923,  0.55990303,  0.97100985, -0.46786416,  0.        ],
        #     [-0.40650493,  0.06135923,  0.55990303,  0.97100985, -0.46786416,  1.        ],
        #     [-0.4172428,  -0.17487381,  0.40957287,  1.20570898, -0.36048549,  0.        ],
        #     [ 0.41417482, -0.0322136,   0.3850292,   1.26706815,  0.40343696,  0.        ],
        #     [0.41417482, 0.12425245, 0.46479619, 1.07992256, 0.40650493, 0.        ],
        #     [ 0.41417482,  0.12425245,  0.46479619,  1.07992256,  0.40650493, -1.        ],
        #     [ 0.40190297, -0.15646605,  0.40190297,  1.23025262,  0.39730105,  0.        ],
        #     [-1.28547597,  0.02147573,  0.30526218,  1.23792255,  0.14726216,  0.        ],
        #     [-1.27473807,  0.14879614,  0.44025251,  0.9909516 ,  0.20095149,  0.        ],
        #     [-1.27473807,  0.14879614,  0.44025251,  0.9909516 ,  0.20095149,  1.        ],
        #     [-1.27934003, -0.09817477,  0.36968938,  1.15508759,  0.19021362,  0.        ],
        #     [-0.40957287, -0.09050487,  0.38196123,  1.26400018, -0.44485444,  0.        ],
        #     [-0.40343696,  0.06289321,  0.56910688,  0.93419433, -0.45099038,  0.        ],
        #     [-0.40343696,  0.06289321,  0.56910688,  0.93419433, -0.45099038, -1.        ],
        #     [-0.01227185,  0.01073787,  0.05675729,  0.00920388,  0.        ,  0.        ]]
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        for waypt in self.waypoints:
            self.rxarm.set_positions(waypt)
            time.sleep(3)

        self.next_state = "idle"

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        # we have self.camera. Thus, we can call the calibration routine 
        self.camera.recover_homogeneous_transform_pnp(self.camera.tag_detections_raw)
        # self.camera.recover_homogeneous_transform_svd(self.camera.tag_detections_raw)
        self.camera.homography_transform(self.camera.tag_detections_raw)
        # Setting the calibration as complete
        self.camera.camera_calibrated = True
        self.status_message = "Calibration - Completed Calibration"

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

    def record_positions(self):
        """!
        @brief      Records a "taught" trajectory in the form of waypoints

        Requires the user to pause for 5 seconds at each waypoint of interest for the task
        """
        self.current_state = "record_positions"
        self.status_message = "Recording positions from manual control"
    
        # Appends current series of joint positions to the taught waypoints
        waypt_positions = np.append(self.rxarm.get_positions(), np.asarray(0))
        # This indicates whether the gripper state should transition. 
        # 0 means no change, 1 means we want the gripper to close, -1 means we want the gripper to open

        # Append the taught waypoint with the gripper state transition
        self.taught_waypts.append(waypt_positions)
        print(f"Last position taught is {self.taught_waypts[-1]}")
        
        self.next_state = "idle"

    def repeat_positions(self):
        """!
        @brief      Repeats the waypoints of the recorded trajectory
        """
        self.current_state = "repeat_positions"
        self.status_message = "Repeating positions taught during manual control"

        # Using the same methodology as the "execute" button
        for taught_waypt in self.taught_waypts:
            # This should execute the 5 joint positions for the desired waypoint
            self.rxarm.set_positions(taught_waypt[:-1])
            time.sleep(2)

            # Here, we change the gripper status if necessary
            if (taught_waypt[-1] == 1):
                self.rxarm.gripper.grasp()
                time.sleep(2)
            elif (taught_waypt[-1] == -1):
                self.rxarm.gripper.release()
                time.sleep(2)

        self.next_state = "idle"
        
    def Bball_repeat_position(self):

        self.current_state = "Bball_repeat_position"
        self.status_message = "Ball Repeating"
        self.bball_taught_waypts = np.asarray(self.bball_taught_waypts)
        
        for bball_taught_waypt in self.bball_taught_waypts:
            print("basketball launch sequence")
            joint_angle = bball_taught_waypt[0:5]
            print(joint_angle)
            gripper_cmd = bball_taught_waypt[5]
            m_time = bball_taught_waypt[6]
            a_time = bball_taught_waypt[7]
            
            self.rxarm.set_moving_time(m_time)
            self.rxarm.set_accel_time(a_time)
            
            self.rxarm.set_positions(joint_angle)
            
            time.sleep(3)

            # Here, we change the gripper status if necessary
            if (gripper_cmd == 1):
                self.rxarm.gripper.grasp()
                time.sleep(1)
            elif (gripper_cmd == -1):
                self.rxarm.gripper.release()
                time.sleep(1)

        self.next_state = "idle"


    def record_gripper_open(self):
        self.current_state = "record_gripper_open"
        self.status_message = "Record Open Gripper for this waypoint"
        # -1 represents an Open Gripper action when these waypoints are being executed
        self.taught_waypts[-1][-1] = -1
        print(f"Recorded gripper open, waypoints now {self.taught_waypts[-1]}")
        self.next_state = "idle"

    def record_gripper_closed(self):
        self.current_state = "record_gripper_closed"
        self.status_message = "Record Close Gripper for this waypoint"

        # 1 represents a Close Gripper action when these waypoints are being executed
        self.taught_waypts[-1][-1] = 1
        print(f"Recorded gripper closed, waypoints now {self.taught_waypts[-1]}")
        self.next_state = "idle"

    def click_grab(self):
        self.current_state = "click_grab"
        self.status_message = "Click to go to a specific point (pre-grasp)"
        
        # Retrieve position
        x, y, z = self.camera.retrieve_clicked_pos(self.camera.last_click[0], self.camera.last_click[1])
        print(f"x, y, z = {x}, {y}, {z}")
        phi, theta, psi = 0.0, np.pi, 0.0 
        
        for block_info in self.camera.blocks_info_list:
            cx, cy = block_info['location']
            if np.abs(self.camera.last_click[0] - cx) < 20 and np.abs(self.camera.last_click[1] - cy) < 20:
                psi = block_info['orientation']
                x, y, z = self.camera.retrieve_clicked_pos(cx, cy)
                self.current_block = block_info
                # print(psi)
                break
                
        
        # Pre-grasp
        pose = np.array((x , y-10, z+55, phi, theta, psi))
        self.rxarm.set_desired_joint_positions(pose)
        time.sleep(3)
        
        # Grasp
        pose = np.array((x, y-10, z+15, phi, theta, psi))
        self.rxarm.set_desired_joint_positions(pose)
        time.sleep(3)
        self.rxarm.gripper.grasp()
        time.sleep(2)
        
        # Post-grasp
        pose = np.array((x, y-10, z+90, phi, theta, psi))
        self.rxarm.set_desired_joint_positions(pose)
        time.sleep(3)
        
        self.next_state = "idle"
        
    def click_place(self):
        self.current_state = "click_place"
        self.status_message = "Click to go to a specific point (pre-grasp)"
        
        # Retrieve position
        x, y, z = self.camera.retrieve_clicked_pos(self.camera.last_click[0], self.camera.last_click[1])
        print(f"x, y, z = {x}, {y}, {z}")
        phi, theta, psi = 0.0, np.pi, 0.0 
        
        # Pre-release
        pose = np.array((x, y-10, z + 90, phi, theta, psi))
        self.rxarm.set_desired_joint_positions(pose)
        time.sleep(3)
        
        # Release
        if self.current_block['size'] == 'large':
            pose = np.array((x, y-10, z + 45, phi, theta, psi))
        else:
            pose = np.array((x, y-10, z + 35, phi, theta, psi))
        
        self.rxarm.set_desired_joint_positions(pose)
        time.sleep(3)
        self.rxarm.gripper.release()
        time.sleep(2)
        
        # Post-release
        pose = np.array((x, y-10, z+90, phi, theta, psi))
        self.rxarm.set_desired_joint_positions(pose)
        time.sleep(3)
        
        self.next_state = "idle"

    def sort_n_stack(self):         # place candidated blocks within x: -150~150, y > 0
        self.current_state = 'sort_n_stack'
        self.status_message = 'Sorting and stacking blocks for event 1'

        # Sort the blocks by color
        self.camera.blocks_info_list.sort(key=lambda x: self.camera.color_order.get(x['color']))
        print(len(self.camera.blocks_info_list))

        # Find blocks needed to sort within x: -150~150, y > 0
        candidated_blocks = []
        for block in self.camera.blocks_info_list:
            cx, cy = block['location']
            block_x, block_y, block_z = self.camera.retrieve_clicked_pos(cx, cy)
            block['color_order'] = self.camera.color_order.get(block['color'])
            if abs(block_x) < 150 and abs(block_y) > 0:
                # print(block_x,  block_y)
                candidated_blocks.append(block)
        
        print(len(candidated_blocks))

        # initialize large and small placing points
        x_large, y_large, z_large = 250, -25, 10
        x_small, y_small, z_small = -250, -25, 7
        # Sort the blocks
        for block in candidated_blocks:
            # Move to the block
            x, y, _ = self.camera.retrieve_clicked_pos(block['location'][0], block['location'][1])
            phi, theta = 0.0, np.pi
            psi = block['orientation']
            if psi > np.pi/4:
                psi = np.pi/2 - psi
                
            if block['size'] == 'large':
                z = 20
            else:
                z = 10

            # Grasping planning algorithm
            pose = np.array((x, y, z, phi, theta, psi))
            self.rxarm.block_grab_planning(pose)
            
            # Placing planning algorithm
            if block['size'] == 'large':                                # store large blocks in right side
                x, y, z = x_large, y_large, z_large
                phi, theta, psi = 0.0, np.pi, 0.0
                # print(z)
                pose = np.array((x, y, z, phi, theta, psi))
                self.rxarm.block_place_planning(pose)
                z_large += 40
                
            else:                                                       # store small blocks in left side
                x, y, z = x_small, y_small, z_small
                phi, theta, psi = 0.0, np.pi, 0.0
                # print(z)
                pose = np.array((x, y, z, phi, theta, psi))
                self.rxarm.block_place_planning(pose)
                z_small += 20
        
        # back to initial position
        self.rxarm.initialize()
        self.next_state = 'idle'

    def line_em_up(self):       # place candidated blocks within y < 125
        self.current_state = 'line_em_up'
        self.status_message = 'Lining blocks for event 2'
        # Sort the blocks by color
        self.camera.blocks_info_list.sort(key=lambda x: self.camera.color_order.get(x['color']))
        # print(len(self.camera.blocks_info_list))

        # Find blocks needed to sort within x: -150~150, y > 0
        candidated_blocks = []
        for block in self.camera.blocks_info_list:
            cx, cy = block['location']
            block_x, block_y, block_z = self.camera.retrieve_clicked_pos(cx, cy)
            block['color_order'] = self.camera.color_order.get(block['color'])
            if abs(block_x) < 400 and abs(block_y) > 0:
                # print(block_x,  block_y)
                candidated_blocks.append(block)
        #print(len(candidated_blocks))

        # initiate large and small lining position
        x_large, y_large = 400, -75
        x_small, y_small = -400, -75
        # Line up the blocks
        for block in candidated_blocks:
            # Move to the block
            x, y, _ = self.camera.retrieve_clicked_pos(block['location'][0], block['location'][1])
            phi, theta = 0.0, np.pi
            psi = block['orientation']
            if psi > np.pi/4:
                psi = np.pi/2 - psi
                
            if block['size'] == 'large':
                z = 20
            else:
                z = 10

            # Grasping planning algorithm
            pose = np.array((x, y, z, phi, theta, psi))
            self.rxarm.block_grab_planning(pose)

            # Placing planning algorithm
            if block['size'] == 'large':     # store large blocks in y = 325
                x, y = x_large, y_large
                z = 10
                phi, theta, psi = 0.0, np.pi, 0.0

                pose = np.array((x, y, z, phi, theta, psi))
                self.rxarm.block_place_planning(pose)
                x_large -= 60                # maximum 6 blocks line length < 300mm
            else:                            # store small blocks in y = 225
                x, y = x_small, y_small
                z = 7
                phi, theta, psi = 0.0, np.pi, 0.0

                pose = np.array((x, y, z, phi, theta, psi))
                self.rxarm.block_place_planning(pose)
                x_small += 60

        # back to initial position
        self.rxarm.initialize()
        self.next_state = 'idle'

    def to_the_sky(self):       # place candidated blocks within y < 125
        self.current_state = 'to_the_sky'
        self.status_message = 'Stacking blocks for event 3'
        
        # print(len(self.camera.blocks_info_list))
        # # Find blocks needed to stack up within (y < 125)
        # candidated_blocks = []
        # for block in self.camera.blocks_info_list:
        #     cx, cy = block['location']
        #     block_x, block_y, block_z = self.camera.retrieve_clicked_pos(cx, cy)
        #     # block['color_order'] = self.color_order.get(block['color'])
        #     if abs(block_y) > 125:
        #         candidated_blocks.append(block)
        # print(len(candidated_blocks))

        x_block, y_block = 250, 0
        z_block = 20
        # Stack up the blocks
        for i in range(25):
        # for block in candidated_blocks:
            # Move to the block
            # x, y, _ = self.camera.retrieve_clicked_pos(block['location'][0], block['location'][1])
            # phi, theta = 0.0, np.pi
            # psi = block['orientation']

            # z = 20
            # Grasping planning algorithm
            pose = np.array((0, 250, 20, 0.0, np.pi, 0.0))
            self.rxarm.block_grab_planning(pose)

            # Placing planning algorithm
            x, y = x_block, y_block
            z = z_block
            phi = 0.0
            if z > 170:              # stack over 6 blocks, theta be pi/2 (horizontal), 35mm * 6
                x = 240
                y = 4
                theta = np.pi/2
                psi = np.pi/2
            else:                   # stack under 3 blocks, theta be pi (vertical)
                theta = np.pi
                psi = 0.0

            pose = np.array((x, y, z, phi, theta, psi))
            self.rxarm.block_place_planning(pose) 
            
            if  185 > z > 170:
                z_block += 5
            else:
                z_block += 38

        self.next_state = 'idle'

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)
            
            
"""
[-0.04908739 -0.30372819  0.3850292  -0.12578642 -0.01227185  0., 2.0    , 0.5        ]
[ 0.          0.34667966  1.02930117 -0.9219225   3.07409763  0., 2.0    , 0.5        ]
[ 0.          0.34667966  1.02930117 -0.9219225   3.07409763  1., 2.0    , 0.5        ]
[-1.60454392  0.59365058  0.74704868 -1.47568953  0.01840777  0., 2.0    , 0.5        ]
[-1.60454392  0.59365058  0.74704868 -1.47568953  0.01840777 -1., 2.0    , 0.5        ]
[-1.93434978  0.81607783  0.44945639 -1.73033035  0.10124274  0., 2.0    , 0.5        ]
[-1.92974794 -0.02454369  0.81914574 -1.81009734  0.05982525  0., 1.0    , 0.3        ]
[-0.01227185 -0.25157285  0.84982538 -0.50314569 -3.12778687  0., 2.0    , 0.5        ]
[-0.01073787  0.15339808  0.88664091 -0.32980588 -3.1247189   0., 2.0    , 0.5        ]
[-0.01073787  0.15339808  0.88664091 -0.32980588 -3.1247189   1., 2.0    , 0.5        ]
[ 0.98174775 -0.40803891  0.64887387 -0.2086214  -0.05368933  0., 2.0    , 0.5        ]
[ 0.98174775 -0.40803891  0.64887387 -0.2086214  -0.05368933 -1., 2.0    , 0.5        ]
[ 0.98328173 -0.15646605  0.81147587 -0.65500981 -0.0398835   0., 2.0    , 0.5        ]
[ 1.02776718 -0.3129321   0.02454369  0.64427197 -0.0966408   0., 1.0    , 0.3        ]
[ 0.05675729 -0.54916513  0.6703496  -0.18714567 -0.02300971  0., 2.0    , 0.5        ]
[-0.00920388 -1.78095174  1.7165246   0.64733994 -0.05982525  0., 2.0    , 0.5        ]

"""

"""
Tower Stacking v1
Last position taught is [-0.01994175 -0.24543694  0.29145637  0.17947575  0.16413595  0.        ]
Last position taught is [ 1.55852449  0.5276894  -0.56757289  1.61067986 -0.03067962  0.        ]
Last position taught is [ 1.56926239  0.57984477 -0.30833015  1.31922352 -0.03067962  0.        ]
Recorded gripper closed, waypoints now [ 1.56926239  0.57984477 -0.30833015  1.31922352 -0.03067962  1.        ]
Last position taught is [ 1.56926239  0.23930101 -0.32366997  1.44194198 -0.0322136   0.        ]
Last position taught is [-0.0322136  -0.4954758   0.41417482  1.60300994 -0.02761165  0.        ]
Last position taught is [ 0.00613592 -0.23469907  0.80533993  1.03850508  0.00613592  0.        ]
Recorded gripper open, waypoints now [ 0.00613592 -0.23469907  0.80533993  1.03850508  0.00613592 -1.        ]
Last position taught is [ 0.0076699  -0.65347582  0.29759228  1.53551483  0.01380583  0.        ]
Last position taught is [ 1.5569905   0.47246608 -0.32520393  1.4005245  -0.03681554  0.        ]
Last position taught is [ 1.56312644  0.58751464 -0.32520393  1.33456337 -0.0398835   0.        ]
Recorded gripper closed, waypoints now [ 1.56312644  0.58751464 -0.32520393  1.33456337 -0.0398835   1.        ]
Last position taught is [ 1.5569905   0.34361172 -0.38042724  1.53551483 -0.03681554  0.        ]
Last position taught is [ 0.02147573 -0.45866027  0.54763114  1.49869931  0.00920388  0.        ]
Last position taught is [ 0.01227185 -0.31446606  0.6703496   1.28547597  0.0076699   0.        ]
Recorded gripper open, waypoints now [ 0.01227185 -0.31446606  0.6703496   1.28547597  0.0076699  -1.        ]
Last position taught is [ 0.01073787 -0.69182533  0.59978652  1.37751484  0.01073787  0.        ]
Last position taught is [ 1.54318476  0.47860202 -0.40497094  1.5079031  -0.06135923  0.        ]
Last position taught is [ 1.55238855  0.55530107 -0.2638447   1.28394198 -0.06289321  0.        ]
Recorded gripper closed, waypoints now [ 1.55238855  0.55530107 -0.2638447   1.28394198 -0.06289321  1.        ]
Last position taught is [ 1.55545652  0.38656318 -0.36508745  1.55545652 -0.05829127  0.        ]
Last position taught is [ 0.01687379 -0.39269909  0.19021362  1.81163132  0.01687379  0.        ]
Last position taught is [ 0.02607767 -0.40497094  0.59518456  1.39745653  0.01687379  0.        ]
Recorded gripper open, waypoints now [ 0.02607767 -0.40497094  0.59518456  1.39745653  0.01687379 -1.        ]
Last position taught is [-0.00920388 -0.86669916  0.53382534  1.37291288  0.01533981  0.        ]
Last position taught is [ 1.55085456  0.50928164 -0.59365058  1.70118475 -0.01533981  0.        ]
Last position taught is [ 1.56619442  0.53535932 -0.25310683  1.27473807 -0.0076699   0.        ]
Recorded gripper closed, waypoints now [ 1.56619442  0.53535932 -0.25310683  1.27473807 -0.0076699   1.        ]
Last position taught is [ 1.56619442e+00  1.64135948e-01 -2.53106833e-01  1.46188378e+00 1.53398083e-03  0.00000000e+00]
Last position taught is [-1.53398083e-03 -3.83495212e-01  2.60776747e-02  1.91134012e+00 -1.68737900e-02  0.00000000e+00]
Last position taught is [ 1.53398083e-03 -3.81961226e-01  3.72757345e-01  1.65669930e+00 -2.30097119e-02  0.00000000e+00]
Recorded gripper open, waypoints now [ 1.53398083e-03 -3.81961226e-01  3.72757345e-01  1.65669930e+00 -2.30097119e-02 -1.00000000e+00]
Last position taught is [ 0.01840777 -0.44792241  0.13192235  1.81009734  0.02761165  0.        ]
Last position taught is [ 1.57386434 -0.07516506  0.36508745  1.27780604  0.03374758  0.        ]
Last position taught is [1.56926239 0.04448544 0.52462143 0.96947587 0.0322136  0.        ]
Recorded gripper closed, waypoints now [1.56926239 0.04448544 0.52462143 0.96947587 0.0322136  1.        ]
Last position taught is [ 1.5677284  -0.30372819  0.43565056  1.28700995  0.03681554  0.        ]
Last position taught is [ 1.58000028  0.45252433 -0.47860202  1.56926239  0.03374758  0.        ]
Last position taught is [ 1.57079637  0.48013601 -0.30833015  1.41126239  0.03528156  0.        ]
Recorded gripper open, waypoints now [ 1.57079637  0.48013601 -0.30833015  1.41126239  0.03528156 -1.        ]
Last position taught is [ 1.55852449  0.30679616 -0.42951465  1.51403904  0.00613592  0.        ]
Last position taught is [ 1.56312644 -0.00920388  0.37122336  1.20877695 -0.01227185  0.        ]
Last position taught is [ 1.56005847  0.04141748  0.53996128  0.96333998 -0.01380583  0.        ]
Recorded gripper closed, waypoints now [ 1.56005847  0.04141748  0.53996128  0.96333998 -0.01380583  1.        ]
Last position taught is [ 1.55545652 -0.18561168  0.41110685  1.29774773 -0.00306796  0.        ]
Last position taught is [ 1.54625273  0.36355346 -0.51695156  1.60761189 -0.06596117  0.        ]
Last position taught is [ 1.5569905   0.46172822 -0.46786416  1.53244686 -0.06135923  0.        ]
Recorded gripper open, waypoints now [ 1.5569905   0.46172822 -0.46786416  1.53244686 -0.06135923 -1.        ]
Last position taught is [ 1.55392253  0.40497094 -0.65040785  1.73646629 -0.05215535  0.        ]
Last position taught is [ 1.50176728 -1.29467988  0.92345643  0.40803891  0.03528156  0.        ]
Last position taught is [ 1.57233036  0.52615541  1.18883514 -1.59994197  0.05368933  0.        ]
Last position taught is [ 1.57233036  0.55530107  0.92345643 -1.38365066  0.05215535  0.        ]
Recorded gripper closed, waypoints now [ 1.57233036  0.55530107  0.92345643 -1.38365066  0.05215535  1.        ]
Last position taught is [ 1.57233036 -0.320602    0.98021376 -0.84368944  0.03681554  0.        ]
Last position taught is [ 0.00613592 -1.35757303  1.11980605  0.3528156  -0.08897088  0.        ]
Last position taught is [ 0.0398835  -1.45114589  1.39285457  0.11504856 -0.0076699   0.        ]
Recorded gripper open, waypoints now [ 0.0398835  -1.45114589  1.39285457  0.11504856 -0.0076699  -1.        ]
Last position taught is [ 0.04295146 -1.10906816  0.62126225  0.51388359  0.03528156  0.        ]
Last position taught is [ 1.47722352 -1.1826992   0.51234961  0.73784477  0.0644272   0.        ]
Last position taught is [ 1.55545652 -0.01994175  0.33747578  1.27627206 -0.02147573  0.        ]
Last position taught is [ 1.55545652  0.10584468  0.46633017  1.00015545 -0.02300971  0.        ]
Recorded gripper closed, waypoints now [ 1.55545652  0.10584468  0.46633017  1.00015545 -0.02300971  1.        ]
Last position taught is [ 1.55545652 -0.2561748   0.51388359  1.10600019 -0.00920388  0.        ]
Last position taught is [ 1.55238855  0.40803891 -0.39269909  1.49716532 -0.07669904  0.        ]
Last position taught is [ 1.55545652  0.49087387 -0.35128161  1.4204663  -0.07363108  0.        ]
Recorded gripper open, waypoints now [ 1.55545652  0.49087387 -0.35128161  1.4204663  -0.07363108 -1.        ]
Last position taught is [ 1.5569905   0.24543694 -0.34667966  1.4680196  -0.07823303  0.        ]
Last position taught is [ 1.55545652 -0.07669904  0.45712629  1.14281571 -0.07669904  0.        ]
Last position taught is [ 1.55545652  0.05829127  0.50467968  0.961806   -0.07516506  0.        ]
Recorded gripper closed, waypoints now [ 1.55545652  0.05829127  0.50467968  0.961806   -0.07516506  1.        ]
Last position taught is [ 1.55238855 -0.1672039   0.43411657  1.21337879 -0.08283497  0.        ]
Last position taught is [ 1.53398085  0.40497094 -0.60745639  1.68737888 -0.08590293  0.        ]
Last position taught is [ 1.55545652  0.46633017 -0.46786416  1.5385828  -0.08283497  0.        ]
Recorded gripper open, waypoints now [ 1.55545652  0.46633017 -0.46786416  1.5385828  -0.08283497 -1.        ]
Last position taught is [ 1.55852449  0.16566993 -0.41877678  1.56005847 -0.079767    0.        ]
Last position taught is [ 1.5677284  -0.82528168  1.52784491 -0.64733994 -0.02761165  0.        ]
Last position taught is [ 1.57079637  0.6304661   0.93112636 -1.51710701 -0.02914564  0.        ]
Recorded gripper closed, waypoints now [ 1.57079637  0.6304661   0.93112636 -1.51710701 -0.02914564  1.        ]
Last position taught is [ 1.57079637 -0.0076699   1.09679627 -1.13207781 -0.03067962  0.        ]
Last position taught is [ 1.56926239 -0.69335932  0.62433022  0.05675729 -0.01994175  0.        ]
Last position taught is [ 0.0322136  -1.21337879  0.68262148  0.52615541  0.0398835   0.        ]
Last position taught is [ 0.02914564 -1.27013612  0.80687392  0.50467968  0.04141748  0.        ]
Recorded gripper open, waypoints now [ 0.02914564 -1.27013612  0.80687392  0.50467968  0.04141748 -1.        ]
Last position taught is [ 0.0076699  -0.75165063 -0.15033013  0.9510681  -0.00920388  0.        ]
Last position taught is [ 1.50023329 -0.719437   -0.2316311   0.88357294 -0.01227185  0.        ]
Last position taught is [1.56926239 0.01840777 0.34667966 1.25019443 0.0398835  0.        ]
Last position taught is [1.5677284  0.12578642 0.41417482 1.05077684 0.04295146 0.        ]
Recorded gripper closed, waypoints now [1.5677284  0.12578642 0.41417482 1.05077684 0.04295146 1.        ]
Last position taught is [ 1.55085456 -0.13652429  0.398835    1.23945653  0.04141748  0.        ]
Last position taught is [ 1.55085456  0.44485444 -0.46172822  1.55238855 -0.03681554  0.        ]
Last position taught is [ 1.55852449  0.52462143 -0.40343696  1.48029149 -0.02761165  0.        ]
Recorded gripper open, waypoints now [ 1.55852449  0.52462143 -0.40343696  1.48029149 -0.02761165 -1.        ]
Last position taught is [ 1.5677284   0.37889326 -0.56297094  1.69965076 -0.03067962  0.        ]
Last position taught is [ 1.56005847 -0.0322136   0.398835    1.20877695 -0.02454369  0.        ]
Last position taught is [ 1.56005847  0.07669904  0.48166999  0.98328173 -0.02607767  0.        ]
Recorded gripper closed, waypoints now [ 1.56005847  0.07669904  0.48166999  0.98328173 -0.02607767  1.        ]
Last position taught is [ 1.5569905   0.33594179 -0.54916513  1.66130126 -0.03681554  0.        ]
Last position taught is [ 1.56159246  0.42337871 -0.37889326  1.45267987 -0.03528156  0.        ]
Recorded gripper open, waypoints now [ 1.56159246  0.42337871 -0.37889326  1.45267987 -0.03528156 -1.        ]
Last position taught is [ 1.56159246 -0.12271847 -0.02454369  1.59227204 -0.02300971  0.        ]
Last position taught is [ 1.5569905   0.63353407  1.12747586 -1.72879636  0.04448544  0.        ]
Last position taught is [ 1.57079637  0.6657477   0.90965062 -1.52784491  0.04295146  0.        ]
Recorded gripper closed, waypoints now [ 1.57079637  0.6657477   0.90965062 -1.52784491  0.04295146  1.        ]
Last position taught is [ 1.5784663  -0.16873789  1.01702929 -0.88357294  0.04141748  0.        ]
Last position taught is [ 1.54625273 -0.54916513  0.03374758  0.46479619  0.04141748  0.        ]
Last position taught is [-0.01227185 -0.78693217 -0.05368933  0.87743706  0.04448544  0.        ]
Last position taught is [ 0.02761165 -0.95567006  0.1840777   0.73631078  0.04601942  0.        ]
Recorded gripper open, waypoints now [ 0.02761165 -0.95567006  0.1840777   0.73631078  0.04601942 -1.        ]
Last position taught is [ 0.04908739 -0.10891264 -1.48029149  1.56619442  0.02761165  0.        ]
Last position taught is [ 1.43427205 -0.19481556 -1.45728183  1.56312644  0.01840777  0.        ]

"""