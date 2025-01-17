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
        self.taught_waypts = [
            [-0.01227185,  0.01073787,  0.05675729,  0.00920388,  0.        ,  0.        ],
            [ 0.42798066, -0.03374758,  0.31139812,  1.35757303,  0.42184472,  0.        ],
            [0.42644668, 0.1043107,  0.49854377, 1.03697109, 0.43411657, 0.        ],
            [0.42644668, 0.1043107,  0.49854377, 1.03697109, 0.43411657, 1.        ],
            [ 0.40803891, -0.17794177,  0.44178647,  1.23178661,  0.42031074,  0.        ],
            [-1.28854394,  0.02147573,  0.17333983,  1.39592254,  0.18100974,  0.        ],
            [-1.26706815,  0.13499032,  0.44485444,  0.98788363,  0.18867964,  0.        ],
            [-1.26706815,  0.13499032,  0.44485444,  0.98788363,  0.18867964, -1.        ],
            [-1.27013612, -0.16260196,  0.42491269,  1.16122353,  0.19328159,  0.        ],
            [-0.40650493, -0.07516506,  0.37429133,  1.28087401, -0.47093213,  0.        ],
            [-0.40650493,  0.06135923,  0.55990303,  0.97100985, -0.46786416,  0.        ],
            [-0.40650493,  0.06135923,  0.55990303,  0.97100985, -0.46786416,  1.        ],
            [-0.4172428,  -0.17487381,  0.40957287,  1.20570898, -0.36048549,  0.        ],
            [ 0.41417482, -0.0322136,   0.3850292,   1.26706815,  0.40343696,  0.        ],
            [0.41417482, 0.12425245, 0.46479619, 1.07992256, 0.40650493, 0.        ],
            [ 0.41417482,  0.12425245,  0.46479619,  1.07992256,  0.40650493, -1.        ],
            [ 0.40190297, -0.15646605,  0.40190297,  1.23025262,  0.39730105,  0.        ],
            [-1.28547597,  0.02147573,  0.30526218,  1.23792255,  0.14726216,  0.        ],
            [-1.27473807,  0.14879614,  0.44025251,  0.9909516 ,  0.20095149,  0.        ],
            [-1.27473807,  0.14879614,  0.44025251,  0.9909516 ,  0.20095149,  1.        ],
            [-1.27934003, -0.09817477,  0.36968938,  1.15508759,  0.19021362,  0.        ],
            [-0.40957287, -0.09050487,  0.38196123,  1.26400018, -0.44485444,  0.        ],
            [-0.40343696,  0.06289321,  0.56910688,  0.93419433, -0.45099038,  0.        ],
            [-0.40343696,  0.06289321,  0.56910688,  0.93419433, -0.45099038, -1.        ],
            [-0.01227185,  0.01073787,  0.05675729,  0.00920388,  0.        ,  0.        ]]

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
            self.record_gripper_closed()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
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
A sample set of waypoints taught to the robot for the task

01/16/2025
Last position taught is [-0.0076699   0.01073787  0.0644272   0.00920388  0.          0.        ]
Last position taught is [0.58904862 0.06902914 0.09050487 1.49256337 0.63660204 0.        ]
Last position taught is [0.57524282 0.2086214  0.33287385 1.10600019 0.61359233 0.        ]
Recorded gripper closed, waypoints now [0.57524282 0.2086214  0.33287385 1.10600019 0.61359233 1.        ]
Last position taught is [0.56910688 0.05675729 0.17333983 1.42506814 0.62586421 0.        ]
Last position taught is [-1.29467988  0.0720971   0.22549519  1.35603905  0.20555343  0.        ]
Last position taught is [-1.282408    0.16566993  0.42951465  1.01396132  0.20095149  0.        ]
Recorded gripper open, waypoints now [-1.282408    0.16566993  0.42951465  1.01396132  0.20095149 -1.        ]
Last position taught is [-1.282408   -0.00153398  0.26537868  1.32229149  0.21782528  0.        ]
Last position taught is [-0.44025251 -0.00460194  0.30066025  1.34376717 -0.5414952   0.        ]
Last position taught is [-0.42337871  0.14726216  0.44945639  1.07992256 -0.48320395  0.        ]
Recorded gripper closed, waypoints now [-0.42337871  0.14726216  0.44945639  1.07992256 -0.48320395  1.        ]
Last position taught is [-0.42184472 -0.09970875  0.38349521  1.21031082 -0.4172428   0.        ]
Last position taught is [0.55530107 0.09510681 0.12578642 1.40512645 0.59058261 0.        ]
Last position taught is [0.57064086 0.23930101 0.30372819 1.12134004 0.59978652 0.        ]
Recorded gripper closed, waypoints now [0.57064086 0.23930101 0.30372819 1.12134004 0.59978652 1.        ]
Recorded gripper open, waypoints now [ 0.57064086  0.23930101  0.30372819  1.12134004  0.59978652 -1.        ]
Last position taught is [0.53996128 0.00153398 0.19788353 1.32075751 0.56450492 0.        ]
Last position taught is [-1.27627206  0.06596117  0.18714567  1.39899051  0.21168935  0.        ]
Last position taught is [-1.2716701   0.18561168  0.38349521  1.08299041  0.2239612   0.        ]
Recorded gripper closed, waypoints now [-1.2716701   0.18561168  0.38349521  1.08299041  0.2239612   1.        ]
Last position taught is [-1.2716701   0.00153398  0.19481556  1.32842743  0.21782528  0.        ]
Last position taught is [-0.42951465  0.          0.36508745  1.26093221 -0.45866027  0.        ]
Last position taught is [-0.42337871  0.14112623  0.45252433  1.06918466 -0.46019426  0.        ]
Recorded gripper open, waypoints now [-0.42337871  0.14112623  0.45252433  1.06918466 -0.46019426 -1.        ]
Last position taught is [-0.42337871 -0.08897088  0.40190297  1.21337879 -0.45099038  0.        ]
Last position taught is [-0.02300971  0.13652429 -0.09050487  0.11965051 -0.01994175  0.        ]

01/17/2025
Last position taught is [-0.01227185  0.01073787  0.05675729  0.00920388  0.          0.        ]
Last position taught is [ 0.42798066 -0.03374758  0.31139812  1.35757303  0.42184472  0.        ]
Last position taught is [0.42644668 0.1043107  0.49854377 1.03697109 0.43411657 0.        ]
Recorded gripper closed, waypoints now [0.42644668 0.1043107  0.49854377 1.03697109 0.43411657 1.        ]
Last position taught is [ 0.40803891 -0.17794177  0.44178647  1.23178661  0.42031074  0.        ]
Last position taught is [-1.28854394  0.02147573  0.17333983  1.39592254  0.18100974  0.        ]
Last position taught is [-1.26706815  0.13499032  0.44485444  0.98788363  0.18867964  0.        ]
Recorded gripper open, waypoints now [-1.26706815  0.13499032  0.44485444  0.98788363  0.18867964 -1.        ]
Last position taught is [-1.27013612 -0.16260196  0.42491269  1.16122353  0.19328159  0.        ]
Last position taught is [-0.40650493 -0.07516506  0.37429133  1.28087401 -0.47093213  0.        ]
Last position taught is [-0.40650493  0.06135923  0.55990303  0.97100985 -0.46786416  0.        ]
Recorded gripper closed, waypoints now [-0.40650493  0.06135923  0.55990303  0.97100985 -0.46786416  1.        ]
Last position taught is [-0.4172428  -0.17487381  0.40957287  1.20570898 -0.36048549  0.        ]
Last position taught is [ 0.41417482 -0.0322136   0.3850292   1.26706815  0.40343696  0.        ]
Last position taught is [0.41417482 0.12425245 0.46479619 1.07992256 0.40650493 0.        ]
Recorded gripper open, waypoints now [ 0.41417482  0.12425245  0.46479619  1.07992256  0.40650493 -1.        ]
Last position taught is [ 0.40190297 -0.15646605  0.40190297  1.23025262  0.39730105  0.        ]
Last position taught is [-1.28547597  0.02147573  0.30526218  1.23792255  0.14726216  0.        ]
Last position taught is [-1.27473807  0.14879614  0.44025251  0.9909516   0.20095149  0.        ]
Recorded gripper closed, waypoints now [-1.27473807  0.14879614  0.44025251  0.9909516   0.20095149  1.        ]
Last position taught is [-1.27934003 -0.09817477  0.36968938  1.15508759  0.19021362  0.        ]
Last position taught is [-0.40957287 -0.09050487  0.38196123  1.26400018 -0.44485444  0.        ]
Last position taught is [-0.40343696  0.06289321  0.56910688  0.93419433 -0.45099038  0.        ]
Recorded gripper open, waypoints now [-0.40343696  0.06289321  0.56910688  0.93419433 -0.45099038 -1.        ]
Last position taught is [ 0.08283497  0.04601942 -0.06135923  0.16413595 -0.18867964  0.        ]

[-0.01227185  0.01073787  0.05675729  0.00920388  0.          0.        ],
[ 0.42798066 -0.03374758  0.31139812  1.35757303  0.42184472  0.        ],
[0.42644668 0.1043107  0.49854377 1.03697109 0.43411657 0.        ],
[0.42644668 0.1043107  0.49854377 1.03697109 0.43411657 1.        ],
[ 0.40803891 -0.17794177  0.44178647  1.23178661  0.42031074  0.        ],
[-1.28854394  0.02147573  0.17333983  1.39592254  0.18100974  0.        ],
[-1.26706815  0.13499032  0.44485444  0.98788363  0.18867964  0.        ],
[-1.26706815  0.13499032  0.44485444  0.98788363  0.18867964 -1.        ],
[-1.27013612 -0.16260196  0.42491269  1.16122353  0.19328159  0.        ],
[-0.40650493 -0.07516506  0.37429133  1.28087401 -0.47093213  0.        ],
[-0.40650493  0.06135923  0.55990303  0.97100985 -0.46786416  0.        ],
[-0.40650493  0.06135923  0.55990303  0.97100985 -0.46786416  1.        ],
[-0.4172428  -0.17487381  0.40957287  1.20570898 -0.36048549  0.        ],
[ 0.41417482 -0.0322136   0.3850292   1.26706815  0.40343696  0.        ],
[0.41417482 0.12425245 0.46479619 1.07992256 0.40650493 0.        ],
[ 0.41417482  0.12425245  0.46479619  1.07992256  0.40650493 -1.        ],
[ 0.40190297 -0.15646605  0.40190297  1.23025262  0.39730105  0.        ],
[-1.28547597  0.02147573  0.30526218  1.23792255  0.14726216  0.        ],
[-1.27473807  0.14879614  0.44025251  0.9909516   0.20095149  0.        ],
[-1.27473807  0.14879614  0.44025251  0.9909516   0.20095149  1.        ],
[-1.27934003 -0.09817477  0.36968938  1.15508759  0.19021362  0.        ],
[-0.40957287 -0.09050487  0.38196123  1.26400018 -0.44485444  0.        ],
[-0.40343696  0.06289321  0.56910688  0.93419433 -0.45099038  0.        ],
[-0.40343696  0.06289321  0.56910688  0.93419433 -0.45099038 -1.        ],
[-0.01227185  0.01073787  0.05675729  0.00920388  0.          0.        ]
Done
"""