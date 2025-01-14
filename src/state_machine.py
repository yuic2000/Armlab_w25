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
        """ self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]] """
        
        self.waypoints = [

        ]
        
        # List for storing taught waypoints
        self.taught_waypts = []

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

    """
    State for teach-and-repeat in ArmLab Task 1.3
    """
    def record_positions(self):
        self.current_state = "record_positions"
        self.status_message = "Recording positions from manual control"
        cur_time = time.time()
        i = 0

        while (time.time() - cur_time < 80):
            self.taught_waypts.append(self.rxarm.get_positions())
            print(f"The Last position {self.taught_waypts[-1]} at step {i}")
            time.sleep(5)
            i += 1

        print("Done")
        self.waypoints = self.taught_waypts
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
The Last position [0.58444667 0.05982525 0.28992239 1.17809725 0.48166999]
The Last position [0.58444667 0.22242722 0.29452431 1.07992256 0.47553405]
The Last position [0.58751464 0.03834952 0.0966408  1.36831093 0.47400007]
The Last position [-1.26860213  0.06902914  0.18561168  1.37598085 -1.26553416]
The Last position [-1.26860213  0.17947575  0.36968938  1.06765068 -1.27934003]
The Last position [-1.26860213 -0.00920388  0.08436894  1.51557302 -1.26553416]
The Last position [-0.43411657 -0.10277671  0.41264084  1.20264101 -0.41264084]
The Last position [-0.43565056 -0.14572819  0.41417482  1.24099052 -0.41417482]
The Last position [-0.44792241  0.06596117  0.53689331  0.9664079  -0.41570881]
The Last position [-0.44638842 -0.10124274  0.31139812  1.29621375 -0.42337871]
The Last position [-0.04601942 -0.73477679  0.68568945  0.20095149  0.19788353]
"""

"""
The Last position [-0.01227185  0.01073787  0.05829127  0.00920388  0.        ] at step 0
The Last position [-0.01073787  0.09510681  0.06902914  0.99862152  0.        ] at step 1
The Last position [0.57524282 0.06135923 0.28071851 1.19497108 0.43871853] at step 2
The Last position [0.57370883 0.21015537 0.31600004 1.06918466 0.44638842] at step 3
The Last position [0.57524282 0.00153398 0.23776703 1.32382548 0.44332045] at step 4
The Last position [-1.28394198  0.04755341  0.25464082  1.33302939 -1.360641  ] at step 5
The Last position [-1.27320409  0.17947575  0.41110685  1.0262332  -1.38058269] at step 6
The Last position [-1.26860213 -0.0322136   0.27918452  1.32229149 -1.37444687] at step 7
The Last position [-0.44332045 -0.05368933  0.38963112  1.21491277 -0.41264084] at step 8
The Last position [-0.44945639  0.11504856  0.46786416  1.04003906 -0.41417482] at step 9
The Last position [-0.42184472 -0.39269909  0.53075737  1.11673808 -0.44025251] at step 10
The Last position [-0.02454369 -0.82374769  0.6948933   0.1672039  -0.04295146] at step 11
The Last position [-0.01840777 -0.84522343  0.74704868  0.16873789 -0.04295146] at step 12
The Last position [-0.01840777 -0.84522343  0.74704868  0.16873789 -0.04295146] at step 13
The Last position [-0.01994175 -0.84522343  0.74704868  0.16873789 -0.04295146] at step 14
The Last position [-0.01994175 -0.84522343  0.74704868  0.16873789 -0.04295146] at step 15
Done
"""