## What is the place for?

This directory is for storing launch files, which can be used to start multiple nodes and configure the ROS2 system.


### To start/test single node
1. Start Realsense2 node
```
ros2 launch realsense2_camera rs_l515_launch.py
```

2. Start AprilTag Dectection node (if needed)
```
ros2 run apriltag_ros apriltag_node --ros-args \
    -r image_rect:=/camera/color/image_raw \
    -r camera_info:=/camera/color/camera_info \
    --params-file `ros2 pkg prefix apriltag_ros`/share/apriltag_ros/cfg/tags_Standard41h12.yaml
```

3. Test the arm node
```
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=rx200
```
- This command will launch rviz with the virtual robot model, the model would show exactly how the arm moving.

```
ros2 service call /rx200/torque_enable interbotix_xs_msgs/srv/TorqueEnable "{cmd_type: 'group', name: 'all', enable: false}"
```
- This command is used to torque off all the motors so arm can be manually manipulated. 

```
ros2 service call /rx200/torque_enable interbotix_xs_msgs/srv/TorqueEnable "{cmd_type: 'group', name: 'all', enable: true}"
```
- This command is used to torque on all the motors so arm can hold a pose you made

### To launch everything
Open a terminal and navigate to the current folder. Then, run the following command:
```
./launch_armlab.sh
```
```
./launch_control_station.sh
```

## Why I cannot run the file?
1. The .sh file is not executable

![](/media/chmod.png)
