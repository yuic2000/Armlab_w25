import numpy as np
import matplotlib.pyplot as plt

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# Create a typestore and get the string class.
typestore = get_typestore(Stores.LATEST)

# Create reader instance and open for reading.
with Reader('/home/student_550_am/rosbag2_2025_01_17-14_06_23') as reader:
   # Topic and msgtype information is available on .connections list.
   for connection in reader.connections:
       print(connection.topic, connection.msgtype)

   # Iterate over messages.
   joint_angles_matrix = []
   for connection, timestamp, rawdata in reader.messages():
       if connection.topic == '/rx200/joint_states':
           msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
           joint_angles_matrix.append(msg.position)

   joint_angles_matrix = np.array(joint_angles_matrix)

   plt.figure(figsize=(10, 5))
   joints = ["waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate", "gripper", 'left_finger', 'right_finger']
   time = np.arange(joint_angles_matrix.shape[0])
   for i in range(len(joints)):
       plt.plot(time, joint_angles_matrix[:, i], label=str(joints[i]))

   plt.xlabel("Time")
   plt.ylabel("Joint angles")
   plt.title("Plot of Joint angles over time")
   plt.legend()
   plt.show()
