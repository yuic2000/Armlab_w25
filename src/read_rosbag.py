import rosbag

with rosbag.Bag('/home/student_550_am/rosbag2_2025_01_17-14_06_23', 'r') as bag:
    for topic, msg, t in bag.read_messages():
        # Process the message here
        print(topic, msg, t)