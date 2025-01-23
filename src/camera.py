#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.camera_calibrated = False
        # self.intrinsic_matrix = np.array([[900.7150268554688, 0, 652.2869262695312], 
        #                               	[0, 900.1925048828125, 358.359619140625], 
        #                             	[0, 0, 1]])
        self.distortion = np.array([0.1490122675895691, -0.5096240639686584, -0.0006352968048304319, 
                                    0.0005230441456660628, 0.47986456751823425])
        self.H = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.last_click = np.array([0, 0]) # This contains the last clicked position
        self.new_click = False # This is automatically set to True whenever a click is received. Set it to False yourself after processing a click
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275], [-250, 275]]  # In order of tag id 1 thru 4
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

        # Intrinsic matrix as calibrated using the checkerboard in Checkpoint 1, Task 4
        self.intrinsic_matrix = np.array([[898.1038628, 0, 644.0920518], 
                                      [0, 900.632657, 340.2840602], 
                                      [0, 0, 1]])
        
        # Extrinsic matrix as physically measured in Checkpoint 1, Task 5
        theta = 188
        self.extrinsic_matrix = np.array([[1, 0, 0, 10], 
                                       [0, np.cos(theta*np.pi/180), -np.sin(theta*np.pi/180), 155], 
                                       [0, np.sin(theta*np.pi/180), np.cos(theta*np.pi/180), 1035],
                                       [0, 0, 0, 1]])           # rotate 172 degree along x axis CCW
        self.projected_tag_locations_2d = [[390, 515], [890, 515], [890, 215], [390, 215]]

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            hom_frame = cv2.warpPerspective(self.VideoFrame, self.H, (self.VideoFrame.shape[1], self.VideoFrame.shape[0]))
            frame = cv2.resize(hom_frame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """

        self.intrinsic_matrix = np.array([[898.1038628, 0, 644.0920518], 
                                      [0, 900.632657, 340.2840602], 
                                      [0, 0, 1]])
        self.extrinsic_matrix = np.array([[1, 0, 0, 0], 
                                       [0, np.cos(173*np.pi/180), -np.sin(173*np.pi/180), 335], 
                                       [0, np.sin(173*np.pi/180), np.cos(173*np.pi/180), 990],
                                       [0, 0, 0, 1]])           # rotate 173 degree along x axis CCW
        
    def transformCoordinate_pixel2world(self, u, v, z):
        """
        @brief      Transforms pixel coordinates into the world frame
        
        """
        mouse_coords = self.H @ np.array([u, v, 1]).reshape(3, 1)
        camera_frame = z * np.linalg.inv(self.intrinsic_matrix) @ mouse_coords # np.array([u, v, 1]).reshape(3, 1)
        world_frame = np.linalg.inv(self.extrinsic_matrix) @ np.append(camera_frame, 1)
        return world_frame[:-1]

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here

        self.GridFrame = modified_image
     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()

        # Write your code here
        # Include tag ID, highlight tag center, tag edges
        for detection in msg.detections:

            tag_id = detection.id

            center_x = detection.centre.x
            center_y = detection.centre.y
            corners = detection.corners

            int_corners = []
            for corner in corners:
              int_corner_x = int(corner.x)
              int_corner_y = int(corner.y)
              int_corners.append((int_corner_x, int_corner_y))
             
             
            # Draw bounding box for AprilTag
            cv2.polylines(modified_image,
                          np.int32([int_corners]),
                          True,
                          (0, 0, 255), 2
                        )

            # Put text for AprilTag ID
            cv2.putText(modified_image,
                      f"ID: {str(tag_id)}",
                      (int(center_x - 25), int(center_y - 50)),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                      )       

            # Highlight AprilTag center
            cv2.circle(modified_image,
                      (int(center_x), int(center_y)),
                      2, 
                      (0, 255, 0), 2
                      )

        self.TagImageFrame = modified_image
        
    def recover_homogeneous_transform_pnp(self, msg):
        """
        Returns an extrinsic matrix after performing PnP pose computation
        """
        image_points_raw = {}
        for detection in msg.detections:
            tag_id = detection.id
            center_x = detection.centre.x
            center_y = detection.centre.y
            image_points_raw[tag_id] = ([center_x, center_y])
            
            if len(image_points_raw) == 4:
                break
           
        # Debugging - print(image_points_raw) 
        
        # Converting known apriltag grid locations into 3D points by appending zeros for the z-axis
        world_points_interm = np.array(self.tag_locations)
        world_points_3d = np.float32(np.column_stack((world_points_interm, np.zeros(4))))
        # Converting image points into int32 representations for cv2
        image_points = np.float32(list(image_points_raw.values()))    
        
        # Debugging
        # print(type(world_points_3d))
        # print(type(image_points))
        
        # Solving for PnP rotation and transformation
        [_, R_exp, t] = cv2.solvePnP(world_points_3d,
                        			image_points,
                                	self.intrinsic_matrix,
                                 	self.distortion,
                                 	flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(R_exp)
        return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))
    
    def homography_transform(self, msg):
        """ 
        Performs the homography transformation to warp the trapezoid view of the board into a rectangular view
        """
        modified_image = self.VideoFrame.copy()
        
        image_points_raw = {}
        for detection in msg.detections:
            tag_id = detection.id
            center_x = detection.centre.x
            center_y = detection.centre.y
            image_points_raw[tag_id] = ([center_x, center_y])
            
            if len(image_points_raw) == 4:
                break
            
        desired_image_points = np.float32(np.array(self.projected_tag_locations_2d))
        # Converting image points into int32 representations for cv2
        image_points = np.float32(list(image_points_raw.values())) 
        
        H = cv2.findHomography(image_points, desired_image_points)[0]
        # modified_image = cv2.warpPerspective(modified_image, H, (modified_image.shape[1], modified_image.shape[0]))
        self.H = H
        

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)
            # Checkpoint 2 Task 2.2 - Autocalibrates the extrinsic matrix based on apriltag detections
            self.camera.extrinsic_matrix = self.camera.recover_homogeneous_transform_pnp(msg)
            self.camera.homography_transform(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    