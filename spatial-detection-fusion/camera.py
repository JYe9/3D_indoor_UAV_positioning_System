from pathlib import Path

import depthai as dai
import cv2
import numpy as np
from typing import List
from detection import Detection
import config
import os
import uuid
import datetime


def find_largest_contour(edgeFrame, bbox):
    thresh = cv2.threshold(edgeFrame[bbox['y_min']:bbox['y_max'],
                           bbox['x_min']:bbox['x_max']], 25, 255, cv2.THRESH_BINARY)[1]
    res = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[-2]

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        result = np.zeros_like(edgeFrame)
        cv2.drawContours(result[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']], [
                         largest_contour], 0, (255, 255, 255), cv2.FILLED)

        rect = cv2.minAreaRect(largest_contour)
        center, _, _ = rect
        center_x, center_y = center

        edgeFrame = cv2.bitwise_or(edgeFrame, result)

        return edgeFrame, center_x + bbox['x_min'], center_y + bbox['y_min']
    else:
        return edgeFrame, -999, -999


class Camera:
    """
    REPLACE THE LABLE OF THE DETECTION MODEL HERE!!! 
    AVAILABLE LABLES ARE: 
    ["GV"] for Ground Vehicle
    ["UAV"] for Drone, if you want to change to detect UAV change the code below to --> label_map = ["UAV"]
    """
    label_map = ["0"]

    NN_IMG_SIZE = 640

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int, show_video: bool = True):
        self.show_video = show_video
        self.show_detph = False
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.rgb_queue = self.device.getOutputQueue(
            name="rgb", maxSize=1, blocking=False)
        self.still_queue = self.device.getOutputQueue(
            name="still", maxSize=1, blocking=False)
        self.control_queue = self.device.getInputQueue(name="control")
        self.nn_queue = self.device.getOutputQueue(
            name="nn", maxSize=1, blocking=False)
        self.depth_queue = self.device.getOutputQueue(
            name="depth", maxSize=1, blocking=False)
        self.edge_queue = self.device.getOutputQueue("edge", 8, False)
        self.window_name = f"[{self.friendly_id}] Camera - mxid: {self.mxid}"
        if show_video:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 360)

        self.frame_rgb = None
        self.detected_objects: List[Detection] = []

        self._load_calibration()

        print(
            "=== Connected to Camera[{}] - mxid: {}".format(self.friendly_id, self.mxid))

    def __del__(self):
        self.device.close()
        print(
            "=== Closed Camera[{}] - mxid: {}".format(self.friendly_id, self.mxid))

    def _load_calibration(self):
        path = os.path.join(os.path.dirname(__file__),
                            f"{config.calibration_data_dir}")
        try:
            extrinsics = np.load(f"{path}/extrinsics_{self.mxid}.npz")
            self.cam_to_world = extrinsics["cam_to_world"]
        except:
            self.cam_to_world = None
            raise RuntimeError(
                f"Could not load calibration data for camera {self.mxid} from {path}!")

    def _create_pipeline(self):
        """
        REPLACE THE MODEL HERE!!!
        For detect UAV change the code below to              --> blob = Path(__file__).parent.joinpath("models", "UAV_openvino_2022.1_6shave.blob")
        For detect ground vehicle change the code to         --> blob = Path(__file__).parent.joinpath("models", "GVonly_openvino_2022.1_6shave.blob")
        To detect both UAV and GV but only shows GV as lable --> blob = Path(__file__).parent.joinpath("models", "GV_openvino_2022.1_6shave.blob")
        """
        blob = Path(__file__).parent.joinpath(
            "models", "fire_openvino_2022.1_6shave.blob")
        model = dai.OpenVINO.Blob(blob)
        dim = next(iter(model.networkInputs.values())).dims
        W, H = dim[:2]
        self.NN_IMG_SIZE = W

        anchors = [
            10, 13, 16, 30, 33, 23,
            30, 61, 62, 45, 59, 119,
            116, 90, 156, 198, 373, 326
        ]
        anchorMasks = {
            "side52": [0, 1, 2],
            "side26": [3, 4, 5],
            "side13": [6, 7, 8]
        }

        pipeline = dai.Pipeline()

        # RGB cam -> 'rgb'
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setPreviewSize(W, H)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewKeepAspectRatio(False)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")

        # Depth cam -> 'depth'
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_stereo = pipeline.create(dai.node.StereoDepth)
        cam_stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        cam_stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        cam_stereo.setOutputSize(
            mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
        mono_left.out.link(cam_stereo.left)
        mono_right.out.link(cam_stereo.right)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")

        # Spatial detection network -> 'nn'
        spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        spatial_nn.setBlob(model)
        spatial_nn.setConfidenceThreshold(0.1)  # confidence threshold setting

        # Yolo specific parameters
        spatial_nn.setNumClasses(1)
        spatial_nn.setCoordinateSize(4)
        spatial_nn.setAnchors(anchors)
        spatial_nn.setAnchorMasks(anchorMasks)
        spatial_nn.setIouThreshold(0.5)

        spatial_nn.input.setBlocking(False)
        spatial_nn.setBoundingBoxScaleFactor(0.2)
        # Define the minimum detectable distance
        spatial_nn.setDepthLowerThreshold(100)
        # Maximum detecable Distance define 5000
        spatial_nn.setDepthUpperThreshold(10000)
        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")

        cam_rgb.preview.link(spatial_nn.input)
        # cam_rgb.preview.link(xout_rgb.input)
        cam_stereo.depth.link(spatial_nn.inputDepth)
        spatial_nn.passthrough.link(xout_rgb.input)
        spatial_nn.passthroughDepth.link(xout_depth.input)
        spatial_nn.out.link(xout_nn.input)

        # Still encoder -> 'still'
        still_encoder = pipeline.create(dai.node.VideoEncoder)
        still_encoder.setDefaultProfilePreset(
            1, dai.VideoEncoderProperties.Profile.MJPEG)
        cam_rgb.still.link(still_encoder.input)
        xout_still = pipeline.createXLinkOut()
        xout_still.setStreamName("still")
        still_encoder.bitstream.link(xout_still.input)

        # Camera control -> 'control'
        control = pipeline.create(dai.node.XLinkIn)
        control.setStreamName('control')
        control.out.link(cam_rgb.inputControl)

        edgeDetectorRgb = pipeline.createEdgeDetector()
        edgeManip = pipeline.createImageManip()

        xoutEdge = pipeline.createXLinkOut()
        xoutEdge.setStreamName("edge")

        edgeDetectorRgb.setMaxOutputFrameSize(
            cam_rgb.getVideoWidth() * cam_rgb.getVideoHeight())
        edgeManip.initialConfig.setResize(W, H)

        cam_rgb.video.link(edgeDetectorRgb.inputImage)
        edgeDetectorRgb.outputImage.link(edgeManip.inputImage)
        edgeManip.out.link(xoutEdge.input)

        self.pipeline = pipeline

    def update(self):
        in_rgb = self.rgb_queue.tryGet()
        in_nn = self.nn_queue.tryGet()
        in_depth = self.depth_queue.tryGet()
        edgeFrame = self.edge_queue.get().getCvFrame()
        global x_angle_offset, y_angle_offset  # global variables

        if in_rgb is None or in_depth is None:
            return

        depth_frame = in_depth.getFrame()  # depthFrame values are in millimeters
        depth_frame_color = cv2.normalize(
            depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depth_frame_color = cv2.equalizeHist(depth_frame_color)
        depth_frame_color = cv2.applyColorMap(
            depth_frame_color, cv2.COLORMAP_HOT)

        self.frame_rgb = in_rgb.getCvFrame()

        if self.show_detph:
            visualization = depth_frame_color.copy()
        else:
            visualization = self.frame_rgb.copy()
        visualization = cv2.resize(
            visualization, (640, 360), interpolation=cv2.INTER_NEAREST)

        height = visualization.shape[0]
        width = visualization.shape[1]

        detections = []
        if in_nn is not None:
            detections = in_nn.detections

        self.detected_objects = []

        for detection in detections:
            roi = detection.boundingBoxMapping.roi
            roi = roi.denormalize(width, height)
            top_left = roi.topLeft()
            bottom_right = roi.bottomRight()
            xmin = int(top_left.x)
            ymin = int(top_left.y)
            xmax = int(bottom_right.x)
            ymax = int(bottom_right.y)

            bbox = {
                'id': uuid.uuid4(),
                'label': detection.label,
                'confidence': detection.confidence,
                'x_min': int(detection.xmin * edgeFrame.shape[1]),
                'x_mid': int(((detection.xmax - detection.xmin) / 2 + detection.xmin) * edgeFrame.shape[1]),
                'x_max': int(detection.xmax * edgeFrame.shape[1]),
                'y_min': int(detection.ymin * edgeFrame.shape[0]),
                'y_mid': int(((detection.ymax - detection.ymin) / 2 + detection.ymin) * edgeFrame.shape[0]),
                'y_max': int(detection.ymax * edgeFrame.shape[0]),
                'depth_x': detection.spatialCoordinates.x / 1000,
                'depth_y': detection.spatialCoordinates.y / 1000,
                'depth_z': detection.spatialCoordinates.z / 1000,
            }

            edgeFrame, target_x, target_y = find_largest_contour(
                edgeFrame, bbox)

            if target_x != -999 and target_y != -999:
                # angle_offset = (target_x - (self.NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 1920
                x_angle_offset = (
                    target_x - (self.NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 640
                y_angle_offset = - \
                    (target_y - (self.NN_IMG_SIZE / 2.0)) * 68.7938003540039 / 640
                x_distance = int(detection.spatialCoordinates.x)
                y_distance = int(detection.spatialCoordinates.y)
                z_distance = int(detection.spatialCoordinates.z)

                # Get the current date and time
                now = datetime.datetime.now()

                # Format the timestamp
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

                # Detection output
                """
                WE ONLY CARE ABOUT THE HORIZONTAL ANGLE, VERTICAL ANGLE AND THE DISTANCE Z. THE X AND Y IS THE DISTANCE BETWEEN THE CENTRAL POINT OF THE CAMERA AND THE TARGET.
                """
                print("[{}] [{}] Camera: \u2220Horizontal:{:.3f}\N{DEGREE SIGN} \u2220Vertical:{:.3f}\N{DEGREE SIGN} distance(z):{}mm x:{}mm y:{}mm".format(
                    timestamp, self.friendly_id, x_angle_offset, y_angle_offset, z_distance, x_distance, y_distance))

                cv2.rectangle(edgeFrame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']),
                              (255, 255, 255), 2)

                cv2.circle(edgeFrame, (int(round(target_x, 0)), int(round(target_y, 0))), radius=5, color=(128, 128, 128),
                           thickness=-1)
                bbox['target_x'] = target_x
                bbox['target_y'] = target_y
                bbox['angle_offset'] = x_angle_offset

            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            try:
                label = self.label_map[detection.label]
            except:
                label = detection.label

            if self.cam_to_world is not None:
                pos_camera_frame = np.array(
                    [[detection.spatialCoordinates.x / 1000, -detection.spatialCoordinates.y / 1000, detection.spatialCoordinates.z / 1000, 1]]).T

                pos_world_frame = self.cam_to_world @ pos_camera_frame

                self.detected_objects.append(
                    Detection(label, detection.confidence, pos_world_frame, self.friendly_id))

            cv2.rectangle(visualization, (xmin, ymin),
                          (xmax, ymax), (100, 0, 0), 2)
            cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(visualization, str(label), (x1 + 10, y1 + 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            cv2.putText(visualization, "conf:{:.2f}%".format(
                detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
            cv2.putText(visualization, f"X: {int(detection.spatialCoordinates.x)} mm", (
                x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
            cv2.putText(visualization, f"Y: {int(detection.spatialCoordinates.y)} mm", (
                x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
            cv2.putText(visualization, f"Z: {int(detection.spatialCoordinates.z)} mm", (
                x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
            cv2.putText(visualization, "x_angle:{:.3f}degree".format(
                x_angle_offset), (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
            cv2.putText(visualization, "y_angle:{:.3f}degree".format(
                y_angle_offset), (x1 + 10, y1 + 110), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))

        if self.show_video:
            cv2.imshow(self.window_name, visualization)
