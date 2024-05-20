import numpy as np
import cv2


class DistanceEstimator:
    def __init__(self):
        self.confThreshold = 0.7  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold
        # 320x320, 416x416 or 608x608 input image in YOLOv3
        self.inpWidth = 320  # Width of input image (to model)
        self.inpHeight = 320  # Height of input image (to model)
        # Classes names
        self.classesFile = 'yolo_configs/coco.names'
        self.classes = None
        self.colors = None
        self.white_list = None
        self.real_width = None
        self.focal_dist = 600
        # Initialize the model, load .cfg and .weights files for YOLOv3-320 from: https://pjreddie.com/darknet/yolo/
        self.modelConfiguration = 'yolo_configs/yolov3.cfg'
        self.modelWeights = 'models/yolov3.weights'
        self.net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.video_source = "examples/rus-video-1.mp4"
        self.video_source = 0
        self.outs = None
        # Parameters of vehicle ahead (bounding box and confidence vector)
        self.vehicle_ahead = {'class_id': None,
                              'confidence': None,
                              'box': [],
                              'center_distance': np.inf,
                              'distance': 20}
        self.N_frames = 5  # YOLO inference every N_frames
        self.cnt = 0  # frame counter (for YOLO inference)
        self.fps = 20

        # Relative speed estimation
        self.relative_speed_array = []
        self.relative_speed = 0

    def load_classes(self):
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        # Define random colour for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')
        # We need only this classes
        self.white_list = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck']
        # Estimate object width (in meters)
        self.real_width = {'person': 0.4,
                           'bicycle': 0.4,
                           'car': 1.8,
                           'motorbike': 0.8,
                           'bus': 2.5,
                           'truck': 2.4}

    # Update FPS (all processing)
    def update_fps(self, fps):
        self.fps = fps

    # Get names of the output layers (layers with unconnected outputs)
    def get_outputs_names(self):
        layers_names = self.net.getLayerNames()
        # return [layers_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return [layers_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def postprocess(self, frame, outs):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        frame_center = [frame_width / 2, frame_height / 2]
        self.vehicle_ahead['class_id'] = None
        self.vehicle_ahead['center_distance'] = np.inf
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        # Draw and append final detections
        for i in indices:
            # i = i[0]  # comment this line if not using CUDA
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            center_x = left + width / 2
            center_y = top - height / 2

            # White list objects only
            if self.classes[class_ids[i]] in self.white_list:
                cur_center_distance = np.linalg.norm(np.array([center_x]) - frame_center[0])
                # Identify vehicle ahead
                if (cur_center_distance < self.vehicle_ahead['center_distance']) and (cur_center_distance < 70):
                    self.vehicle_ahead['center_distance'] = cur_center_distance
                    self.vehicle_ahead['class_id'] = class_ids[i]
                    self.vehicle_ahead['confidence'] = confidences[i]
                    self.vehicle_ahead['box'] = [left, top, width, height]

                self.draw_pred(frame, class_ids[i], left, top, left + width, top + height, width)

        if self.vehicle_ahead['class_id'] is not None:
            left = self.vehicle_ahead['box'][0]
            top = self.vehicle_ahead['box'][1]
            width = self.vehicle_ahead['box'][2]
            height = self.vehicle_ahead['box'][3]
            self.draw_pred(frame, self.vehicle_ahead['class_id'],
                           left, top, left + width, top + height, width,
                           vehicle_ahead=True)

    # Calculate relative speed (in km/h)
    def get_relative_speed(self, cur_distance):
        return min(3.6 * (self.vehicle_ahead['distance'] - cur_distance) / (self.N_frames / self.fps), 60)

    def min_distance_criterion(self):
        return 10 * (self.relative_speed / 30) ** 2 + self.relative_speed / 3.6 + 5

    def rec_distance_criterion(self):
        return 20 * (self.relative_speed / 30) ** 2 + self.relative_speed / 3.6 + 15

    def draw_pred(self, frame, class_id, left, top, right, bottom, width, vehicle_ahead=False):
        # Distance estimation (in meters)
        real_w = self.real_width[self.classes[class_id]]
        pix_w = width
        distance = (real_w * self.focal_dist) / pix_w

        # Bounding box
        color = (
            int(self.colors[int(class_id)][0]), int(self.colors[int(class_id)][1]), int(self.colors[int(class_id)][2]))

        if vehicle_ahead:
            color = (0, 255, 0)
            # Relative speed estimation (in km/h)
            relative_speed = self.get_relative_speed(distance)
            if relative_speed > 1:
                self.relative_speed_array.append(relative_speed)

            # Update distance to vehicle ahead
            self.vehicle_ahead['distance'] = distance

            if self.vehicle_ahead['distance'] < self.rec_distance_criterion():
                color = (0, 175, 255)
            if self.vehicle_ahead['distance'] < self.min_distance_criterion():
                color = (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 1)

        # label = f'{self.classes[class_id]} {conf:.2f}' # Draw class and confidence score
        label = f'{self.classes[class_id]} {distance:.1f} m'  # Draw class and estimated distance (in meters)
        # Label at the top of bounding box
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        text_w, text_h = text_size
        cv2.rectangle(frame, (left, top), (left + text_w, top - text_h),
                      color, cv2.FILLED)
        cv2.putText(frame, label,
                    (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    def operate(self, frame):
        self.cnt += 1
        if self.cnt == self.N_frames:
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)
            self.net.setInput(blob)
            self.outs = self.net.forward(self.get_outputs_names())

            self.cnt = 0
        try:
            self.postprocess(frame, self.outs)
            # Mean relative speed recalculation
            if len(self.relative_speed_array) * (1 / self.fps) > 1:
                self.relative_speed = np.mean(self.relative_speed_array)
                self.relative_speed_array = []
                #  print(f"Relative speed: {self.relative_speed}")

        except:
            pass
        return frame, self.vehicle_ahead
