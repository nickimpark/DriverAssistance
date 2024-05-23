import cv2
from timeit import default_timer as timer

from lane_detector import LaneDetector
from distance_estimator import DistanceEstimator

if __name__ == "__main__":
    video_input = "examples/test_unclear.mp4"
    # video_input = 0
    lane_detector = LaneDetector()
    distance_estimator = DistanceEstimator()
    distance_estimator.load_classes()

    cap = cv2.VideoCapture(video_input)
    # press Q or Esc to stop
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv2.waitKey(3000)
            # Release device
            cap.release()
            break

        start = timer()
        h, w = frame.shape[:2]
        frame = frame[0:h - 0, 0:w]
        frame = cv2.resize(frame, (1280, 720))
        distance_output, vehicle_ahead = distance_estimator.operate(frame)
        try:
            combined_output = lane_detector.operate(distance_output, vehicle_ahead)
            # combined_output = distance_output
        except:
            combined_output = distance_output

        end = timer()
        fps = 1.0 / (end - start)
        distance_estimator.update_fps(fps)
        info_framerate = "{0:4.1f} FPS".format(fps)
        cv2.putText(combined_output, info_framerate, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Lane and Distance Control', combined_output)
