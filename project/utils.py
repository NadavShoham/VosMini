import cv2
import numpy as np
from typing import List


def extract_frames(video_path: str) -> List[np.ndarray]:
    video = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = video.read()
        if ret:
            frames.append(frame)
        else:
            break

    video.release()
    return frames


def build_video_from_frames(frames: List[np.ndarray], output_path: str, show_video: bool = False):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    if show_video:
        video_capture = cv2.VideoCapture(output_path)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        video_capture.release()
        cv2.destroyAllWindows()


def resize_frames(frames: List[np.ndarray], ratio: float) -> List[np.ndarray]:
    return [cv2.resize(frame, (0, 0), fx=ratio, fy=ratio) for frame in frames]


