import cv2
from PIL import Image


# Extracts frames from a video file and returns them as a list
# The list len is the count of frames
# Each object shape is (Frame height, Frame width, 3)
def extract_frames(video):
    frames = []

    # Read frames until the video ends
    while True:
        # Read the current frame
        ret, frame = video.read()

        # If the frame was read successfully
        if ret:
            # Add the frame to the list
            frames.append(frame)

            # Show the frame
            # img = Image.fromarray(frame.astype('uint8'), 'RGB')
            # img.show()
        else:
            break

    # Release the video object
    video.release()

    return frames


if __name__ == '__main__':
    # Open the video file
    video = cv2.VideoCapture('videos/river.mp4')

    # Extract frames from the video
    frame_list = extract_frames(video)

    # Print the number of frames extracted
    print("Number of frames:", len(frame_list))