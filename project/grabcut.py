import cv2
import numpy as np


class GrabCut:
    def __init__(self, image: np.ndarray):
        self.image = image
        # Create an initial mask with zeros (all background)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        # Create a window for user interaction
        self.window_name = 'Mark Foreground'

        self.foreground = None
        self.background = None

    # Function to perform GrabCut segmentation
    def _perform_grabcut(self) -> np.ndarray:
        foreground = np.zeros((1, 65), np.float64)
        background = np.zeros((1, 65), np.float64)

        # Run GrabCut algorithm
        cv2.grabCut(
            img=self.image,
            mask=self.mask,
            rect=None,
            bgdModel=background,
            fgdModel=foreground,
            iterCount=5,
            mode=cv2.GC_INIT_WITH_RECT
        )

        # Assign definite background and probable background as 0, others as 1
        mask = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')

        # Apply the mask to the input image
        segmented_image = self.image * mask[:, :, np.newaxis]

        return segmented_image

    def run(self):
        cv2.namedWindow(self.window_name)

        # Display the image and wait for user interaction
        cv2.imshow(self.window_name, self.image)
        # Prompt the user to select the object by drawing a bounding box
        object_box = cv2.selectROI(self.window_name, self.image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(self.window_name)

        # Perform GrabCut segmentation based on the user markings
        segmented_image = self._perform_grabcut()

        # Create a black background image with the same size
        background = np.zeros_like(self.image)

        # Display the segmented foreground and background images
        cv2.imshow('Foreground', segmented_image)
        cv2.imshow('Background', background)
        cv2.waitKey(0)

        self.foreground = segmented_image
        self.background = background

        # Close all windows
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # test grabcut
    image = cv2.imread('images/tinky_winky.jpeg')
    grabcut = GrabCut(image)
    grabcut.run()
    print(grabcut.foreground)
    print(grabcut.background)
    cv2.imwrite('images/tinky_winky_foreground.jpg', grabcut.foreground)
    cv2.imwrite('images/tinky_winky_background.jpg', grabcut.background)

