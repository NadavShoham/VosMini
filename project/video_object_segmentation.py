import cv2
import numpy as np
from grabcut import GrabCut
from utils import extract_frames, build_video_from_frames, resize_frames
from julia.api import Julia
from dpmmpythonStreaming.dpmmwrapper import DPMMPython
from dpmmpythonStreaming.priors import niw
from gaussian_mixture import GaussianMixture
from tqdm import tqdm
jl = Julia(compiled_modules=False)


def extract_gaussian_data_and_create_mixture(model):
    clusters = model.group.local_clusters
    means = []
    covariances = []
    for cluster in clusters:
        distribution = cluster.cluster_params.cluster_params.distribution
        means.append(distribution.μ)
        covariances.append(distribution.Σ)
    return GaussianMixture(means, covariances)


def adjust_frame_for_algorithm(frame: np.ndarray, include_xy: bool = False, xy_factor: float = 1) -> np.ndarray:
    shape = frame.shape
    adjusted_frame = frame
    if include_xy:
        shape = (shape[0], shape[1], shape[2] + 2)
        adjusted_frame = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                adjusted_frame[i, j] = np.append(frame[i, j], [i * xy_factor, j * xy_factor])
    adjusted_frame = adjusted_frame.reshape(shape[2], shape[0] * shape[1]).astype(np.float32)
    return adjusted_frame


class VideoObjectSegmentation:
    def __init__(self, name: str):
        self.name = "wolf"
        self.video_path = f"data/videos/{name}.mp4"
        self.fg = None
        self.bg = None

        self.fgm = None
        self.bgm = None
        self.verbose = False
        self.save_images = False
        self.frame_idx = 0

    def _classify_first_frame(self, frame: np.ndarray, transparency: float = 0.2) -> np.ndarray:
        if self.verbose:
            print("Recreating first frame with foreground coloring...")
        foreground_coloring = np.zeros(frame.shape)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if np.array_equal(frame[i, j], self.fg[i, j]):
                    foreground_coloring[i, j] = [0, 255, 0]

        marked_frame = cv2.addWeighted(foreground_coloring, transparency, frame, 1 - transparency, 0, dtype=cv2.CV_8U)
        if self.verbose:
            print("Done.")
        if self.save_images:
            cv2.imwrite(f"data/images/{self.name}/frame{self.frame_idx}.png", marked_frame)
        return marked_frame

    def _classify_pixels_and_adjust_for_algorithm(
            self,
            frame: np.ndarray,
            transparency: float = 0.2,
            include_xy: bool = False,
            xy_factor: float = 1.0
    ) -> (np.ndarray, np.ndarray, np.ndarray):

        # classification set up
        foreground_coloring = np.zeros(frame.shape)
        foreground = np.zeros(frame.shape)
        background = np.zeros(frame.shape)

        # algorithm frame adjustment set up
        shape = frame.shape
        adjusted_fg = foreground
        adjusted_bg = background
        if include_xy:
            shape = (shape[0], shape[1], shape[2] + 2)
            adjusted_fg = np.zeros(shape)
            adjusted_bg = np.zeros(shape)

        # using the same loop for 2 purposes to save time
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                pixel = frame[i, j]
                extended_pixel = np.append(pixel, [i * xy_factor, j * xy_factor])
                pixel_to_sample = pixel if not include_xy else extended_pixel
                if self.fgm.pdf(pixel_to_sample) > self.bgm.pdf(pixel_to_sample):
                    foreground[i, j] = pixel
                    foreground_coloring[i, j] = [0, 255, 0]
                    if include_xy:
                        adjusted_fg[i, j] = extended_pixel
                else:
                    background[i, j] = pixel
                    if include_xy:
                        adjusted_bg[i, j] = extended_pixel

        # classification
        marked_frame = cv2.addWeighted(foreground_coloring, transparency, frame, 1 - transparency, 0, dtype=cv2.CV_8U)
        if self.save_images:
            cv2.imwrite(f"data/images/{self.name}/frame{self.frame_idx}.png", marked_frame)
            cv2.imwrite(f"data/images/{self.name}/frame{self.frame_idx}_foreground.png", foreground)
            cv2.imwrite(f"data/images/{self.name}/frame{self.frame_idx}_background.png", background)

        # algorithm frame adjustment
        self.fg, self.bg = adjusted_fg, adjusted_bg
        adjusted_fg = adjusted_fg.reshape(shape[2], shape[0] * shape[1]).astype(np.float32)
        adjusted_bg = adjusted_bg.reshape(shape[2], shape[0] * shape[1]).astype(np.float32)

        return adjusted_fg, adjusted_bg, marked_frame

    def _get_foreground_background(self, frame: np.ndarray, show: bool = False) -> (np.ndarray, np.ndarray):
        if self.verbose:
            print("Getting foreground and background of first frame...")

        first_frame_foreground = cv2.imread(f"data/images/{self.name}_first_frame_foreground.png")
        first_frame_background = cv2.imread(f"data/images/{self.name}_first_frame_background.png")

        if first_frame_foreground is None or first_frame_background is None:

            if self.verbose:
                print("No foreground and background found, running GrabCut...")

            grab_cut = GrabCut(frame)
            grab_cut.run(show)

            first_frame_foreground, first_frame_background = grab_cut.foreground, grab_cut.background

            cv2.imwrite(f"data/images/{self.name}_first_frame_foreground.png", first_frame_foreground)
            cv2.imwrite(f"data/images/{self.name}_first_frame_background.png", first_frame_background)

            if self.verbose:
                print(f"GrabCut done. Saved first frame foreground and background of {self.name} to data/images/")

        return first_frame_foreground, first_frame_background

    def segment(self,
                frames_num: int = 100,
                iters_of_fit_partial: int = 10,
                resize: bool = True,
                show_images: bool = False,
                save_images: bool = False,
                show_video: bool = False,
                verbose: int = 0,
                include_xy: bool = False,
                xy_factor: float = 1.0
                ):
        self.verbose = verbose
        self.save_images = save_images
        self.frame_idx = 0
        frames = extract_frames(self.video_path)
        if resize:
            frames = resize_frames(frames, 0.3)
        shape = frames[0].shape

        if self.verbose:
            print(f"Loaded {len(frames)} frames from {self.video_path} with size {shape}")

        self.fg, self.bg = self._get_foreground_background(frames[0], show=show_images)
        new_frames = [self._classify_first_frame(frames[0])]

        if self.verbose:
            print("Initializing models prerequisites...")
        frames = frames[1:frames_num]
        self.frame_idx += 1
        adjusted_fg = adjust_frame_for_algorithm(self.fg, include_xy=include_xy, xy_factor=xy_factor)
        adjusted_bg = adjust_frame_for_algorithm(self.bg, include_xy=include_xy, xy_factor=xy_factor)
        # Initialize the models
        pixel_dim = adjusted_fg.shape[0]
        # # TODO: optimize niw parameters like in or dinaris julia example
        prior = niw(1, np.zeros(pixel_dim), pixel_dim, np.eye(pixel_dim) * 0.5)
        alpha = 10.0
        epsilon = 0.0000001

        if self.verbose:
            print("Initializing models...")
            print("Running fit_init on first frame foreground and background...")
        models_verbose = verbose > 1
        bg_model = DPMMPython.fit_init(
            data=adjusted_fg, alpha=alpha, prior=prior, verbose=models_verbose, burnout=5, gt=None, epsilon=epsilon
        )
        fg_model = DPMMPython.fit_init(
            data=adjusted_bg, alpha=alpha, prior=prior, verbose=models_verbose, burnout=5, gt=None, epsilon=epsilon
        )
        if self.verbose:
            print("Done.")
            print("Starting segmentation loop on rest of frames...")

        iterable = frames if models_verbose else tqdm(frames)
        for frame in iterable:
            if models_verbose:
                print(f"Segmenting frame {self.frame_idx}...")
                print("Creating gaussian mixtures from models...")

            self.fgm = extract_gaussian_data_and_create_mixture(fg_model)
            self.bgm = extract_gaussian_data_and_create_mixture(bg_model)

            if models_verbose:
                print("Done.")
                print("Classifying pixels to foreground and background...")

            adjusted_fg, adjusted_bg, new_frame = \
                self._classify_pixels_and_adjust_for_algorithm(frame, include_xy=include_xy, xy_factor=xy_factor)
            new_frames.append(new_frame)

            if models_verbose:
                print("Done.")
                print("Running fit_partial on new foreground and background...")

            fg_model = DPMMPython.fit_partial(model=fg_model, data=adjusted_fg, t=2, iterations=iters_of_fit_partial)
            bg_model = DPMMPython.fit_partial(model=bg_model, data=adjusted_bg, t=2, iterations=iters_of_fit_partial)

            self.frame_idx += 1

            if models_verbose:
                print("Done.")

        if self.verbose:
            print("Segmentation done.")

        build_video_from_frames(new_frames, f"data/videos/{self.name}_new.mp4", show_video=show_video)

