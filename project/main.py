import numpy as np
import sys
from grabcut import GrabCut
from typing import List
from PIL import Image
from utils import gif_to_stream, stream_to_np_array, transpose_np_array, resize_stream
from julia.api import Julia
from dpmmpythonStreaming.dpmmwrapper import DPMMPython
from dpmmpythonStreaming.priors import niw
from sklearn.mixture import GaussianMixture
jl = Julia(compiled_modules=False)


#TODO: tasks:
# 1. create foreground and background extraction function
# 2. create a function that extracts gaussian data from the algorithm returned object and creates a mixture
# 3. create the main method that goes as follows:
#   a. create a stream of images as np arrays of pixels (RGBXY)
#   b. extract foreground and background of the first image
#   c. run 2 models with 'fit.init' one for foreground and one for background
#   d. create a mixture of gaussians for each model
#   e. for each image in the rest of the stream:
#       i. determine for each pixel if it belongs to the foreground or background using the mixture of gaussians
#       ii. call 'fit.partial' for each model with the pixels that belong to it
#       iii. create a new mixture of gaussians for each model
# optional: optimize niw parameters like in or dinaris julia example

def extract_foreground_background(image: np.ndarray) -> (np.ndarray, np.ndarray):
    grab_cut = GrabCut(image)
    grab_cut.run()
    foreground = grab_cut.foreground
    background = grab_cut.background
    return foreground, background


def extract_gaussian_data_and_create_mixture(model):
    clusters = model.group.local_clusters
    means = []
    covs = []
    for cluster in clusters:
        means.append(cluster.cluster_params.cluster_params.distribution.μ)
        covs.append(cluster.cluster_params.cluster_params.distribution.Σ)
    gmm = GaussianMixture(n_components=len(clusters), covariance_type='full')
    gmm.means_init = np.array(means)
    gmm.covariances_init = np.array(covs)
    gmm.fit(np.array(model.data))
    return gmm



def main(*args, **kwargs):
    # TODO: option to use a cmd line script
    # gif_path = args[0]
    # resize = kwargs["resize"]
    gif_path = "./gifs/border_collie_running.gif"
    resize = False

    # TODO: separate code to helper functions
    frames: List[Image.Image] = gif_to_stream(gif_path)
    if resize:
        frames = resize_stream(frames, (200, 150))
    print(f"Loaded {len(frames)} frames from {gif_path} with size {frames[0].size}")
    first_frame_foreground, first_frame_background = extract_foreground_background(np.array(frames[0]))

    # # TODO: clean this part of code, functions are confusing, need to also support videos, dont need to insert to frames and extract again...
    # frames.insert(0, Image.fromarray(first_frame_foreground))
    # frames.insert(0, Image.fromarray(first_frame_background))
    # data = stream_to_np_array(frames, dtype=np.float32)
    # # print(f"Data shape: {data.shape}")
    # data = transpose_np_array(data)
    # shape = data.shape
    # # print(f"Data shape after transpose of each frame: {shape}")
    #
    # # Initialize the models
    # # TODO: optimize niw parameters like in or dinaris julia example
    # prior = niw(1, np.zeros(shape[1]), 3, np.eye(shape[1]) * 0.5)
    # alpha = 10.0
    # head_frames = data[:2]
    # tail_frames = data[2:]
    #
    # background_model = DPMMPython.fit_init(
    #     data=head_frames[0], alpha=alpha, prior=prior, verbose=True, burnout=5, gt=None, epsilon=0.0000001
    # )
    # foreground_model = DPMMPython.fit_init(
    #     data=head_frames[1], alpha=alpha, prior=prior, verbose=True, burnout=5, gt=None, epsilon=0.0000001
    # )



if __name__ == '__main__':
    # TODO: option to use a cmd line script
    # args = sys.argv[1:]
    # kwargs = {}
    # for arg in args:
    #     if arg.startswith("--"):
    #         key, value = arg.split("=")
    #         kwargs[key[2:]] = value
    # main(*args, **kwargs)
    main()
