from PIL import Image
from typing import List
import numpy as np


def gif_to_stream(gif_path: str) -> List[Image.Image]:
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.copy()
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
            frames.append(frame)
            gif.seek(len(frames))  # Skip to next frame
    except EOFError:
        pass
    return frames


def stream_to_gif(images: List[Image.Image], gif_path: str):
    images[0].save(gif_path,
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0)


def resize_stream(images: List[Image.Image], size: tuple) -> List[Image.Image]:
    resized_images = []
    for image in images:
        resized_image = image.copy()
        resized_image = resized_image.resize(size, Image.ANTIALIAS)
        resized_images.append(resized_image)
    return resized_images


def stream_to_np_array(images: List[Image.Image], dtype) -> np.ndarray:
    flat_arrays = []
    for image in images:
        array = np.array(image, dtype=dtype)
        # flatten the array
        flat_array = array.reshape((array.shape[0] * array.shape[1], array.shape[2]))
        flat_arrays.append(flat_array)

    return np.array(flat_arrays)


def np_array_to_stream(np_array: np.ndarray, size: tuple, dtype) -> List[Image.Image]:
    images = []
    for array in np_array:
        array = array.reshape((size[0], size[1], 3))
        image = Image.fromarray(array.astype(dtype))
        images.append(image)
    return images


def transpose_np_array(np_array: np.ndarray) -> np.ndarray:
    transposed_np_array = np_array.transpose((0, 2, 1))
    return transposed_np_array



if __name__ == '__main__':
    # test gif to images
    my_gif = "./gifs/border_collie_running.gif"
    images = gif_to_stream(my_gif)

    # show images
    # for image in images:
    #     image.show()



