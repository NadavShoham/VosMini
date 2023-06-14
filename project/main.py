from video_object_segmentation import VideoObjectSegmentation
#TODO: tasks:
# 1. fix bug that results in fgm pdf always winning
# 2. optimize niw parameters like in or dinaris julia example
# 3. optimize as talked with oren
# 4. create latex report
# 5. try other videos
# 6. add a cmd line script
# 7. add a readme


def main(*args, **kwargs):
    # TODO: option to use a cmd line script
    # name = args[0]
    # resize = kwargs["resize"]
    # show_images = kwargs["show_images"]

    name = "wolf"
    resize = True
    show_images = False

    vos = VideoObjectSegmentation(name=name)
    vos.segment(
        frames_num=10,
        iters_of_fit_partial=5,
        resize=resize,
        show_images=show_images,
        verbose=1,
        save_images=True,
        include_xy=True,
        xy_factor=1.0
    )


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


