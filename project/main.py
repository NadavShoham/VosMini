from video_object_segmentation import VideoObjectSegmentation
#TODO: tasks:
# 1. fix bug that results in fgm pdf always winning
# 2. optimize niw parameters like in or dinaris julia example
# 3. optimize as talked with oren
# 4. create latex report
# 5. try other videos
# 6. add a cmd line script
# 7. add a readme
# 8. add a requirements.txt?
# 9. clean up code and directory structure


def main(*args, **kwargs):
    # TODO: option to use a cmd line script
    # name = args[0]
    # resize = kwargs["resize"]
    # show_images = kwargs["show_images"]

    name = "dog"

    vos = VideoObjectSegmentation(
        name=name,
        include_xy=True,
        resize_ratio=0.5,
        complement_with_white=False
    )
    vos.segment(
        frames_num=20,
        verbose=1,
        use_max_pdf=False,
        alpha=100.0,
        epsilon=0.0000001,
        use_niw_prior=True,
        run_fit_partial=False,
        iters_of_fit_partial=5,
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


