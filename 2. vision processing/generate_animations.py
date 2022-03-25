from pathlib import Path
from argparse import ArgumentParser

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
from torch.nn.functional import avg_pool2d

from body_postures import BodyPostureEvents, BodyPostureFrames


ap = ArgumentParser()

ap.add_argument("-f", "--frames", dest="frames", action="store_true")
ap.add_argument("-r", "--raster", dest="raster", action="store_true")
ap.add_argument("--hot", dest="hot_pixel_filter_freq", default=60, type=int)
ap.add_argument("-c", "--count", dest="event_count", default=3000, type=int)
ap.add_argument("-s", "--slice", dest="slice_dt", default=2e6, type=float)
ap.add_argument("-b", "--bin", dest="bin_dt", default=2e3, type=float)
ap.add_argument("-p", "--pooling", dest="pooling", default=1, type=int)
ap.add_argument("-d", "--downsample", dest="downsample", default=1, type=int)
ap.add_argument("--output_format", default="mpeg", type=str)
ap.add_argument("--fps", default=20, type=int)
ap.add_argument("--num_frames", default=100, type=int)

args = ap.parse_args()

data_path = Path("body-postures-dataset") / "data" / "test"

if args.output_format == "mpeg":
    writer = "ffmpeg"
    ending = "mp4"
elif args.output_format == "gif":
    writer = "imagemagick"
    ending = "gif"
else:
    raise ValueError(f"Output format `{args.output_format}` not recognized.")


def get_first_rec_of_label(label, dataset):
    current_idx = 0

    # Skip samples that are different from label
    while current_idx < len(dataset) and dataset[current_idx][1] != label:
        current_idx += 1

    while current_idx < len(dataset) and dataset[current_idx][1] == label:
        yield dataset[current_idx][0]
        current_idx += 1


def pool(data, kernel_size):
    original_shape = data.shape
    data = torch.as_tensor(data)
    if data.ndim == 3:
        data.unsqueeze_(0)

    original_dtype = data.dtype
    data = data.float() * kernel_size ** 2
    pooled = avg_pool2d(data, kernel_size).type(original_dtype)

    if len(original_shape) == 3:
        pooled.squeeze_(0)

    return pooled.numpy()


if args.frames:

    # Validation dataset
    frame_dataset = BodyPostureFrames(
        data_path=data_path,
        event_count=args.event_count,
        hot_pixel_filter_freq=args.hot_pixel_filter_freq,
        metadata_path=f'metadata/frames/test/{args.event_count}events_{args.hot_pixel_filter_freq}filter',
    )

    lbl2clss = {lbl: clss for clss, lbl in frame_dataset.classes.items()}
    data = {
        lbl: list(get_first_rec_of_label(lbl, frame_dataset))
        for lbl in frame_dataset.classes.values()
    }

    fig = plt.figure(figsize=(12, 6))
    axes = [fig.add_subplot(2, 4, lbl + 1) for lbl in range(len(frame_dataset.classes))]
    screens = [
        ax.imshow(pool(data[lbl][0], args.pooling)[0])
        for lbl, ax in enumerate(axes)
    ]
    for lbl, ax in enumerate(axes):
        ax.set_title(f"Label: {lbl} - `{lbl2clss[lbl]}`")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )
    plt.tight_layout()

    def draw(idx):
        print(idx, end="\r")
        for lbl, scrn in enumerate(screens):
            scrn.set_data(
                pool(data[lbl][(args.downsample * idx) % len(data[lbl])], args.pooling)[0]
            )
        return screens

    anim = FuncAnimation(fig, draw, frames=args.num_frames, blit=True, interval=1)
    anim.save("frames." + ending, writer=writer, fps=args.fps)


if args.raster:

    raster_dataset = BodyPostureEvents(
        data_path=data_path,
        cache_path=f"cache/test/{args.slice_dt}/{args.bin_dt}",
        slice_dt=args.slice_dt,
        bin_dt=args.bin_dt,
        metadata_path=f"metadata/raster/test/{args.slice_dt}/{args.bin_dt}",
        hot_pixel_filter_freq=args.hot_pixel_filter_freq,
    )
    lbl2clss = {lbl: clss for clss, lbl in raster_dataset.classes.items()}
    data = {
        lbl: list(get_first_rec_of_label(lbl, raster_dataset))
        for lbl in raster_dataset.classes.values()
    }

    for lbl, smpls in data.items():
        # data[lbl] = pool(np.vstack(smpls)[:, 0], args.pooling)
        data[lbl] = np.vstack(smpls)[:, 0]

    for lbl, smpls in data.items():
        data[lbl] = pool(smpls, args.pooling)

    fig = plt.figure(figsize=(12, 6))
    axes = [fig.add_subplot(2, 4, lbl + 1) for lbl in range(len(raster_dataset.classes))]
    # screens = [ax.imshow(data[lbl][0]) for lbl, ax in enumerate(axes)]

    scatters = [
        ax.scatter(*(np.where(data[lbl][0])[::-1]), s=1, color="k")
        for lbl, ax in enumerate(axes)
    ]

    for lbl, ax in enumerate(axes):
        ax.set_title(f"Label: {lbl} - `{lbl2clss[lbl]}`")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )
        ax.invert_yaxis()
    plt.tight_layout()

    def draw(idx):
        print(idx, end="\r")
        for lbl, sctr in enumerate(scatters):
            # Make sure index does not exceed individual max
            index = (args.downsample * idx) % len(data[lbl])
            offsets = np.where(data[lbl][index])[::-1]
            sctr.set_offsets(np.c_[offsets])
        return scatters

    num_frames = min(len(smpl) for smpl in data.values())
    anim = FuncAnimation(fig, draw, frames=args.num_frames, blit=True, interval=1)
    anim.save("raster." + ending, writer=writer, fps=args.fps)

    # fig, ax = plt.subplots()

    # screen = ax.imshow(data[0][0])

    # def draw(idx):
    #     print(idx, end="\r")
    #     screen.set_data(data[0][idx * 10])
    #     return screen,

    # anim = FuncAnimation(fig, draw, frames=100, blit=True, interval=1)
    # # plt.show()
    # anim.save('raster.gif', writer='imagemagick', fps=50)

    fig = plt.figure(figsize=(12, 6))
    axes = [fig.add_subplot(2, 4, lbl + 1) for lbl in range(len(frame_dataset.classes))]
    # screens = [
    #     ax.imshow(pool(data[lbl][0], args.pooling)[0], cmap=cm.Greys, vmin=0, vmax=20)
    #     for lbl, ax in enumerate(axes)
    # ]
    for lbl, ax in enumerate(axes):
        ax.set_title(f"Label: {lbl} - `{lbl2clss[lbl]}`")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )
    plt.tight_layout()

    def draw(idx):
        print(idx, end="\r")
        # for lbl, scrn in enumerate(screens):
        #     scrn.set_data(
        #         pool(data[lbl][(args.downsample * idx) % len(data[lbl])], args.pooling)[0]
        #     )
        for lbl, scrn in enumerate(scatters):
            scrn.set_data(
                *np.where(pool(data[lbl][(args.downsample * idx) % len(data[lbl])], args.pooling)[0])
            )
        return screens
