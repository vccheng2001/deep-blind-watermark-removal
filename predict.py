import os
import random
import shutil
import sys

import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image, ImageChops

import scripts.datasets as datasets
import scripts.models as models
from scripts.utils.imutils import im_to_numpy

from os import listdir
from os.path import isfile, join
filenames = [
    shutil.copy(join("./natural", f), join("./natural", f).split("-")[0] + ".jpg")
    for f in listdir("./natural")
    if isfile(join("./natural", f))
]


def get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)

    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[2] * 255.0))

    return colormap_int


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


def gray2color(gray_array, color_map):

    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            #             log(256,2) = 8 , log(1,2) = 0 * 8
            color_array[i, j] = color_map[
                clamp(int(abs(gray_array[i, j]) * 10), 0, 255)
            ]

    return color_array


class objectview(object):
    def __init__(self, *args, **kwargs):
        d = dict(*args, **kwargs)
        self.__dict__ = d


jet_map = get_jet()

resume_path = "27kpng_model_best.pth.tar"  # path of pretrained model
samples = [320, 1364, 1868]  # random.sample(range(4000), 1) # show random sample

data_config = objectview(
    {
        "input_size": 256,
        "limited_dataset": 0,
        "normalized_input": False,
        "data_augumentation": False,
        "base_dir": ".",
        "data": "_images",
    }
)

val_loader = torch.utils.data.DataLoader(
    datasets.COCO("val", config=data_config, sample=samples)
)

print("input          | target              | coarser            | final")
print("----------------------------------------------------------------------------")
print("predicted mask | predicted watermark | coarser difference | final difference")

with torch.no_grad():

    model = models.__dict__["vvv4n"]().cuda()
    model.load_state_dict(torch.load(resume_path)["state_dict"])
    model.eval()

    for i, batches in enumerate(val_loader):

        plt.figure(figsize=(48, 12))

        im, mask, target = (
            batches["image"].cuda(),
            batches["mask"].cuda(),
            batches["target"].cuda(),
        )

        imoutput, immask, imwatermark = model(im)

        imcoarser, imrefine, imwatermark = (
            imoutput[1] * immask + im * (1 - immask),
            imoutput[0] * immask + im * (1 - immask),
            imwatermark * immask,
        )

        ims1 = im_to_numpy(
            torch.clamp(
                torch.cat([im, target, imcoarser, imrefine], dim=3)[0] * 255,
                min=0.0,
                max=255.0,
            )
        ).astype(np.uint8)

        imcoarser, imrefine, target = (
            im_to_numpy((imcoarser[0] * 255)).astype(np.uint8),
            im_to_numpy((imrefine[0] * 255)).astype(np.uint8),
            im_to_numpy((target[0] * 255)).astype(np.uint8),
        )
        immask, imwatermark = im_to_numpy((immask.repeat(1, 3, 1, 1)[0] * 255)).astype(
            np.uint8
        ), im_to_numpy((imwatermark[0] * 255)).astype(np.uint8)

        coarsenp = gray2color(
            np.array(
                ImageChops.difference(
                    Image.fromarray(imcoarser), Image.fromarray(target)
                ).convert("L")
            ),
            jet_map,
        )
        finenp = gray2color(
            np.array(
                ImageChops.difference(
                    Image.fromarray(imrefine), Image.fromarray(target)
                ).convert("L")
            ),
            jet_map,
        )

        imfinal = np.concatenate(
            [ims1, np.concatenate([immask, imwatermark, coarsenp, finenp], axis=1)],
            axis=0,
        )

        plt.imshow(imfinal, vmin=0.0, vmax=255.0)
        plt.imsave(f"{i}.jpg", imfinal, vmin=0.0, vmax=255.0)
        exit(-1)