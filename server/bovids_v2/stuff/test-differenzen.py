# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:35:22 2023

@author: Max
"""

from moviepy.editor import VideoFileClip, ImageSequenceClip
from tqdm import tqdm
import numpy as np
from skimage.io import imsave
from skimage.transform import resize

clip = VideoFileClip("F:/Bov2Test/differences/00_Testvideo.avi")
num_frames = int(clip.duration * clip.fps)
interval_len = 7

interval_list = [
    [i - 1, i, i + 1, i + 2]
    for i in range(num_frames)
    if i % interval_len == 1 and i < num_frames - interval_len - 1
]
image_list = []


def stuff(intervals):
    global image_list
    frames = [clip.get_frame(j) for j in intervals]
    average_image = np.zeros(frames[0].shape)
    for f in frames:
        average_image += f
    average_image /= len(frames)

    differences = np.abs(average_image - frames[1])
    differences = np.minimum(4 * differences, 255 * np.ones(differences.shape)).astype(
        np.uint8
    )

    # res1 = np.concatenate( [ frames[0], frames[2]], axis = 1)
    res = np.concatenate([frames[1], differences], axis=1)
    # res = np.concatenate( [res1, res2], axis = 0 )
    imsave("F:/Bov2Test/differences/" + str(intervals[0]) + ".jpg", res)
    image_list.append("F:/Bov2Test/differences/" + str(intervals[0]) + ".jpg")


for intervals in tqdm(interval_list):
    stuff(intervals)

# new_clip = ImageSequenceClip(image_list, fps=1)
# new_clip.write_videofile('F:/Bov2Test/differences/00_Output.avi', fps=1)
