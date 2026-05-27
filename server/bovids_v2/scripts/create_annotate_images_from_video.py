#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Used to sample single images from enclosure recordings to
- either annotate them manually 
- or to annotate them by an object detector 
In classification mode, xml files with bounding boxes / segmentation information
are created to be further processed.

Input is an excel file (.xlsx) which contains the enclosure IDs and the dates
which should be used.
"""

__author__ = "Max Hahn-Klimroth"
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"


### Global configuration
VIDEO_ANCHORPOINT = "G:/"

### Script configuration
#### image creation
SAVEPATH_IMAGEFILES = "F:/Bov2Test/images_from_video_2/"
NUMBER_OF_IMAGES_PER_ENCLOSURE = 50
RANDOM_POSITIONS = (
    False  # if True, it samples random frames per video (might be useful sometimes)
)
FILELIST_CREATION = (
    "F:/GitHub Repositories/BOVIDS2/server/bovids_v2/stuff/bilder_erstellen.xlsx"
)

#### object detection / segmentation
INPUT_IMAGES_TO_DETECT = "F:/Bov2Test/images_from_video/"  # normally the same as SAVEPATH_IMAGEFILES but might be different in some cases, requires enclosure_id/images/ folders in there. I suggest this to be a local storage (!)
# creates enclosure_id/labels_{name of network}/ folder
FILELIST_DETECTION = "F:/GitHub Repositories/BOVIDS2/server/bovids_v2/stuff/bilder_erstellen.xlsx"  # probably the same as "FILELIST_CREATION", in some cases might be different
DEVICE = "cpu"  # 0,1 for GPU if cuda is avaiable, cpu differently (need to check)
CLEAN_BOXES = True  # if True, boxes are post-processed (at most one box per class), if false the actual prediction is given

### Tasks
CREATION_MODE = True
CLASSIFICATION_MODE = False


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


def _predict_image(f, dates, starting_time):
    """
    Returns True if and only if the file f belongs to one of the dates
    """
    yyyy, mm, dd, hh, minute, seconds = (
        f[-19:][0:4],
        f[-19:][4:6],
        f[-19:][6:8],
        f[-19:][9:11],
        f[-19:][11:13],
        f[-19:][13:15],
    )
    timeobj = datetime(
        year=int(yyyy),
        month=int(mm),
        day=int(dd),
        hour=int(hh),
        minute=int(minute),
        second=int(seconds),
    )
    if timeobj.hour < int(starting_time):
        new_date = timeobj - timedelta(minutes=60 * 24, seconds=0)
    else:
        new_date = timeobj
    if new_date.strftime("%Y-%m-%d") in dates:
        return True
    return False


if __name__ == "__main__":

    # make it possible to import own modules
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append("../")

    from lib.availability_checks import get_avaiable_video_dates
    from lib.func import (
        get_videoname,
        write_json_file_boundingbox_as_polygon,
        write_json_file_polygon,
        ensure_directory,
    )
    from lib.func import read_inputfile
    from lib.image_manipulation import (
        IO_write_only_video_frames as IO_write_video_frames,
    )
    from lib.object_detection import predict_folder_raw
    from config.get_config import get_enclosure_information

    # import standard libraries
    import pandas as pd
    import numpy as np
    import random
    from tqdm.contrib.concurrent import thread_map, cpu_count
    from tqdm import tqdm
    from moviepy.editor import VideoFileClip
    from datetime import datetime, timedelta
    from skimage.io import imsave, imshow

    # result = thread_map(functionname, args, max_workers=4)

    if CREATION_MODE:
        to_process = read_inputfile(FILELIST_CREATION)
        df_enc_info = get_enclosure_information(to_process.keys())

        for enclosure_id, dates in to_process.items():

            if not enclosure_id in df_enc_info.index:
                print("ERROR: no configuration available for {}".format(enclosure_id))
                continue
            else:
                print(
                    "{}: Creating images for {}".format(
                        datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), enclosure_id
                    )
                )

            save_path = SAVEPATH_IMAGEFILES + enclosure_id + "/"
            ensure_directory(save_path + "images/")

            # calculate which frames to be taken from each video
            images_per_video = round(NUMBER_OF_IMAGES_PER_ENCLOSURE / len(dates))
            rec_start = datetime(
                year=2000,
                month=1,
                day=2,
                hour=int(df_enc_info.loc[enclosure_id, "recording_start"]),
            )
            rec_end = datetime(
                year=2000,
                month=1,
                day=3,
                hour=int(df_enc_info.loc[enclosure_id, "recording_end"]),
            )
            total_frames = (rec_end - rec_start).seconds

            video_to_process = []

            for date in dates:

                if not RANDOM_POSITIONS:
                    required_frames = [
                        total_frames // images_per_video * (x + 1) - 1
                        for x in range(images_per_video)
                    ]
                else:
                    required_frames = random.sample(
                        [n for n in range(total_frames)], images_per_video
                    )

                yyyy, mm, dd = date.split("-")
                start_date = datetime(
                    year=int(yyyy),
                    month=int(mm),
                    day=int(dd),
                    hour=int(df_enc_info.loc[enclosure_id, "recording_start"]),
                )

                video_list = []
                for j in range(
                    len(df_enc_info.loc[enclosure_id, "video_stream_folder"].split(";"))
                ):
                    video_list.append(
                        get_videoname(
                            date,
                            VIDEO_ANCHORPOINT,
                            j,
                            df_enc_info.loc[enclosure_id, "video_stream_folder"].split(
                                ";"
                            ),
                            df_enc_info.loc[enclosure_id, "video_name"].split(";"),
                        )
                    )

                video_to_process.append(
                    [
                        required_frames,
                        start_date,
                        save_path,
                        enclosure_id,
                        video_list,
                        "images",
                    ]
                )

            # work on video files in parallel threads
            thread_map(IO_write_video_frames, video_to_process, max_workers=cpu_count())

        print(
            "{}: Finished creation of images for all enclosures.".format(
                datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            )
        )

    if CLASSIFICATION_MODE:

        resources_folder = os.path.abspath(os.path.dirname(__file__) + "/../res/")

        to_process = read_inputfile(FILELIST_DETECTION)
        df_enc_info = get_enclosure_information(to_process.keys())

        for enclosure_id, dates in to_process.items():

            if not enclosure_id in df_enc_info.index:
                print("ERROR: no configuration available for {}".format(enclosure_id))
                continue

            TASK = df_enc_info.loc[enclosure_id, "task"]
            if not TASK in ["detect", "segment"]:
                print("ERROR: No task chosen for {}".format(enclosure_id))
                continue
            elif (
                TASK == "detect"
                and not len(df_enc_info.loc[enclosure_id, "object_detector"]) > 0
            ):
                print("ERROR: no object detector given for {}".format(enclosure_id))
                continue
            elif (
                TASK == "segment"
                and not len(df_enc_info.loc[enclosure_id, "image_segmentor"]) > 0
            ):
                print("ERROR: no image segmentor given for {}".format(enclosure_id))
                continue
            elif TASK == "detect" and not (
                os.path.exists(
                    resources_folder
                    + "/objectdetection/"
                    + df_enc_info.loc[enclosure_id, "object_detector"]
                    + ".pt"
                )
            ):
                print("ERROR: object detector does not exist ({})".format(enclosure_id))
                continue
            elif TASK == "segment" and not (
                os.path.exists(
                    resources_folder
                    + "/imagesegmentation/"
                    + df_enc_info.loc[enclosure_id, "image_segmentor"]
                    + ".pt"
                )
            ):
                print("ERROR: image segmentor does not exist ({})".format(enclosure_id))
                continue
            else:
                print(
                    "{}: Starting {} for {}".format(
                        datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), TASK, enclosure_id
                    )
                )

            img_path = INPUT_IMAGES_TO_DETECT + enclosure_id + "/images/"
            if not os.path.exists(img_path):
                print("No images for {}".format(enclosure_id))
                continue

            images_to_predict = [
                img_path + f
                for f in os.listdir(img_path)
                if (
                    f.endswith("jpg")
                    and _predict_image(
                        f, dates, int(df_enc_info.loc[enclosure_id, "recording_start"])
                    )
                )
            ]
            if not len(images_to_predict) > 0:
                print("No images for {} with fitting dates.".format(enclosure_id))
                continue

            predict_folder_raw(
                TASK,
                INPUT_IMAGES_TO_DETECT,
                images_to_predict,
                enclosure_id,
                DEVICE,
                CLEAN_BOXES,
            )
