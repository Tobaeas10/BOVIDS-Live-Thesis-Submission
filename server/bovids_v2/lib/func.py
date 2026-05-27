#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains small help functions that are used throughout BOVIDS
"""

__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

# make it possible to import own modules
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
from server.bovids_v2.config.get_config import (
    get_boris_information,
    SECONDS_PER_INTERVAL,
    get_boris_behaviormap,
)
from server.bovids_v2.lib.availability_checks import _ac_cache_avaiable
import numpy as np
import pandas as pd
import yaml, json
from collections import Counter
import shutil
from typing import List


def zip_od_images(args):
    """
    args list such that args = [path to folder, output zip file, ...]
    """
    return shutil.make_archive(args[1], "zip", args[0])


def zip_pp(args):
    """
    args list such that args = [path to folder, output zip file, ...]
    """
    return shutil.make_archive(args[1], "zip", args[0])


def get_video_files(enclosure_id, base_vid, enc_conf, date):
    """

    Parameters
    ----------
    enclosure_id : String
        Enclosure_id as defined in the configuration files..
    base_vid : TYPE, optional
        Anchorpoint for the videos.
    enc_conf : TYPE, optional
        enclosure configuration file.

    Returns
    -------
    List
        List of video paths.

    """

    df = enc_conf

    vid_stream_folders = df.loc[enclosure_id, "video_stream_folder"].split(";")
    vid_stream_folders = [base_vid + x for x in vid_stream_folders]
    vid_name_suffix = df.loc[enclosure_id, "video_name"].split(";")

    if len(vid_stream_folders) != len(vid_name_suffix):
        print("ERROR in configuration {}. Mismatch in the number of videofiles.")
        return []

    datelists = []  # datelists als Liste von Videopfaden
    for j in range(len(vid_stream_folders)):
        for f in sorted(os.listdir(vid_stream_folders[j])):
            if not f.endswith(vid_name_suffix[j]):
                continue
            if not date == f.split("_")[0]:
                continue
            datelists.append(vid_stream_folders[j] + "/" + f)
    print(datelists)
    return datelists


def get_boris_annotation(anchorpath, individual_id, date, mode, start_time, end_time):
    """
    For the date and the individual, it returns a sequence [i] where i is a behavior code
    of the length given in the BORIS configuration.
    If no boris annotation is present, returns [].

    Behavior mapping: e.g. { 'Running': 11 } where Running is the BORIS entry
    start_time and end_time: desired start and end of sequence
    """

    def _get_boris_sequence(df, individual_name, behavior_mapping, new_format):
        """returns [ [time, behavior, start/stop] ]"""
        ret = []
        unknown_classes = set()
        for j in df.index:

            if not df.loc[j, "Subject"] == individual_name:
                continue

            if not df.loc[j, "Behavior"] in behavior_mapping.keys():
                unknown_classes.add(df.loc[j, "Behavior"])
                continue

            curr_time = int(float(df.loc[j, "Time"]))

            curr_behav = behavior_mapping[df.loc[j, "Behavior"]]
            ret.append([curr_time, curr_behav, df.loc[j, "Status"]])

        if len(unknown_classes) > 0:
            print(
                f"WARNING: The following classes/behaviors are not known {unknown_classes}"
            )

        return ret

    def _clean_list(boris_seq):
        """Deletes [x, 0, start], [x, 0, stop], [x, 0, start], [x, 0, stop]"""
        new_seq = []
        j = 0
        while j < (len(boris_seq) - 3):
            new_seq.append(boris_seq[j])
            if not (
                boris_seq[j][1] == boris_seq[j + 1][1]
                and boris_seq[j][1] == boris_seq[j + 2][1]
                and boris_seq[j][1] == boris_seq[j + 3][1]
            ):
                j += 1
            else:
                j += 3
        # append the last three entries.
        while j <= len(boris_seq) - 1:
            new_seq.append(boris_seq[j])
            j += 1

        return new_seq

    def _check_sanity(boris_seq):
        j = 0
        while j < (len(boris_seq) - 1):
            if not (
                boris_seq[j][1] == boris_seq[j + 1][1]
                and boris_seq[j][2] == "START"
                and boris_seq[j + 1][2] == "STOP"
            ):
                return False
            j += 2
        return True

    def _second_exact_sequences(boris_sequence, skip_seconds_start, skip_seconds_end):
        """
        Returns a sequence (list) of behavior codes, one per second
        """
        boris_sequence = _clean_list(boris_sequence)
        if not _check_sanity(boris_sequence):
            return []

        ret_seq = []

        curr_behav = boris_sequence[0][1]
        curr_time = 0

        for j in range(1, len(boris_sequence)):
            new_time = boris_sequence[j][0]
            time_diff = new_time - curr_time

            append_list = [curr_behav] * time_diff
            ret_seq.extend(append_list)

            curr_behav = boris_sequence[j][1]
            curr_time = boris_sequence[j][0]

        if skip_seconds_end > 0:
            ret_seq = ret_seq[skip_seconds_start : (-1) * skip_seconds_end]

        if skip_seconds_start > 0:
            ret_seq = ret_seq[skip_seconds_start:]

        return ret_seq

    def _map_sequence_to_intervals(seq):
        """
        Given a sequence of behaviors in seconds, this method returns a list such that SECONDS_PER_INTERVAL
        many consecutive numbers are mapped to one number by a majority vote.
        """
        tmp_seq = [
            seq[i : i + SECONDS_PER_INTERVAL]
            for i in range(0, len(seq), SECONDS_PER_INTERVAL)
        ]
        ret = [Counter(x).most_common()[0][0] for x in tmp_seq]

        return ret

    boris_info = get_boris_information(individual_id)
    if len(boris_info.keys()) == 0:
        print(f"ERROR. No entry in the boris_information.xlsx for {individual_id}.")
        return []
    boris_file_xlsx = f'{anchorpath}{boris_info["relative_path"]}/{date}-_{boris_info["borisfiles_name"]}.xlsx'
    if not os.path.exists(boris_file_xlsx):
        print(f"ERROR. No boris annotation found for {date} of {individual_id}")
        print("Missing file:", boris_file_xlsx)
        return []

    behavior_mapping = get_boris_behaviormap(individual_id, mode)
    # do the actual translation of the boris document to a sequence

    # get first reasonable row
    df = pd.read_excel(boris_file_xlsx)
    header_row = 0
    for j in df.index:
        if df.loc[j, "Observation id"] in ["Time", "time"]:
            header_row = j
            break

    # read dataframe
    df = pd.read_excel(
        boris_file_xlsx, header=0 if header_row == 0 else header_row + 1
    )  # header_row + 1 in old format?
    df = df.rename(columns={"Behavior type": "Status"})
    individual_name = boris_info["boris_name"]
    boris_sequence = _get_boris_sequence(
        df, individual_name, behavior_mapping, new_format=header_row == 0
    )
    boris_sequence = _clean_list(boris_sequence)

    if not _check_sanity(boris_sequence):
        print("ERROR: There is some error in the BORIS sequence.", date, individual_id)
        return []

    skip_seconds_start = (start_time - boris_info["start"]) * 3600
    skip_seconds_end = (boris_info["end"] - end_time) * 3600

    if skip_seconds_start < 0 or skip_seconds_end < 0:
        print(
            "ERROR: Required starting time or required ending time are invalid.",
            date,
            individual_id,
        )
        return []

    seconds_sequence = _second_exact_sequences(
        boris_sequence, skip_seconds_start, skip_seconds_end
    )
    interval_sequence = _map_sequence_to_intervals(seconds_sequence)

    return interval_sequence


def read_inputfile_datasetcreation(f):
    """
    Returns a dictionary {enclosure_id: {dates: list of dates to process, desired_starts: list, desired_ends: list} }
    The inputfile needs, at least, "date" and "enclosure_id" as well as "desired_start", "desired_end" columns.
    """

    try:
        df = pd.read_excel(f)
        df["date_string"] = df["date"].dt.strftime("%Y-%m-%d")
    except:
        print("ERROR. Either {} does not exist, or one date is invalid".format(f))
        return {}

    enc_ids = list(set(df["enclosure_id"].values))
    ret = {
        enc_id: {
            "dates": [],
            "desired_starts": [],
            "desired_ends": [],
            "dismissed_individuals": [],
        }
        for enc_id in enc_ids
    }

    try:
        for row in df.index:
            ret[df.loc[row, "enclosure_id"]]["dates"].append(df.loc[row, "date_string"])
            ret[df.loc[row, "enclosure_id"]]["desired_starts"].append(
                int(df.loc[row, "desired_start"])
            )
            ret[df.loc[row, "enclosure_id"]]["desired_ends"].append(
                int(df.loc[row, "desired_end"])
            )
            ret[df.loc[row, "enclosure_id"]]["dismissed_individuals"].append(
                [ind for ind in str(df.loc[row, "dismiss_individuals"]).split(";")]
            )
    except:
        print("ERROR. Invalid file {}.".format(f))
        return {}

    return ret


def read_inputfile(f):
    """
    Returns a dictionary {enclosure_id: list of dates to process}
    The inputfile needs, at least, "date" and "enclosure_id" columns.
    """

    try:
        df = pd.read_excel(f)
        df["date_string"] = df["date"].dt.strftime("%Y-%m-%d")
    except:
        print("ERROR. Either {} does not exist, or one date is invalid".format(f))
        return {}

    enc_ids = list(set(df["enclosure_id"].values))
    ret = {enc_id: [] for enc_id in enc_ids}

    for row in df.index:
        ret[df.loc[row, "enclosure_id"]].append(df.loc[row, "date_string"])

    return ret


def get_videoname(date, anchor, index, stream_folders, video_names):
    """


    Parameters
    ----------
    date : YYYY-MM-DD.
    index : integer
    stream_folders : string as in configuration
    video_names : string as in configuration

    Returns
    -------
    video full path

    """
    # slash hinzugefügt
    return anchor + stream_folders[index] + "/" + date + "_" + video_names[index]


def ensure_directory(p):
    """
    Creates directory the safe way.
    """

    dirpath = os.path.dirname(p)
    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    except Exception as e:
        print("WARNING:", e)
    return p


def write_yaml_file(path, train, val, clsnames, output_file):
    """
    Writes a YOLO yaml file.
    """
    ensure_directory(output_file)
    yaml_dict = {
        "path": path,
        "train": train,
        "val": val,
        "nc": len(clsnames.keys()),
        "names": {v: k for k, v in clsnames.items()},
    }
    with open(output_file, "w+") as file:
        yaml.dump(yaml_dict, file, sort_keys=False)


def write_json_file_polygon(
    output_path, polygons, rel_image_path, image_res=(640, 640)
):
    """
    Requires outputpath, polygons as [ ('classname',  np.array( [x,y] ) ],
    relative path from json to image (normally ..\\images\\imagename.jpg or ../images/imagename.jpg)
    """

    output = {
        "imagePath": rel_image_path,
        "imageData": None,
        "imageHeight": image_res[1],
        "imageWidth": image_res[0],
        "flags": {},
        "version": "5.2.0.post4",
        "shapes": [],
    }

    for classname, polygon in polygons:
        tmp = {
            "group_id": None,
            "shape_type": "polygon",
            "label": classname,
            "flags": {},
            "description": {},
            "points": [],
        }
        for x, y in list(polygon):
            tmp["points"].append([int(x), int(y)])

        output["shapes"].append(tmp)

    json_object = json.dumps(output, indent=4)
    ensure_directory(output_path)
    with open(output_path, "w+") as f:
        f.write(json_object)


def write_json_file_boundingbox_as_polygon(
    output_path, bbs, rel_image_path, image_res=(640, 640)
):
    """
    Requires outputpath, bounding boxes as [ ('classname' , [xcenter, ycenter, width, height] ) ],
    relative path from json to image (normally ..\\images\\imagename.jpg or ../images/imagename.jpg)
    """
    output = {
        "imagePath": rel_image_path,
        "imageData": None,
        "imageHeight": image_res[1],
        "imageWidth": image_res[0],
        "flags": {},
        "version": "5.2.0.post4",
        "shapes": [],
    }

    for classname, bb in bbs:
        tmp = {
            "group_id": None,
            "shape_type": "polygon",
            "label": classname,
            "flags": {},
            "description": {},
            "points": [],
        }
        x_center, y_center, width, height = bb
        p0 = [int((x_center - width / 2)), int((y_center - height / 2))]
        p1 = [int((x_center - width / 2)), int((y_center + height / 2))]
        p2 = [int((x_center + width / 2)), int((y_center + height / 2))]
        p3 = [int((x_center + width / 2)), int((y_center - height / 2))]
        tmp["points"].append(p0)
        tmp["points"].append(p1)
        tmp["points"].append(p2)
        tmp["points"].append(p3)

        output["shapes"].append(tmp)

    json_object = json.dumps(output, indent=4)
    ensure_directory(output_path)
    with open(output_path, "w+") as f:
        f.write(json_object)


def polygon_to_bounding_box(polygon):
    """
    Given a polygon ([ [x1 y1], [x2, y2], ... ]) returns the smalles rectangle
    containing all points as x_center y_center width height
    """
    x_min = 9999999
    y_min = 9999999
    x_max = 0
    y_max = 0
    for x, y in list(polygon):
        y_min = min(y_min, y)
        x_min = min(x_min, x)
        y_max = max(y_max, y)
        x_max = max(x_max, x)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def get_polygons_from_json(filepath):
    """
    Returns a list of lists [ ['classname', np.array(polygon points) ] ]
    """

    if not os.path.exists(filepath):
        return []

    with open(filepath) as json_file:
        data = json.load(json_file)
        poly_points = []
        for poly in data["shapes"]:
            if poly["label"] == "black_region":
                continue
            poly_points.append([poly["label"], np.array(poly["points"])])
    return poly_points


def get_images_ac_mode(
        prediction_mode, base_directory_ac_prediction, date, individual, od_image_directory
) -> List[str]:
    if prediction_mode == "StLy":

        ret = [
            img_name
            for img_name in sorted(os.listdir(od_image_directory))
            if img_name.endswith(".jpg")
        ]

    else:

        df_stly_path = f"{base_directory_ac_prediction}{date}_{individual}_StLy.csv"
        if not os.path.exists(df_stly_path):
            print(f"ERROR: {df_stly_path} required but not existent.")
            return []

        df = pd.read_csv(df_stly_path)
        if prediction_mode == "StFo":
            ret = [
                img_name
                for img_name in list(df[df["Standing"] >= df["Lying"]]["img_name"].values)
            ]
        elif prediction_mode == "LHULHD":
            ret = [
                img_name
                for img_name in list(df[df["Standing"] < df["Lying"]]["img_name"].values)
            ]

    return ret

def _ac_operations_conducted(
    remaining_individuals, date, ac_mode, savepath_ac, df, row
):
    """Returns a list of individualcodes for which the ac_mode prediction needs to be conducted at the night specified by df and row"""
    ret = []
    for individual_id in remaining_individuals:
        # the following ac predict modes are set to True if
        # 1. the user wants to do the step and 2. either user does not want to use cached files or there are no cached files
        if int(df.loc[row, f"{ac_mode}"]) == 0:
            continue
        if int(df.loc[row, f"use_cached_{ac_mode}"]) == 0:
            ret.append(individual_id)
            continue
        if not _ac_cache_avaiable(individual_id, date, ac_mode, savepath_ac):
            ret.append(individual_id)

    return ret

def _copy_file(l):
    """
    copies a file from source src to destination dst,
    src = list of videopaths in l[0],
    dst = list of videopaths in l[1]
    """
    if len(l) != 2:
        return
    ensure_directory(l[1])
    shutil.copy(src=l[0], dst=l[1])