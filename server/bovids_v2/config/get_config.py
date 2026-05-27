#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stored in config/
Module to read out global information
    - get_enclosure_information( [enc_ids] )
    - the black polygons and the truncation areas from labelMe xml
    - does also contain very global variables as SECONDS_PER_INTERVAL, FRAMES_PER_INTERVAL
"""

import os, json
import pandas as pd
import numpy as np

__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

from local_pc.thesis_tests import timer

# og werte 7 und 3

SECONDS_PER_INTERVAL = 7 # length of a time interval
IMAGES_PER_INTERVAL = 3  # it has to be 3 images per intervall so that the difference images can be calculated
# IMAGES_PER_INTERVAL = 2  # it has to be 3 images per intervall so that the difference images can be calculated. To my (tobi) knowledge, two are enough!

# TODO FPS as parameter - right now, FPS = 1

# NOT USED IN PROGRAM
# ATTENTION: it requires that we actually want to record and observe a night (from something today to something tomorrow) otherwise, it will return nonsense!
# og werte 19 und 6
EVALUATION_START = 22
EVALUATION_END = 1

AC_MODES = [
    "StLy",
    "StFo",
    "LHULHD",
]

BEHAVIORS_BY_MODE = {
    "StLy": {"Standing": 1, "Lying": 2},
    "StFo": {"Standing_no_food": 10, "Food": 11},
    "LHULHD": {"LHU": 20, "LHD": 21},
}

BEHAVIORS_BY_MODE_2 = {
    "StLy": {"Out of View": 0, "Standing": 1, "Lying": 2},
    "StFo": {"Standing_no_food": 10, "Food": 11},
    "LHULHD": {"LHU": 20, "LHD": 21},
}

IMAGE_WIDTH_ACTION_CLASSIFICATION = 384
IMAGE_HEIGHT_ACTION_CLASSIFICATION = 384

BEHAVIOR_STATISTICS = {
    0: "Out of View",
    1: "Standing",
    2: "Lying",
    10: "Standing No Food",
    11: "Standing Food",
    20: "Lying Head Up",
    21: "Lying Head Down",
}

BEHAVIOR_VISUALIZATION_CATEGORY = {
    "Out of View": 3,
    "Standing": 1,
    "Lying": 2,
    "Standing No Food": 0.8,
    "Standing Food": 1.2,
    "Lying Head Up": 2.2,
    "Lying Head Down": 1.8,
}
# Moving: 1.4

BEHAVIOR_COLOR_MAPPING = {
    "Standing": "cornflowerblue",
    "Lying": "forestgreen",
    "Lying Head Up": "forestgreen",
    "Lying Head Down": "lime",
    "Standing No Food": "navy",
    "Standing Food": "cyan",
    "Out of View": "darkgrey",
}
#"Moving": "navy",

BEHAVIOR_VISUALIZATION_NAMES = {
    "Out of View": "Out",
    "Standing": "Standing",
    "Lying": "Lying",
    "Standing No Food": "No Food",
    "Standing Food": "Food",
    "Lying Head Up": "LHU",
    "Lying Head Down": "LHD",
}

BATCH_SIZE_PREDICTION_AC = 4


def use_frames_per_interval(time_interval):
    """
    Given interval x (0-indexed), returns a list of frames that belong to this interval.
    It is supposed that intervals range from 0 to n and frames from 0 to SECONDS_PER_INTERVAL*n (1 fps)
    TODO: make sure that thoses are the only functions that needs adjustment if different fps is used
    """
    return [
        time_interval * SECONDS_PER_INTERVAL + i for i in range(IMAGES_PER_INTERVAL)
    ]


def get_required_timeintervals(enclosure_information, enclosure_id, eval_start, eval_end):
    """
    Given the enclosure, the method figures out which time intervals of a video file (!) are needed during prediction.
    In particular, if the video starts at 17 and observation at 18, then the first needed interval is 514 (if 7 seconds / interval).
    ATTENTION: it requires that we actually want to record and observe a night (from something today to something tomorrow)
    otherwise, it will return nonsense!
    """

    if not enclosure_id in enclosure_information.index:
        return []

    #start_bias = EVALUATION_START - enclosure_information.loc[enclosure_id, "recording_start"]
    #end_bias = enclosure_information.loc[enclosure_id, "recording_end"] - EVALUATION_END
    start_bias = eval_start - enclosure_information.loc[enclosure_id, "recording_start"]
    #end_bias = enclosure_information.loc[enclosure_id, "recording_end"] - eval_end

    if start_bias < 0 :
        start_bias += 24

    intervals_in_start_bias = (start_bias*60*60) // SECONDS_PER_INTERVAL

    if eval_start > eval_end:
        hours_observation = 24 - eval_start + eval_end
    else:
        hours_observation = eval_end - eval_start
    number_intervals_observation = 3600 * hours_observation // SECONDS_PER_INTERVAL

    # for x in range(number_intervals_observation):
    #     print(x)

    return [intervals_in_start_bias + x for x in range(number_intervals_observation)]


def get_boris_behaviormap(individual_id, mode):
    """
    Returns a behavior mapping for BORIS annotation files
    Returns a dictionary {boris_action:behavior_id}
    """

    df = pd.read_excel(
        os.path.dirname(os.path.abspath(__file__)) + "/boris_information.xlsx",
        index_col="individual_id",
        sheet_name="IndividualNames",
    )
    if not individual_id in df.index:
        return {}
    sheet_info = df.loc[individual_id, f"BehaviorMapping_{mode}"]

    df = pd.read_excel(
        os.path.dirname(os.path.abspath(__file__)) + "/boris_information.xlsx",
        index_col="BorisAction",
        sheet_name=sheet_info,
    )
    return {k: int(df.loc[k, "Behavior"]) for k in df.index}


def get_boris_information(individual_id):
    """
    Returns the relative path to the BORIS annotation files of the individual.
    Returns moreover start, end of BORIS annotation and
    Gets information from /config/boris_information.xlsx
    """
    df = pd.read_excel(
        os.path.dirname(os.path.abspath(__file__)) + "/boris_information.xlsx",
        index_col="individual_id",
        sheet_name="IndividualNames",
    )
    ret = {}
    if not individual_id in df.index:
        return ret
    ret["start"] = df.loc[individual_id, "boris_start"]
    ret["end"] = df.loc[individual_id, "boris_end"]
    ret["relative_path"] = df.loc[individual_id, "borisfiles_folder"]
    ret["boris_name"] = df.loc[individual_id, "boris_name"]
    ret["borisfiles_name"] = df.loc[individual_id, "borisfiles_name"]
    return ret


def get_black_polygons(enclosure_id):
    """
    Given the enclosure ID, the method finds all polygon points required for black regions

    Parameters
    ----------
    enclosure_id : enclosure id

    Returns
    -------
    list of np arrays (for creating polygons)

    """

    filepath = os.path.dirname(
        os.path.abspath(__file__)
    ) + "/poly_trunc/{}.json".format(enclosure_id)
    if not os.path.exists(filepath):
        return []

    with open(filepath) as json_file:
        data = json.load(json_file)
        poly_points = []
        for poly in data["shapes"]:
            if poly["label"] != "black_region":
                continue
            poly_points.append(np.array(poly["points"]))
    return poly_points


def get_enclosure_information(enc_ids=[], filtering=True):
    """
    Reads .xlsx file containing all enclosure information

    Parameters
    ----------
    enc_ids : list
        list of enclosure ids.
    filtering: bool
        if false, complete dataframe is returned

    Returns
    -------
    dataframe with enclosure_id as index, only the rows given if filtering = True
    """

    try:
        df = pd.read_excel(
            os.path.dirname(os.path.abspath(__file__)) + "/enclosure_information.xlsx",
            index_col="enclosure_id",
        )
    except:
        print(
            f"FATAL ERROR. Configuration file {os.path.dirname(os.path.abspath(__file__))}/enclosure_information.xlsx not found."
        )
        return None

    if filtering and len(enc_ids) > 0:
        df = df[df.index.isin(enc_ids)]
    return df


def get_postprocessing_rules(pp_rulename, ac_mode):
    """
    Returns a dataframe object with rows 'comment', 'previous', 'current', 'next', 'current_corrected', 'min_length'.
    comment is a description like Standing-Lying-Standing
    previous, current, next, current_corrected are behavioral codes
    min_length is a float in minutes
    We get a specific minimum length and the actual new behavior by
        i = df[ (df['previous'] == x) & (df['current'] == y) & (df['next']==z)  ].index[0]
        min_length, behavior_new = df.loc[i, 'min_length'], df.loc[i, 'current_corrected']
    """
    try:
        df = pd.read_excel(
            os.path.dirname(os.path.abspath(__file__)) + "/post_processing_rules.xlsx",
            sheet_name=ac_mode,
        )
    except:
        print(f"WARNING. No postprocessing rules available for mode {ac_mode}.")
        return None

    if not pp_rulename in df.keys():
        print(f"WARNING. {pp_rulename} not available for mode {ac_mode}.")
        return None

    df2 = df[
        ["comment", "previous", "current", "next", "current_corrected", pp_rulename]
    ]
    df2.columns = [
        "comment",
        "previous",
        "current",
        "next",
        "current_corrected",
        "min_length",
    ]
    try:
        df2 = df2.astype(
            {
                "previous": "int",
                "current": "int",
                "next": "int",
                "current_corrected": "int",
                "min_length": "float",
            }
        )
    except:
        print(f"WARNING. No complete ruleset. Missing entries.")
        return None

    return df2


def get_individual_information():
    """
    Reads .xlsx file containing all individual information

    Returns
    -------
    dataframe with individual_id as index.
    """

    try:
        df = pd.read_excel(
            os.path.dirname(os.path.abspath(__file__)) + "/individual_information.xlsx",
            index_col="individual_id",
        )
    except:
        print(
            f"FATAL ERROR. Configuration file {os.path.dirname(os.path.abspath(__file__))}/individual_information.xlsx not found."
        )
        return None

    return df
