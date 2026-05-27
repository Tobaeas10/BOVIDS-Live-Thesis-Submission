# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains methods to 
    - evaluate action classification by a neural network

Methods:
    - image_evaluation()

"""

__author__ = ["Lea Möller", "Judith Ballmann"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

# import standard libraries
import pandas as pd
import cv2
import numpy as np
import shutil
from skimage.io import imread


# makes it possible to import own modules
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
from lib.func import ensure_directory
import random
import shutil as sh
from config.get_config import BEHAVIORS_BY_MODE

"""
# input and output path
INPUT_BASE = r'' # folder with project
OUTPUT_BASE = r'' # folder to save dataset
"""

INPUT_BASE = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/action_classification_save/"  # folder with project
IMAGES_BASE = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/objection_detection/"
OUTPUT_BASE = r"C:/Users/Judith/OneDrive - Johann Wolfgang Goethe Universität/BOVIDS/ac_dataset_create/"  # folder to save dataset


# values for hard examples
CRITICAL_VALUE = 0.85
PERCENTAGE_HARD_IMAGES = 0.7

# color codes for displaying boxes {action : color_code}
COLORCODES_EVALUATION = {
    "none": (100, 100, 100),
    "standing": (255, 0, 0),
    "standing_food": (0, 205, 102),
    "lying": (0, 0, 255),
    "lying_head_up": (0, 255, 255),
    "lying_head_down": (0, 127, 225),
    "missing": (180, 0, 180),
}

# script control keys
MOVE_LEFT = "4"
MOVE_RIGHT = "5"
END_EVALUATION = "p"
MOVE_NEXT_UNLABELLED = "9"
MOVE_PREV_UNLABELLED = "8"

SET_STANDING = "a"
SET_STANDING_FOOD = "s"
SET_LYING = "l"
SET_LYING_HEAD_UP = "k"
SET_LYING_HEAD_DOWN = "j"
SET_UNLABELLED = "u"
SET_MISSING = "m"


def generate_statistics(individual, mode, date):
    """
    Outputs a xlsx file that contains information how the predicted actions compare to the user evaluation.
    """
    save_path_ohem_csv = "{}{}/raw/{}/ohem/{}_{}_{}_ohem.csv".format(
        INPUT_BASE, individual, mode, date, individual, mode
    )

    # checks if path to .csv file exists
    if not os.path.exists(save_path_ohem_csv):
        print("WARNING: Path does not exist.", save_path_ohem_csv)
    else:
        df = pd.read_csv(save_path_ohem_csv)

    number_images = len(df.index)

    df_stats = pd.DataFrame(
        {"Number_images": [number_images], "correct": [0], "incorrect": [0]}
    )

    # checks if predicted action matches evaluated one
    for row in df.index:
        if df.loc[row, "Evaluation"] == df.loc[row, "Action"]:
            df_stats.loc[0, "correct"] += 1
        elif df.loc[row, "Evaluation"] != df.loc[row, "Action"]:
            df_stats.loc[0, "incorrect"] += 1

    percentage_correct = round(
        df_stats.loc[0, "correct"] / df_stats.loc[0, "Number_images"] * 100
    )
    percentage_incorrect = round(
        df_stats.loc[0, "incorrect"] / df_stats.loc[0, "Number_images"] * 100
    )

    print("Statistics for individual: ", individual)
    print("Total number of evaluated images: ", df_stats.loc[0, "Number_images"])
    print(
        "Number of correct images:",
        df_stats.loc[0, "correct"],
        "this corresponds to",
        percentage_correct,
        "%",
    )
    print(
        "Number of incorrect images:",
        df_stats.loc[0, "incorrect"],
        "this corresponds to",
        percentage_incorrect,
        "%",
    )

    df_stats.to_csv(
        "{}{}/raw/{}/ohem/{}_{}_{}_ohem_stats.csv".format(
            INPUT_BASE, individual, mode, date, individual, mode
        ),
        index=False,
    )


def image_evaluation():
    """
    Main method to actually perform the evaluation.
    - creates GUI to evaluate the action
    - saving user evaluation
    """

    def _add_borders(img, evaluation):
        """
        Extends the image by a border visualising the given evaluation with a specific color.
        """
        global COLORCODES_EVALUATION

        # creation of image border
        img_border = np.zeros((420, 420, 3), np.uint8)
        img_border[0:420, 0:420] = COLORCODES_EVALUATION[evaluation]
        img_border[18:402, 18:402] = img

        return img_border

    def _add_action(img_border, action):
        """
        Shows predicted action to user.
        """

        # creates space above the image to show predicted action
        action_border = np.zeros((480, 420, 3), np.uint8)
        action_border[0:480, 0:420] = (255, 255, 255)
        action_border[60:480, 0:420] = img_border

        # settings for display of the predicted action
        font = cv2.FONT_HERSHEY_DUPLEX
        position = (30, 40)
        font_scale = 1
        font_color = (0, 0, 0)

        # shows predicted action
        cv2.putText(action_border, action, position, font, font_scale, font_color)

        return action_border

    def _save_changes(subset_df, individual, mode, date):
        """
        Saves csv with evaluated values.
        """

        save_path_ohem_csv = "{}{}/raw/{}/ohem/{}_{}_{}_ohem.csv".format(
            INPUT_BASE, individual, mode, date, individual, mode
        )

        # concatenates new and existing csv to save together
        # is needed, when evaluating more than once
        if os.path.exists(save_path_ohem_csv):
            previous_ohem_df = pd.read_csv(save_path_ohem_csv)
            concatenated_ohem_df = pd.concat([previous_ohem_df, subset_df], axis=0)
            concatenated_ohem_df.to_csv(save_path_ohem_csv, index=False)
        # saves csv the first time
        else:
            subset_df.to_csv(save_path_ohem_csv, index=False)

    def _create_subset_csv(df, subset_images):
        """
        Creates subset of images to evaluate and saves as .csv
        """

        # creates dataframe for image subset
        df_columns = df.columns
        subset_df = pd.DataFrame(columns=df_columns)

        # saves chosen images to subset csv
        for row in df.index:
            if df.loc[row, "img_name"] in subset_images:
                subset_df.loc[len(subset_df)] = df.loc[row]

        # creates empty rows for action prediction and user evaluation
        subset_df["Action"] = ""
        subset_df["Evaluation"] = "none"

        # casting evaluation for the statistic
        if mode == "StLy":
            subset_df["Casted_Evaluation"] = "none"

        # sets predicted action to the most probable action, the name of the action is taken from the headings of the csv file
        for row in subset_df.index:
            if (
                subset_df.loc[row, subset_df.columns[1]]
                > subset_df.loc[row, subset_df.columns[2]]
            ):
                subset_df.loc[row, "Action"] = subset_df.columns[1].lower()
            else:
                subset_df.loc[row, "Action"] = subset_df.columns[2].lower()

        return subset_df

    def _read_csv(individual, mode, date):
        """
        Reads CSV file with prediction and determines the most probable action.
        """

        # checks if path to .csv file exists
        path_ac_prediction = "{}{}/raw/{}/prediction/{}_{}_{}.csv".format(
            INPUT_BASE, individual, mode, date, individual, mode
        )

        try:
            os.path.exists(path_ac_prediction)
            df = pd.read_csv(path_ac_prediction)
        except:
            print("Error: file {} not existent.".format(path_ac_prediction))

        return df

    def get_ohem_information():
        """
        Reads .xlsx file containing all ohem information.
        Returns: dataframe with ohem information.
        """

        path_ohem_information = (
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            + "/config/ohem_information.xlsx"
        )
        ohem_df = pd.read_excel(path_ohem_information)

        return ohem_df

    def create_image_subset(df, number, action_one, action_two):
        """
        Determines which images belong to the subset which gets evaluated.
        """

        critical_predictions = []

        # choose which images have critical value
        for i in range(len(df.index)):
            if (
                df.loc[i, action_one] < df.loc[i, action_two]
                and df.loc[i, action_two] < CRITICAL_VALUE
            ):
                critical_predictions.append(df.loc[i, "img_name"])
            elif (
                df.loc[i, action_one] > df.loc[i, action_two]
                and df.loc[i, action_one] < CRITICAL_VALUE
            ):
                critical_predictions.append(df.loc[i, "img_name"])

        # test if there are enough critical value images
        if len(critical_predictions) >= (round(number * PERCENTAGE_HARD_IMAGES)):
            # choose right amount of critical value images randomly
            selected_predictions = random.choices(
                critical_predictions, k=round(number * PERCENTAGE_HARD_IMAGES)
            )
        else:
            selected_predictions = critical_predictions

        # append with random images
        while len(selected_predictions) < number:
            random_prediction = df.sample(n=1)
            if not random_prediction.iloc[0]["img_name"] in selected_predictions:
                selected_predictions.append(random_prediction.iloc[0]["img_name"])

        return selected_predictions

    def create_subset_folder(subset_images):
        """
        Creates folder to save images that get evaluated later.
        """

        ensure_directory(
            "{}{}/raw/{}/ohem/evaluation_subset/".format(INPUT_BASE, individual, mode)
        )

        destination = "{}{}/raw/{}/ohem/evaluation_subset/".format(
            INPUT_BASE, individual, mode
        )

        for i in subset_images:
            current_image_path = "{}predicted_images/{}/{}/{}/images/0/{}".format(
                IMAGES_BASE, enclosure, date, individual, i
            )
            sh.copy2(src=current_image_path, dst=destination)

    def store_images_mode(
        action_one, action_two, mode, individual, class_values, dataset
    ):
        """
        Creates folders to save images that get evaluated gathered by their mode.
        """
        # creates directories to store evaluated images for each mode
        ensure_directory(
            f"{OUTPUT_BASE}{dataset}/{mode}/{class_values[0]}/{individual}/images/0/"
        )  # vor mode subset namen
        ensure_directory(
            f"{OUTPUT_BASE}{dataset}/{mode}/{class_values[1]}/{individual}/images/0/"
        )

        # path to csv file with evaluations and to evaluated images
        evaluated_csv = "{}{}/raw/{}/ohem/{}_{}_{}_ohem.csv".format(
            INPUT_BASE, individual, mode, date, individual, mode
        )

        # read csv file and store in dataframe
        evaluated_df = pd.read_csv(evaluated_csv)

        for evaluated_image in evaluated_df.index:
            # selects folder in which image gets insetred
            source = "{}{}/raw/{}/ohem/evaluation_subset/{}".format(
                INPUT_BASE,
                individual,
                mode,
                evaluated_df.loc[evaluated_image, "img_name"],
            )
            if (
                evaluated_df.loc[evaluated_image, "Casted_Evaluation"]
                == action_one.lower()
            ):
                destination = f"{OUTPUT_BASE}{dataset}/{mode}/{class_values[0]}/{individual}/images/0/"
                # copy image to folder
                sh.copy2(src=source, dst=destination)
            elif (
                evaluated_df.loc[evaluated_image, "Casted_Evaluation"]
                == action_two.lower()
            ):
                destination = f"{OUTPUT_BASE}{dataset}/{mode}/{class_values[1]}/{individual}/images/0/"
                # copy image to folder
                sh.copy2(src=source, dst=destination)

    # get user ohem information
    ohem_df = get_ohem_information()

    # starts evaluation
    for row in ohem_df.index:

        # get parameters from excel
        enclosure = ohem_df.loc[row, "enclosure_id"]
        individual = ohem_df.loc[row, "individual_ids"]
        mode = ohem_df.loc[row, "ac_mode"]
        dataset_name = ohem_df.loc[row, "dataset_name"]

        class_names = list(BEHAVIORS_BY_MODE[mode].keys())
        class_values = list(BEHAVIORS_BY_MODE[mode].values())

        action_one = class_names[0]
        action_two = class_names[1]

        date = ohem_df.loc[row, "date"].strftime("%Y-%m-%d")
        number_images_subset = ohem_df.loc[row, "number_images"]

        # creates image subset from all images
        df = _read_csv(individual, mode, date)
        subset_images = create_image_subset(
            df, number_images_subset, action_one, action_two
        )
        subset_df = _create_subset_csv(df, subset_images)

        create_subset_folder(subset_images)

        # parameters for evaluation
        number_of_images = len(subset_df)
        current_image_id = 0
        display = True

        # creates window to show image and receive user input
        while display:
            # sets current image path and action
            current_image_path = "{}{}/raw/{}/ohem/evaluation_subset/{}".format(
                INPUT_BASE, individual, mode, subset_images[current_image_id]
            )
            # reads current image
            img = imread(current_image_path)

            # adds border and action to image
            action = subset_df.loc[current_image_id, "Action"]
            image_to_display = _add_borders(
                img, subset_df.loc[current_image_id, "Evaluation"]
            )
            image_to_display = _add_action(image_to_display, action)

            # creates window where the current image is shown
            cv2.imshow("Please evaluate the label.", image_to_display)

            # receives user input from keyboard
            key_pressed = False
            while not key_pressed:
                key = cv2.waitKey(0)

                # keyboard input to move between the images
                if key == ord(MOVE_LEFT) and current_image_id > 0:
                    current_image_id -= 1
                    key_pressed = True

                elif key == ord(MOVE_RIGHT) and current_image_id < (
                    number_of_images - 1
                ):
                    current_image_id += 1
                    key_pressed = True

                elif key == ord(END_EVALUATION):
                    display = False
                    key_pressed = True

                elif key == ord(MOVE_NEXT_UNLABELLED):
                    while current_image_id < number_of_images - 1:
                        current_image_id += 1
                        if df.loc[current_image_id, "Evaluation"] == "none":
                            break
                    key_pressed = True

                elif key == ord(MOVE_PREV_UNLABELLED):
                    while current_image_id > 0:
                        current_image_id -= 1
                        if df.loc[current_image_id, "Evaluation"] == "none":
                            break
                    key_pressed = True

                # keyboard input to set and save user evaluation for images
                elif key == ord(SET_UNLABELLED):
                    subset_df.loc[current_image_id, "Evaluation"] = "none"
                    key_pressed = True
                elif key == ord(SET_STANDING):
                    subset_df.loc[current_image_id, "Evaluation"] = "standing"
                    if mode == "StLy":
                        subset_df.loc[current_image_id, "Casted_Evaluation"] = (
                            "standing"
                        )
                        key_pressed = True
                elif key == ord(SET_STANDING_FOOD):
                    subset_df.loc[current_image_id, "Evaluation"] = "standing_food"
                    if mode == "StLy":
                        subset_df.loc[current_image_id, "Casted_Evaluation"] = (
                            "standing"
                        )
                    key_pressed = True
                elif key == ord(SET_LYING):
                    subset_df.loc[current_image_id, "Evaluation"] = "lying"
                    if mode == "StLy":
                        subset_df.loc[current_image_id, "Casted_Evaluation"] = "lying"
                    key_pressed = True
                elif key == ord(SET_LYING_HEAD_UP):
                    subset_df.loc[current_image_id, "Evaluation"] = "lying_head_up"
                    if mode == "StLy":
                        subset_df.loc[current_image_id, "Casted_Evaluation"] = "lying"
                    key_pressed = True
                elif key == ord(SET_LYING_HEAD_DOWN):
                    subset_df.loc[current_image_id, "Evaluation"] = "lying_head_down"
                    if mode == "StLy":
                        subset_df.loc[current_image_id, "Casted_Evaluation"] = "lying"
                    key_pressed = True
                elif key == ord(SET_MISSING):
                    subset_df.loc[current_image_id, "Evaluation"] = "missing"
                    key_pressed = True
                elif key == ord(END_EVALUATION):
                    display = False
                    key_pressed = True
        cv2.destroyAllWindows()
        _save_changes(subset_df, individual, mode, date)

        store_images_mode(
            action_one, action_two, mode, individual, class_values, dataset_name
        )

        generate_statistics(individual, mode, date)


if __name__ == "__main__":
    image_evaluation()
