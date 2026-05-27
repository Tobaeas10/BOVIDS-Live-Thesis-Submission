# Thesis Submision Note
This is an anonymized copy of the BOVIDS GitHub repository right before thesis submission. The main contribution of the thesis can be found in `local_pc/run_stream_service.py`.

Other files have also been created an implemented by me, including `bovids_live_test_stream_management_script.py`, `thesis_tests.py`, `status_monitor_for_BOVIDS_Live.py`, the automatic installers and start shortcuts in the base directory as well as all stream-related testing .xlsx's.

A list of other changed files I co-authored is: `action_classification.py`, `image_manipulation.py`, `object_detection.py` and `post_processor.py`.

To actually test BOVIDS-Live, you will need videos of within a zoo enclosure which are not included here due to data limits. Email `tobias.weiss@tuhh.de` to request access to them. Alternatively, just put any two videos in `/server/bovids_v2/test/active videos/` and start BOVIDS-Live by running `run_BOVIDS2_in_stream_mode.bat`

Thank you for taking the time to read this.

-- Tobias

# Automatic Installation Guide
Just run `install_BOVIDS2.bat` as an administrator. It will install python, add it to the system path and then install all BOVIDS2 dependencies.

To run BOVIDS2 in stream mode, simply run `run_BOVIDS2_in_stream_mode.bat`.

# Manual Installation Guide
Before starting the programm it is nessesary to install several required packages. All packages can be found in the file "requirements_yolov8.txt". 
For installation, run the following commands in the terminal:

pip install -r requirements_yolov8.txt

pip install ultralytics

# BOVIDS2

## General
- all paths that are entered into the code follow the unix standard (/) and not windows standard (\\)

## BOVIDS Prediction Pipeline
The prediction pipeline is the main part of BOVIDS. It performs the desired predictions for specified individuals.

### Excel files
To use the different functionalities of this program the following .xlsx files are needed: config/enclosure_information.xlsx, config/individual_information.xlsx and config/post_processing_rules.xlsx.
You only need to change those files when you want to add new enclosures, individuals or post processing rules to rerun the program.  
Additionaly the stuff/predict_example.xlsx file needs to be edited for each execution of the pipeline, to specify the executed tasks.

#### enclosure_information.xlsx
This file stores the information for the possibly used enclosures.
Therefore you have to fill in the .xlsx file as follows:
  - enclosure_id: set the enclosure id as given in the video file name (e.g. Animal-Enclosure_Zoo1_1)
  - individual_ids: id for each individual in the video seperated by comma (e.g. Animal-Enclosure_Zoo1_ind_1, Animal-Enclosure_Zoo1_ind_2)
  - object_detector: name of the object detector you want to use without file format, the detector has to be located in the folder res/objectdetection/ (e.g. basenet_individual1_individual2)
  - od_confidence: confidence value for object detection between 0 and 1 (pre-defined value: 0.1)
  - od_iou: value to define movement (pre-defined value: 0.5)
  - image_segementor: currently not used
  - seg_confidence: currently not used
  - seg_iou: currently not used
  - task: currently only "detect" is possible
  - recording_start: time point at which the recording of the video starts (e.g. 17)
  - recording_end: time point at which the recording of the video ends (e.g. 7)
  - video_stream_folder: path that leads to folder that contains video file, starting from the anchorpoint for all videos (e.g. videomaterial/species/zoo/videos)
  - video_name: videofile name with only enclosure id and file format, without date (e.g. Animal-Enclosure_Zoo1_1.avi)
  - comments: personal comments, not used in the program

#### individual_information.xlsx
This file stores the information for the possibly used individuals.
Therefore you have to fill in the .xlsx file as follows:
  - individual_id: same ids as in the enclosure_information.xlsx, but each individual as new row (e.g. Animal-Enclosure_Zoo1_ind_1)
  - action_classifier_StLy: name of the StLy action classifier you want to use with file format, the classifier has to be located in the folder res/actionclassification/ (e.g. 2023-12-14_StLy_Bovids-best.pth)
  - action_classifier_StFo: name of the StFo action classifier you want to use with file format, the classifier has to be located in the folder res/actionclassification/ (e.g. 2024-05-31_StFo_Bovids-best.pth)
  - action_classifier_LHULHD: name of the LHULHD action classifier you want to use with file format, the classifier has to be located in the folder res/actionclassification/ (e.g.2024-05-28_LHULHD_Bovids-best.pth)
  - IOU_Moving: currently not used
  - postproc_StLy: set postprocessing rules for StLy to the name used in post_processing_rules.xlsx (e.g. bovidae_standard)
  - postproc_StFo: set postprocessing rules for StFo to the name used in post_processing_rules.xlsx (e.g. bovidae_standard)
  - postproc_LHULHD: set postprocessing rules for LHULHD to the name used in post_processing_rules.xlsx (e.g. bovidae_standard)
  - postproc_Moving: currently not used

#### post_processing_rules.xlsx
Additionally, this file is used to define the rules used for post processing.
These rules are used to filter out short phases, which means that an action has such a short duration that it doesn't have to be considered.  
The file consists of 3 tabs, for each action (StLy, StFo, LHULHD) one tab, with the specific rules.  
The first columns (A-E) define the behavior sequence for each rule and do not need to be changed.
Starting at column F different rulesets can be defined, which vary in the duration of the action that is filtered out.  
Each ruleset has an own name that can be accessed from individual_information.xlsx, to defined a ruleset for each individual.
A predefined ruleset (bovidae_standard) is already available, therefore the file only has to be changed to add a new ruleset if needed.

#### predict_example.xlsx
This contains all information used for one specific pipeline run. 
It defines which specific tasks should be done for each individual.
Therefore you have to fill in the .xlsx file as follows:

- enclosure_id: set the enclosure id as given in the video file name (e.g. Animal-Enclosure_Zoo1_1)
- date: set the date as given in the video file name (e.g. 22.11.2022)
- evaluation_start: time point at which the evaluation of the video starts (e.g. 17)
- evaluation_end: time point at which the evaluation of the video ends (e.g. 7)
- dismiss_individuals: individual_ids that are not considered (e.g. Animal-Enclosure_Zoo1_ind_1)
- StLy, StFo, LHULHD: set for which mode(s) the prediction is executed (0 for no and 1 for yes)
- Moving: currently not usable, leave at 0
- use_cached_od: set if already computed object detection data is used (0 for no and 1 for yes)
- use_cached_StLy, use_cached_StFo, use_cached_LHULHD: set for which mode(s) already computed action classification data is used (0 for no and 1 for yes)
- use_cached_Moving: currently not usable, leave at 0
- apply_postprocessing: set if post processing is applied (0 for no and 1 for yes)
- stats_StLy, stats_StFo, stats_LHULHD: currently not used
- stats_Moving: currently not used
- stats_Final: currently not used

### Global variables

To start the prediction pipeline the script local_pc/run_prediction.py needs to be executed.
Before the execution you need to adjust the following variables:

- ANCHORPATH_VIDEOFILES: anchor path for all videos, such that in this folder the relative path in the .xlsx file enclosure_information starts 
- SERVERPATH_OBJECTDETECTION_FILES: anchor path to save (and later access) object detection and segmentation images on the server (e.g. path to server + /object_detection/predicted_images/)
- SERVERPATH_ACTION_CLASSIFICATION: anchor path to save (and later access) action classification on the server (e.g. path to server + /action_classification/)
- TEMPORAL_LOCAL_STORAGE: local path to store temporary files such as videos and cutout images (e.g. C:/BOVIDS/object_detection/)
- BOVIDS_AC: local or server path to store action classification and post processing results  (e.g. C:/BOVIDS/action_classification/)
- BATCH_SIZE: number of videos that are fetched and translated into single images in parallel (predefined value: 4)
- BOVIDS_LIBRARY: path to BOVIDS library (predefined path: ../server/bovids_v2/)
- PREDICTION_XLSX: path to excel file that contains the information which nights should be predicd (e.g. ../server/bovids_v2/stuff/predict_example.xlsx)
- DEVICE: sets device that is used to cuda or cpu (predefined value: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

## Train Action Classification Network
Action classification networks can be trained to predict actions of individuals.

### Excel files

To train a prediction network the following .xlsx files are needed: stuff/ac_test.xlsx and config/boris_information.xlsx. 
The file ac_test.xlsx needs to be changed for every run, while boris_information.xlsx only needs to changed when new boris data should be added. 

#### ac_test.xlsx:
In ac_test.xlsx is defined, for which enclosures and night the network gets trained. 
Therefore you have to fill in the .xlsx file as follows:

  - enclosure_id: set the enclosure id as given in the video file name (e.g. Animal-Enclosure_Zoo1_1)
  - date: set the date as given in the video file name (e.g. 22.11.2022)
  - desired_start: set time from when the data is used to train network (e.g. 17)
  - desired_end: set time until when the data is used to train network (e.g. 7)
  - dismiss_individuals: individual_ids that are not considered (e.g. Animal-Enclosure_Zoo1_ind_1)
  - comment: personal comments, not used in the program

#### boris_information.xlsx
In boris_information.xlsx the information for the possibly used boris files is stored.
Therefore you have to fill in the .xlsx files as follows:

- individual_id: id from the individual that was evaluated in the boris file (e.g. Animal-Enclosure_Zoo1_ind_1)
- boris_name: individual name used in the column 'subject' of the boris file
- boris_start: start time of boris evaluation (e.g. 17)
- boris_end: end time of boris evaluation (e.g. 7)
- borisfiles_folder: path leading to the folder that contains boris files, starting from the anchorpoint for all boris information (e.g. Animal-Enclosure/)
- borisfiles_name: name of the boris file without the date (e.g. for file 2022-11-22-_Animal-Enclosure_Zoo1_1.xlsx: Animal-Enclosure_Zoo1_1)
- BehaviourMapping_StLy: predefined value is Bovid_StLy
- BehaviourMapping_LHULHU: predefined value is Bovid_StLHULHD
- BehaviourMapping_StFo: predefined value is Bovid_StFo
- Comments: personal comments, not used in the programm

### Global variables

To train the prediction network the script scripts/train_action_classification.py needs to be executed.
Before the execution you need to adjust the following variables:

#### General configuration
- DEVICE: sets device that is used to cuda or cpu (predefined value: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
- MODE: set mode for which the network should be created (StLy = Standing / Lying, LHULHD = LHD/LHU, StFo = Standing / Food)

#### Creation of a novel dataset
- INPUT_XLSX_DATASETCREATION: path to .xlsx file defining which enclosures are used for training (e.g. ../server/bovids_v2/stuff/ac_test.xlsx)
- OUTPUT_PATH_NEW_DATASETCREATION: path leading to a folder to save all datasets, each folder for a dataset is given an unique name (e.g. C:/BOVIDS/ac_datasets/Animal-Enclosure/)
- ANCHORPATH_BORIS_ANNOTATIONS: path that leads to the folder containing boris information, such that config/boris_information.xlsx contains the relative path to the boris annotation files (e.g. C:/BOVIDS/boris_data/)
- ANCHORPOINT_VIDEOFILES: anchor path for all videos, such that in this folder the relative path in the .xlsx file enclosure_information starts 
- OBJECT_DETECTION_MODE: use the mode from object detection, possible values are detect or segment, do not merge them (e.g.'detect')
- MERGE: boolean value, if datasets should be merged (e.g. True or False) 
- MAXIMUM_NUMER_SAMPLES_PER_VIDEO: caps the number of time intervals (balanced) sampled from one video (predefined value: 30)
- REMOVE_TEMPORARY_FILES: boolean value, if samples (intervals) and corresponding difference images should be deleted (e.g. True or False)

#### Merging of existing datasets
- MERGE_DATASETS: dictionary to merge existing datasets
  - 'datasets': list of paths leading to datasets (e.g. ['C:/BOVIDS/ac_datasets/Animal-Enclosure1/', 'C:/BOVIDS/ac_datasets/Animal-Enclosure2/'])
  - 'fraction_images_per_class': fraction of images for each action for each dataset(e.g. [{1: 1.0, 2: 1.0}, {1: 1.0, 2: 1.0}])
  - 'dismiss_individuals': individual_ids that are not considered, can be left blank (e.g. [[], []])

- AUTO_BALANCE: balances classes after merging datasets (e.g. True or False)
- OUTPUT_PATH_MERGED_DATASET: path to save merged dataset with unique name (e.g. C:/BOVIDS/ac_datasets/Animal-Enclosure-merged/)

#### preparing a training and validation set from a dataset
- DATASET_PATH: path to dataset used for training the network, either the novel or the merged dataset (e.g. C:/BOVIDS/ac_datasets/Animal-Enclosure/ or C:/BOVIDS/ac_datasets/Animal-Enclosure-merged/)
- TRAINING_VALIDATION_PATH: path to a new folder to store a subset of our dataset to train and validate (e.g. C:/BOVIDS/ac_datasets/Animal-Enclosure-trainingset/)
- VALIDATION_SPLIT: percentage of images used for validation (predefined value: 0.2)

- DATA_DIR_TRAINING_AND_VALIDATION: path to folder that contains the folders train/ and val/ (e.g. C:/BOVIDS/ac_datasets/Animal-Enclosure-trainingset/StLy/)
- PRETRAINED_NETWORK: .pth file that can be used as a base network, if nothing is given, imagenet pretrained network will be chosen
- MODELNAME: unique name to save trained model (e.g. 2023-07-12_Animal-Enclosure_StLy_Bovids)
- SAVEPATH_MODEL: directory in which the model, the training history and the checkpoints will be saved (C:/BOVIDS/ac_networks/)

- NUM_EPOCHS: number of training epochs (predefined value: 3)
- BATCH_SIZE: number of used batch (predefined value: 16)
- SAVE_EVERY_EPOCH: number of epochs, that are saved (predefined value: 3)


## OHEM AC

Online hard example mining (OHEM) is a way to create a new dataset with manually evaluated images to improve the prediction network.

### Excel files

To perform OHEM the following .xlsx file is needed: config/ohem_information.xlsx. 
This file specifies for which modes and individuals the evaluation should be done.
Therefore you have to fill in the .xlsx file as follows:

- enclosure_id: set the enclosure id as given in the video file name (e.g. Animal-Enclosure_Zoo1_1)
- individual_ids: id for each individual in the video (e.g. Animal-Enclosure_Zoo1_ind_1)
- ac_mode: set mode for which ohem is executed (possible values: StLy, StFo, LHULHD)
- date: set the date as given in the video file name (e.g. 22.11.2022)
- number_images: number of images that can be evaluated
- dataset_name: define a unique name to save the new dataset (e.g. Ohem_Animal-Enclosure_1)

### Global variables

To start OHEM the script scripts/ohem_action_classification.py needs to be executed.
Before the execution you need to adjust the following variables:

- INPUT_BASE: anchor path to folder where action classification results are saved (e.g. C:/BOVIDS/action_classification/)
- IMAGE_BASE: anchor path to folder where object detection results are saved (e.g. C:/BOVIDS/object_detection/)
- OUTPUT_BASE: path to save the new datasets with ohem (e.g. C:/BOVIDS/ac_datasets/)
- CRITICAL_VALUE: sets value that classifies predictions as uncertain (predefined value: 0.85)
- PERCENTAGE_HARD_IMAGES: percentage of images that get evaluated that have a critical value (predefined value: 0.7)

### Control Keys

To evaluate the actions the following keys can be used to navigate through the shown images.

- move to previous image: 4
- move to next image: 5
- end evaluation: p
- move to next unlabeled image: 9
- move to previous unlabeled image : 8

- evaluate image as standing: a
- evaluate image as standing food: s
- evaluate image as lying: l
- evaluate image as lying head up: k
- evaluate image as lying head down: j
- evaluate image as unlabeled: u
- evaluate image as missing: m
