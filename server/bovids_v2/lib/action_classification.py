#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains methods to 
	- predict a folder of images and to return a dictionary {img_name : class}
Methods:
	- predict_folder_ac( folder_path, ac_network, mode, device )
	  returns a dictionary {image_name: { class_index: (class_name, certainty) } }
"""

__author__ = ["Max Hahn-Klimroth", "Judith Ballmann", "Lea Möller"]
__copyright__ = "Copyright 2023, BOVIDS"
__credits__ = ["J. Gübert", "P. Dierkes"]
__license__ = "AGPLv3"
__version__ = "2.0"
__status__ = "Development"

import os, sys
import traceback
import pandas as pd

# make it possible to import own modules

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../")
from server.bovids_v2.config.get_config import BEHAVIORS_BY_MODE, BATCH_SIZE_PREDICTION_AC
from func import ensure_directory

# general libraries
import onnxruntime as ort

# torch libraries
from torchvision.models import efficientnet_v2_s
import torchvision.transforms as T
import torch
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


# Global caching variable
cached_ac_models = {}


class ImageFolderWithPaths(ImageFolder):
	"""Custom dataset that includes image file paths. Extends
	torchvision.datasets.ImageFolder
	"""

	# override the __getitem__ method. this is the method that dataloader calls
	def __getitem__(self, index):
		return super(ImageFolderWithPaths, self).__getitem__(index) + (
			self.imgs[index][0],
		)


def _load_model(ac_network, device):
	global cached_ac_models
	model = None
	# Optimize inference. ONNX and ENGINE export modes are not compatible to transfer between different CPUs/GPUs!
	# TODO: test if optimized version is stable and equivalent on GPU! Tobi confirmed this for CPU only!
	base_model_path = os.path.splitext(ac_network)[0]
	model_path_pt = ac_network
	model_path_onnx = f"{base_model_path}.onnx"
	model_path_engine = f"{base_model_path}.engine"

	is_gpu = str(device).startswith("cuda")
	if is_gpu:
		target_model_path = model_path_engine
		export_format = "engine"
	else:
		target_model_path = model_path_onnx
		export_format = "onnx"

	# Export model if not already exported
	if not os.path.exists(target_model_path):
		try:
			base_model = efficientnet_v2_s()
			n_features = base_model.classifier[1].in_features
			base_model.classifier[1] = nn.Linear(n_features, 2)
			base_model.load_state_dict(torch.load(model_path_pt, map_location=device))
			base_model.eval()
			base_model.to(device)

			dummy_input = torch.randn(1, 3, 384, 384, device=device)

			if export_format == "onnx":
				torch.onnx.export(
					base_model,
					dummy_input,
					target_model_path,
					opset_version=17,
					input_names=["input"],
					output_names=["output"],
					dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
				)
			else:  # engine (TensorRT)
				import tensorrt as trt
				# Export to ONNX first as intermediate step, then build TRT engine
				onnx_tmp_path = f"{base_model_path}_tmp.onnx"
				torch.onnx.export(
					base_model,
					dummy_input,
					onnx_tmp_path,
					opset_version=17,
					input_names=["input"],
					output_names=["output"],
					dynamic_axes=None,  # static for TensorRT, usually faster
				)
				logger = trt.Logger(trt.Logger.WARNING)
				builder = trt.Builder(logger)
				network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
				parser = trt.OnnxParser(network, logger)
				with open(onnx_tmp_path, "rb") as f:
					parser.parse(f.read())
				config = builder.create_builder_config()
				config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))  # 4 GB
				if builder.platform_has_fast_fp16:
					config.set_flag(trt.BuilderFlag.FP16)
				serialized_engine = builder.build_serialized_network(network, config)
				with open(target_model_path, "wb") as f:
					f.write(serialized_engine)
				os.remove(onnx_tmp_path)

		except Exception as e:
			print(f"WARNING. Could not export AC model to {export_format}, falling back to .pt ({e})")
			target_model_path = model_path_pt

	if target_model_path in cached_ac_models:
		model = cached_ac_models[target_model_path]
	else:
		try:
			if target_model_path.endswith(".onnx"):
				model = ort.InferenceSession(
					target_model_path,
					providers=["CPUExecutionProvider"],
				)
			elif target_model_path.endswith(".engine"):
				import tensorrt as trt
				logger = trt.Logger(trt.Logger.WARNING)
				runtime = trt.Runtime(logger)
				with open(target_model_path, "rb") as f:
					model = runtime.deserialize_cuda_engine(f.read())
			else:  # .pt fallback
				model = efficientnet_v2_s()
				n_features = model.classifier[1].in_features
				model.classifier[1] = nn.Linear(n_features, 2)
				model.load_state_dict(torch.load(target_model_path, map_location=device))
				model.eval()
				model.to(device)
			cached_ac_models[target_model_path] = model
		except Exception as e:
			print(f"ERROR. Could not load model weights ({target_model_path}): {e}")
	return model


@torch.inference_mode()
def predict_folder_ac(args):
	"""
	folder_path: path to folder, containing jpg images to predict
	ac_network: path to action classification model
	mode: StLy, StFo, LHULHD
	device: cuda:0 or cpu

	returns a dictionary
	{image_name: { class_index: (class_name, certainty) } }
	"""
	(
		folder_path,
		imagefilenames_to_use,
		ac_network,
		mode,
		device,
		path_ac_save,
		date,
		individual,
		path_ac_stly,
		pot_available_imgs,
	) = args

	# load model if existent
	model = _load_model(ac_network, device)

	# use same transformations as in validation while training
	# TODO: here and in train_action_classification, we should use global vars
	transformations = T.Compose(
		[
			T.Resize(size=384, interpolation=T.InterpolationMode.BILINEAR),
			T.CenterCrop(size=384),
			T.ToTensor(),
			T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)

	# load images to predict
	try:
		image_data = ImageFolderWithPaths(folder_path, transformations)
		idx = [
			i
			for i in range(len(image_data))
			if os.path.basename(image_data.imgs[i][0]) in imagefilenames_to_use
		]
		subset_data = Subset(image_data, idx)
		data_loader = DataLoader(
			subset_data, batch_size=BATCH_SIZE_PREDICTION_AC, shuffle=False
		)

	except:
		print(f"ERROR. Could not load image folder {folder_path}.")
		return {}

	try:
		class_names = list(BEHAVIORS_BY_MODE[mode].keys())
		class_indices = list(BEHAVIORS_BY_MODE[mode].values())
	except:
		print(f"ERROR. Mode {mode} unknown.")
		return {}

	num_classes_model = model.__dict__["_modules"]["classifier"][1].__dict__["out_features"]  # TODO: is there a better way? this seems stupid


	if num_classes_model != len(class_names):
		print(
			f"ERROR. The loaded model has {num_classes_model} output features while the classification task {mode} requires {len(class_names)} classes."
		)
		return {}

	predictions = {}
	# num images: len(image_data), classes image_data.classes
	# with torch.no_grad():
	for inputs, labels, paths in data_loader:
		# inputs and labels are the single batches
		inputs = inputs.to(device)
		labels = labels.to(device)
		outputs = model(inputs)  # list of list of certainties
		prob = torch.nn.functional.softmax(
			outputs, dim=1
		)  # list of predicted labels (maximum class in outputs)
		top_p, top_class = prob.topk(len(class_names), dim=1)
		# paths: list of image paths to which the predictions belong
		tmp = {
			os.path.basename(paths[j]): {
				class_indices[top_class[j][cl]]: (
					class_names[top_class[j][cl]],
					float(top_p[j][cl]),
				)
				for cl in range(len(class_names))
			}
			for j in range(len(prob))
		}
		predictions.update(tmp)

	df_ac_data = save_ac_predictions(predictions, path_ac_save, date, mode, individual)
	if mode == "StLy":
		create_stly_sequence(
			df_ac_data, path_ac_save, date, mode, individual, pot_available_imgs
		)
	else:
		create_subaction_sequence(
			df_ac_data, path_ac_save, date, mode, individual, path_ac_stly, pot_available_imgs
		)

	return predictions


def save_ac_predictions(ac_predictions, path_ac_save, date, mode, individual):
	"""
	Method to save action classification predictions
	"""

	class_names = list(BEHAVIORS_BY_MODE[mode].keys())
	action_one = []
	action_two = []

	for value in ac_predictions.values():
		for x in value.values():
			if x[0] == class_names[0]:
				action_one.append(x[1])
			elif x[0] == class_names[1]:
				action_two.append(x[1])

	# creating dataframe with image name, percentage standing and percentage Lying
	df_ac_data = pd.DataFrame(data=ac_predictions.keys(), columns=["img_name"])
	df_ac_data[class_names[0]] = action_one
	df_ac_data[class_names[1]] = action_two
	# safe dataframe as .csv
	df_ac_data.to_csv(f"{path_ac_save}{date}_{individual}_{mode}.csv", index=False)

	return df_ac_data


def create_stly_sequence(
	df_ac_data, path_ac_save, date, mode, individual, pot_available_imgs
):

	behavior_seq = []

	class_names = list(BEHAVIORS_BY_MODE[mode].keys())

	list_all_imgs = pot_available_imgs[individual][date]
	list_all_imgs = [i + ".jpg" for i in list_all_imgs]
	list_df = list(df_ac_data["img_name"])

	for i in range(len(list_all_imgs)):
		if list_all_imgs[i] not in set(list_df):
			# append 0 for image out of view
			behavior_seq.append(0)
		else:
			index = list_df.index(list_all_imgs[i])
			# if the probability of Standing is higher
			if (
				df_ac_data.loc[index, class_names[0]]
				>= df_ac_data.loc[index, class_names[1]]
			):
				behavior_seq.append(BEHAVIORS_BY_MODE[mode][class_names[0]])
			# if the probability of Lying is higher
			elif (
				df_ac_data.loc[index, class_names[0]]
				< df_ac_data.loc[index, class_names[1]]
			):
				behavior_seq.append(BEHAVIORS_BY_MODE[mode][class_names[1]])

	behavior_df = pd.DataFrame(behavior_seq)
	behavior_df.to_csv(
		f"{path_ac_save}{date}_{individual}_{mode}_behavior_seq.csv",
		index=False,
		header=False,
	)


def create_subaction_sequence(
	subaction_probs, path_ac_save, date, mode, individual, path_ac_stly, pot_imgs
):

	behavior_seq = []
	stly_probs = pd.read_csv(f"{path_ac_stly}{date}_{individual}_StLy.csv")
	class_names = list(BEHAVIORS_BY_MODE[mode].keys())
	list_all_imgs = pot_imgs[individual][date]
	list_all_imgs = [i + ".jpg" for i in list_all_imgs]

	subaction_row = 0
	list_df = list(stly_probs["img_name"])
	counter = 0

	for i in range(len(list_all_imgs)):
		if list_all_imgs[i] not in set(list_df):
			# append 0 for image out of view
			behavior_seq.append(None)
			counter += 1
		else:

			if (len(list(subaction_probs)) > 0):
				behavior_seq.append(None)
				continue
			if (
					stly_probs.loc[i - counter, "img_name"]
					== subaction_probs.loc[subaction_row, "img_name"]
			):
				# if the probability for no food / LHU is higher
				if (
						subaction_probs.loc[subaction_row, class_names[0]]
						>= subaction_probs.loc[subaction_row, class_names[1]]
				):
					behavior_seq.append(BEHAVIORS_BY_MODE[mode][class_names[0]])
				# if the probability for food / LHD is higher
				elif (
						subaction_probs.loc[subaction_row, class_names[0]]
						< subaction_probs.loc[subaction_row, class_names[1]]
				):
					behavior_seq.append(BEHAVIORS_BY_MODE[mode][class_names[1]])
				if subaction_row < len(subaction_probs) - 1:
					subaction_row += 1
			else:
				behavior_seq.append(None)



	"""
	for stly_row in stly_probs.index:
		#print("stly_probs image name: ",stly_probs.loc[stly_row, "img_name"])
		#print("subaction probs img name:", subaction_probs.loc[subaction_row, "img_name"])

		

		if (
			stly_probs.loc[stly_row, "img_name"]
			== subaction_probs.loc[subaction_row, "img_name"]
		):
			# if the probability for no food / LHU is higher
			if (
				subaction_probs.loc[subaction_row, class_names[0]]
				>= subaction_probs.loc[subaction_row, class_names[1]]
			):
				behavior_seq.append(BEHAVIORS_BY_MODE[mode][class_names[0]])
			# if the probability for food / LHD is higher
			elif (
				subaction_probs.loc[subaction_row, class_names[0]]
				< subaction_probs.loc[subaction_row, class_names[1]]
			):
				behavior_seq.append(BEHAVIORS_BY_MODE[mode][class_names[1]])
			if subaction_row < len(subaction_probs) - 1:
				subaction_row += 1
		else:
			print("append none")
			behavior_seq.append(None)
	"""
	behavior_df = pd.DataFrame(behavior_seq)
	behavior_df.to_csv(
		f"{path_ac_save}{date}_{individual}_{mode}_behavior_seq.csv",
		index=False,
		header=False,
	)
