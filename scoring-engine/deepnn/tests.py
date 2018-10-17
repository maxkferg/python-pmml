import json
import keras
import numpy as np
from core.conversion import convert
from core.intermediate import DeepNeuralNetwork
from core.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet201
from keras.applications.mobilenet import MobileNet
from scipy.ndimage import imread
from deepdiff import DeepDiff

VGG_16_MODEL = "models/VGG16/model.pmml"
VGG_19_MODEL = "models/VGG19/model.pmml"
RESNET50_MODEL = "models/ResNet50/model.pmml"
DENSENET121_MODEL = "models/DenseNet121/model.pmml"
DENSENET201_MODEL = "models/DenseNet201/model.pmml"
MOBILENET_PATH = "models/MobileNet/model.pmml"



def load_imagenet_classes():
	"""
	Return a map between class_id (int) and class_name (string)
	"""
	with open("assets/imagenet_classes.json") as fp:
		class_map = json.load(fp)
	class_map = {int(k):v for k,v in class_map.items()}
	return class_map


def test_convert_keras_to_pmml():
	print("--- Test Keras to PMML ---")
	keras_model = VGG16()
	class_map = load_imagenet_classes()
	pmml = convert(keras_model, class_map=class_map, description="VGG-16 Deep Neural Network Model")
	pmml.save_pmml(VGG_16_MODEL)

	keras_model = ResNet50()
	pmml = convert(keras_model, class_map=class_map,  description="ResNet50 Deep Neural Network Model")
	pmml.save_pmml(RESNET50_MODEL)


def test_pmml_to_intermediate():
	"""
	Load an PMML file to an intermediate form
	"""
	print("\n--- Test PMML to intermediate ---")
	intermediate = DeepNeuralNetwork(filename=VGG_16_MODEL)
	assert(intermediate.description=="VGG-16 Deep Neural Network Model")
	assert(len(intermediate.layers)==23)


def test_intermediate_to_keras(pmml_file, keras_model):
	"""
	Test conversion between intermediate form and keras
	Compare the loaded PMML file to @keras_model
	"""
	print("\n--- Test Intermediate to Keras (%s) ---"%pmml_file)
	intermediate = DeepNeuralNetwork(filename=pmml_file)
	new_keras_model = intermediate.get_keras_model()
	new_config = new_keras_model.get_config()
	old_config = keras_model.get_config()
	print(DeepDiff(old_config,new_config))
	"""
	for key,value in old_config.items():
		if key not in new_config:
			"Key %s missing from new (PMML) config"
		elif value != old_config[key]:
			"Values are different, Old={} New={}".format(old_config[key], value)
		else:
			print(key,"=",value)

	for key,value in old_config['layers'].items():
		if key not in new_config[]:
			"Key %s missing from new (PMML) config"
		elif value != old_config[key]:
			"Values are different, Old={} New={}".format(old_config[key], value)
		else:
			print(key,"=",value)
	"""

def test_cat_classification(model):
	print("\n--- Test cat classification (%s) ---"%model)
	filename = "assets/cat.jpg"
	intermediate = DeepNeuralNetwork(filename=model)
	class_name = intermediate.predict(imread(filename))
	print("Model selected class '{0}' for image {1}".format(class_name, filename))





if __name__=="__main__":
	#test_convert_keras_to_pmml()
	#test_pmml_to_intermediate()
	test_intermediate_to_keras(VGG_16_MODEL, VGG16())
	test_intermediate_to_keras(MOBILENET_PATH, MobileNet())
	test_cat_classification(VGG_16_MODEL)
	test_cat_classification(RESNET50_MODEL)


