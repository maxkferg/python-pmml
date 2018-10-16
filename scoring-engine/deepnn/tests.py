import json
import keras
import numpy as np
from conversion import convert
from intermediate import DeepNeuralNetwork
from layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet201
from scipy.ndimage import imread

VGG_16_MODEL = "models/VGG16/model.pmml"
VGG_19_MODEL = "models/VGG19/model.pmml"
RESNET50_MODEL = "models/ResNet50/model.pmml"
DENSENET121_MODEL = "models/DenseNet121/model.pmml"
DENSENET201_MODEL = "models/DenseNet201/model.pmml"


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
	pmml = convert(keras_model, description="VGG-16 Deep Neural Network Model")
	pmml.save_pmml(VGG_16_MODEL)

	keras_model = VGG19()
	pmml = convert(keras_model, description="VGG-19 Deep Neural Network Model")
	pmml.save_pmml(VGG_19_MODEL)

	keras_model = ResNet50()
	pmml = convert(keras_model, description="ResNet50 Deep Neural Network Model")
	pmml.save_pmml(RESNET50_MODEL)

	keras_model = DenseNet121()
	pmml = convert(keras_model, description="DenseNet121")
	pmml.save_pmml(DENSENET121_MODEL)



def test_pmml_to_intermediate():
	"""
	Load an PMML file to an intermediate form
	"""
	print("\n--- Test PMML to intermediate ---")
	intermediate = DeepNeuralNetwork(filename=VGG_16_MODEL)
	assert(intermediate.description=="VGG-16 Deep Neural Network Model")
	assert(len(intermediate.layers)==23)


def test_intermediate_to_keras_vgg():
	"""
	Test conversion between intermediate form and keras
	"""
	print("\n--- Test Intermediate to Keras ---")
	intermediate = DeepNeuralNetwork(filename=VGG_19_MODEL)
	model = intermediate.get_keras_model()


def test_intermediate_to_keras_resnet():
	"""
	Test conversion between intermediate form and keras
	"""
	print("\n--- Test Intermediate to Keras (ResNet50) ---")
	intermediate = DeepNeuralNetwork(filename=RESNET50_MODEL)
	model = intermediate.get_keras_model()


def test_cat_classification(model):
	print("\n--- Test cat classification (%s) ---"%model)
	intermediate = DeepNeuralNetwork(filename=model)
	model = intermediate.get_keras_model()
	filename = "assets/cat.jpg"
	cat = imread(filename)
	batch = cat[None,:]
	result = model.predict(batch)
	class_id = np.argmax(result)
	class_map = load_imagenet_classes()
	class_name = class_map[class_id]
	print("Model selected class '{0}' for image {1}".format(class_name, filename))


if __name__=="__main__":
	#test_convert_keras_to_pmml()
	#test_pmml_to_intermediate()
	#test_intermediate_to_keras_vgg()
	#test_intermediate_to_keras_resnet()
	test_cat_classification(VGG_16_MODEL)
	test_cat_classification(VGG_19_MODEL)
	test_cat_classification(RESNET50_MODEL)
	test_cat_classification(DENSENET121_MODEL)
	test_cat_classification(DENSENET201_MODEL)



