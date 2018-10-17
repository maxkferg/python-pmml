import os
import json
import keras
import numpy as np
from pprint import pprint
from core.conversion import convert
from core.intermediate import DeepNeuralNetwork
from core.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from scipy.ndimage import imread


VGG_16_PATH = "models/VGG16/model.pmml"
VGG_19_PATH = "models/VGG19/model.pmml"
RESNET50_PATH = "models/ResNet50/model.pmml"
MOBILENET_PATH = "models/MobileNet/model.pmml"
INCEPTION_V3_PATH = "models/InceptionResNetV2/model.pmml"
INCEPTION_RESNET_PATH= "models/InceptionV3/model.pmml"
DENSENET121_PATH = "models/DenseNet121/model.pmml"
DENSENET169_PATH = "models/DenseNet169/model.pmml"
DENSENET201_PATH = "models/DenseNet201/model.pmml"


def dump_config(old_config, new_config, directory=""):
    old_config_dir = os.path.join(directory, "old_config.json")
    new_config_dir = os.path.join(directory, "new_config.json")
    with open(old_config_dir,"w") as fp:
        pprint(old_config, stream=fp)
    with open(new_config_dir,"w") as fp:
        pprint(new_config, stream=fp)
  

def load_imagenet_classes():
    """
    Return a map between class_id (int) and class_name (string)
    """
    with open("assets/imagenet_classes.json") as fp:
        class_map = json.load(fp)
    class_map = {int(k):v for k,v in class_map.items()}
    return class_map


def convert_keras_to_pmml(keras_model, output_path, description, debug=True):
    """
    Convert a keras model to PMML
    """
    print("\nGenerating model: %s"%description)
    class_map = load_imagenet_classes()
    pmml = convert(keras_model, class_map=class_map, description=description)
    pmml.save_pmml(output_path)
    if debug:
        print("Checking model: %s"%output_path)
        intermediate = DeepNeuralNetwork(filename=output_path)
        old_config = keras_model.get_config()
        new_config = intermediate.get_keras_model().get_config()
        directory = os.path.dirname(output_path)
        dump_config(old_config, new_config, directory)
        test_prediction(output_path, "assets/cat.jpg")


def test_prediction(output_path, image_path):
    intermediate = DeepNeuralNetwork(filename=output_path)
    class_name = intermediate.predict(imread(image_path))
    print("Model selected class '{0}' for image {1}".format(class_name, intermediate.description))


if __name__=="__main__":
    convert_keras_to_pmml(VGG16(), VGG_16_PATH, "VGG-16 ImageNet Model")
    convert_keras_to_pmml(VGG19(), VGG_19_PATH, "VGG-19 ImageNet Model")
    convert_keras_to_pmml(ResNet50(), RESNET50_PATH, "ResNet50 ImageNet Model")
    convert_keras_to_pmml(MobileNet(),                                  MOBILENET_PATH, "MobileNet ImageNet Model")
    #ERROR convert_keras_to_pmml(InceptionV3(input_shape=(299,299,3)),         INCEPTION_V3_PATH, "InceptionV3 Deep Neural Network Model")
    #ERROR convert_keras_to_pmml(InceptionResNetV2(input_shape=(299, 299, 3)), INCEPTION_RESNET_PATH, "InceptionResNetV2 Deep Neural Network Model")
    convert_keras_to_pmml(DenseNet121(),                                DENSENET121_PATH, "DenseNet121 ImageNet Model")
    convert_keras_to_pmml(DenseNet169(),                                DENSENET169_PATH, "DenseNet169 ImageNet Model")
    convert_keras_to_pmml(DenseNet201(),                                DENSENET201_PATH, "DenseNet201 ImageNet Model")
