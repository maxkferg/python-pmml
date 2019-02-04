import os
import json
import keras
import numpy as np
from pprint import pprint
from core.intermediate import DeepNetwork
from core.layers import Conv2D, MaxPooling2D, Flatten, Dense
from converters.keras import convert
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from scipy.ndimage import imread

root = "../../examples/deepnetwork/"


output_paths = {
    "VGG_16": os.path.join(root, "VGG16.pmml"),
    "VGG_19": os.path.join(root, "VGG19.pmml"),
    "RESNET_50": os.path.join(root, "ResNet50.pmml"),
    "MOBILENET": os.path.join(root, "MobileNet.pmml"),
    "INCEPTION_V3": os.path.join(root, "InceptionResNetV2.pmml"),
    "INCEPTION_RESNET": os.path.join(root, "InceptionV3.pmml"),
    "DENSENET_121": os.path.join(root, "DenseNet121.pmml"),
    "DENSENET_169": os.path.join(root, "DenseNet169.pmml"),
    "DENSENET_201": os.path.join(root, "DenseNet201.pmml")
}

weight_urls = {
    "VGG_16": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/VGG16.h5",
    "VGG_19": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/VGG19.h5",
    "RESNET_50": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/ResNet50.h5",
    "MOBILENET": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/MobileNet.h5",
    "INCEPTION_V3": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/InceptionResNetV2.h5",
    "INCEPTION_RESNET": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/InceptionV3.h5",
    "DENSENET_121": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/DenseNet121.h5",
    "DENSENET_169": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/DenseNet169.h5",
    "DENSENET_201": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/DenseNet201.h5"
}

keras_models = {
    "VGG_16": VGG16,
    "VGG_19": VGG19,
    "RESNET_50": ResNet50,
    "MOBILENET": MobileNet,
    "INCEPTION_V3": InceptionV3,
    "INCEPTION_RESNET": InceptionV3,
    "DENSENET_121": DenseNet121,
    "DENSENET_169": DenseNet169,
    "DENSENET_201": DenseNet201,
}

descriptions = {
    "VGG_16": "VGG-16 ImageNet Model",
    "VGG_19": "VGG-19 ImageNet Model",
    "RESNET_50": "ResNet-50 ImageNet Model",
    "MOBILENET": "MobileNet ImageNet Model",
    "INCEPTION_V3": "Inception V3 ImageNet Model",
    "INCEPTION_RESNET": "Inception ResNet V2 ImageNet Model",
    "DENSENET_121": "DenseNet121 ImageNet Model",
    "DENSENET_169": "DenseNet169 ImageNet Model",
    "DENSENET_201": "DenseNet201 ImageNet Model",
}


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


def convert_keras_to_pmml(keras_model, output_path, weights_path, description, debug=True):
    """
    Convert a keras model to PMML
    """
    print("\nGenerating model: %s"%description)
    class_map = load_imagenet_classes()
    pmml = convert(keras_model, class_map=class_map, description=description)
    pmml.save_pmml(output_path, weights_path=weights_path, save_weights=False)
    if debug:
        print("Checking model: %s"%output_path)
        intermediate = DeepNetwork(filename=output_path)
        old_config = keras_model.get_config()
        new_config = intermediate.get_keras_model().get_config()
        directory = os.path.dirname(output_path)
        dump_config(old_config, new_config, directory)
        test_prediction(output_path, "assets/cat.jpg")


def test_prediction(output_path, image_path):
    intermediate = DeepNetwork(filename=output_path)
    class_name = intermediate.predict(imread(image_path))
    print("Model selected class '{0}' for image {1}".format(class_name, intermediate.description))


def build_models(models):
    """
    Build all models listed in @models
    As an example build_models(["VGG_16", "VGG_19"])
    """
    for model_name in models:
        model = keras_models[model_name]()
        description = descriptions[model_name]
        weights_path = weight_urls[model_name]
        output_path = output_paths[model_name]
        convert_keras_to_pmml(model, output_path, weights_path, description)



if __name__=="__main__":
    build_models(["RESNET_50"])