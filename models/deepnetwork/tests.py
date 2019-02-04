import os
import json
import keras
import numpy as np
from pprint import pprint
from converters.keras import convert
from core.intermediate import DeepNetwork
from core.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from scipy.ndimage import imread
from deepdiff import DeepDiff

root = "../../examples/deepnetwork"

VGG_16_MODEL = os.path.join(root, "VGG16.pmml")
RESNET50_MODEL = os.path.join(root, "ResNet50.pmml")
MOBILENET_PATH = os.path.join(root, "MobileNet.pmml")


def diff_config(old_config, new_config, strict=False):
    nums = '0123456789'
    diff = DeepDiff(old_config, new_config)
    for key in ["values_changed"]:
        for changed_key in list(diff[key].keys()):
            before = diff[key][changed_key]["new_value"]
            after = diff[key][changed_key]["old_value"]
            if before=="Activation" and after=="ReLU":
                del diff[key][changed_key]
            if not strict and type(before) is str and type(after) is str and before.rstrip(nums) == after.rstrip(nums):
                del diff[key][changed_key]
    return diff


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
    pmml = convert(keras_model, class_map=class_map, description="VGG-16 ImageNet Model")
    pmml.save_pmml(VGG_16_MODEL)

    keras_model = ResNet50()
    pmml = convert(keras_model, class_map=class_map,  description="ResNet50 ImageNet Model")
    pmml.save_pmml(RESNET50_MODEL)

    keras_model = MobileNet()
    pmml = convert(keras_model, class_map=class_map,  description="MobileNet ImageNet Model")
    pmml.save_pmml(MOBILENET_PATH)


def test_pmml_to_intermediate():
    """
    Load an PMML file to an intermediate form
    """
    print("\n--- Test PMML to intermediate ---")
    intermediate = DeepNetwork(filename=VGG_16_MODEL)
    #assert(intermediate.description=="VGG-16 Deep Neural Network Model")
    #assert(len(intermediate.layers)==23)


def test_intermediate_to_keras(pmml_file, keras_model):
    """
    Test conversion between intermediate form and keras
    Compare the loaded PMML file to @keras_model
    """
    print("\n--- Test Intermediate to Keras (%s) ---"%pmml_file)
    intermediate = DeepNetwork(filename=pmml_file)
    new_keras_model = intermediate.get_keras_model()
    new_config = new_keras_model.get_config()
    old_config = keras_model.get_config()
    #pprint(diff_config(old_config, new_config))


def test_cat_classification(model):
    print("\n--- Test cat classification (%s) ---"%model)
    filename = "assets/cat.jpg"
    intermediate = DeepNetwork(filename=model)
    class_name = intermediate.predict(imread(filename))
    print("Model selected class '{0}' for image {1}".format(class_name, filename))



if __name__=="__main__":
    test_convert_keras_to_pmml()
    test_pmml_to_intermediate()
    test_intermediate_to_keras(VGG_16_MODEL, VGG16())
    test_intermediate_to_keras(MOBILENET_PATH, MobileNet())
    test_cat_classification(VGG_16_MODEL)
    test_cat_classification(RESNET50_MODEL)
    test_cat_classification(MOBILENET_PATH)


