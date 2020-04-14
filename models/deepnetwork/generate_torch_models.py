import os
import json
import keras
import numpy as np
from pprint import pprint
from os.path import join, dirname, abspath
from .core.intermediate import DeepNetwork
from .core.layers import Conv2D, MaxPooling2D, Flatten, Dense
from .converters.torch import convert
import torchvision.models as models
from scipy.ndimage import imread


root_dir = dirname(dirname(dirname(os.path.realpath(__file__))))
example_dir = abspath(join(root_dir, "examples/deepnetwork/"))

output_paths = {
    "VGG_16": os.path.join(example_dir, "torch/VGG16.pmml"),
    "VGG_19": os.path.join(example_dir,  "torch/VGG19.pmml"),
    "RESNET_50": os.path.join(example_dir,  "torch/ResNet50.pmml"),
    # "INCEPTION_V3": os.path.join(example_dir, "torch/InceptionV3.pmml"),
    # "DENSENET_121": os.path.join(example_dir, "torch/DenseNet121.pmml"),
    # "DENSENET_169": os.path.join(example_dir, "torch/DenseNet169.pmml"),
    # "DENSENET_201": os.path.join(example_dir, "torch/DenseNet201.pmml")
}

weight_urls = {
    "VGG_16": "https://s3.amazonaws.com/stanford-machine-learning/torch/pmml-models/VGG16.h5",
    "VGG_19": "https://s3.amazonaws.com/stanford-machine-learning/torch/pmml-models/VGG19.h5",
    "RESNET_50": "https://s3.amazonaws.com/stanford-machine-learning/torch/pmml-models/ResNet50.h5",
    # "DENSENET_121": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/torch/DenseNet121.h5",
    # "DENSENET_169": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/torch/DenseNet169.h5",
    # "DENSENET_201": "https://s3.amazonaws.com/stanford-machine-learning/pmml-models/torch/DenseNet201.h5"
}

torch_models = {
    "VGG_16": models.vgg16,
    "VGG_19": models.vgg19,
    "RESNET_50": models.resnet50,
    # "INCEPTION_V3": models.inception_v3,
    # "DENSENET_121": models.densenet121,
    # "DENSENET_169": models.densenet169,
    # "DENSENET_201": models.densenet201,
}

descriptions = {
    "VGG_16": "VGG-16 ImageNet Model (from torchvision)",
    "VGG_19": "VGG-19 ImageNet Model (from torchvision)",
    "RESNET_50": "ResNet-50 ImageNet Model (from torchvision)",
    # "INCEPTION_V3": "Inception V3 ImageNet Model (from torchvision)",
    # "DENSENET_121": "DenseNet121 ImageNet Model (from torchvision)",
    # "DENSENET_169": "DenseNet169 ImageNet Model (from torchvision)",
    # "DENSENET_201": "DenseNet201 ImageNet Model (from torchvision)",
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


def convert_torch_to_pmml(torch_model, output_path, weights_path, description, debug=True):
    """
    Convert a torch model to PMML
    """
    print("\nGenerating model: %s"%description)
    class_map = load_imagenet_classes()
    pmml = convert(torch_model, class_map=class_map, description=description)
    pmml.save_pmml(output_path, weights_path=weights_path, save_weights=False)
    if debug:
        print("Checking model: %s"%output_path)
        intermediate = DeepNetwork(filename=output_path)
        old_config = torch_model.get_config()
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
        model = torch_models[model_name](pretrained=True)
        description = descriptions[model_name]
        weights_path = weight_urls[model_name]
        output_path = output_paths[model_name]
        convert_torch_to_pmml(model, output_path, weights_path, description, debug=False)



if __name__=="__main__":
    build_models(["RESNET_50"])


