"""
Read PMML files and make predictions

Example usage:
    python main.py predict \
        --model=examples/deepnetwork/VGG16/model.pmml \
        --input=test/assets/cat.jpg

    python main.py runserver \
        --model=examples/deepnetwork/VGG16/model.pmml

    python main.py build_torch_examples

    python main.py build_keras_examples

    python main.py validate
"""
import json
import glob
import argparse
from lxml import etree
from imageio import imread
from models.gpr.parser import GaussianProcessParser
from models.deepnetwork.core.intermediate import DeepNetwork
from models.deepnetwork.core.utils import strip_namespace



def load_pmml(filename):
    """
    Load a PMML file
    The model type is determined read from the PMML file
    """
    tree = etree.iterparse(filename)
    root = strip_namespace(tree).root

    config = {}
    config["filename"] = filename

    header = root.find("Header")
    if "description" in header.attrib:
        config["description"] = header.attrib["description"]
    if "copyright" in header.attrib:
        config["copyright"] = header.attrib["copyright"]

    dnn = root.find("DeepNetwork")
    gpr = root.find("GaussianProcessModel")

    if dnn is not None:
        model = DeepNetwork(**config)
        model.load_pmml(root)
    elif gpr is not None:
        parser = GaussianProcessParser()
        model = parser.parse(filename)
    else:
        raise ValueError("Could not find a valud model in %s"%filename)
    return model



def build_keras_examples():
    """
    Automatically build examples from publically available models
    """
    from models.deepnetwork import generate_keras_models
    generate_keras_models.build_models([
        "VGG_16",
        "VGG_19",
        "RESNET_50",
        "MOBILENET",
        "INCEPTION_V3",
        "INCEPTION_RESNET",
        "DENSENET_121",
        "DENSENET_169",
        "DENSENET_201"
    ])


def build_pytorch_examples():
    """
    Automatically build examples from publically available models
    """
    from models.deepnetwork import generate_torch_models
    generate_torch_models.build_models([
        "VGG_16",
        "VGG_19",
        "RESNET_50",
        #"MOBILENET",
        #"INCEPTION_V3",
        #"INCEPTION_RESNET",
        #"DENSENET_121",
        #"DENSENET_169",
        #"DENSENET_201"
    ])


def validate_models_using_schema(filename):
    """
    Validate a file against the schema
    Validates all models if a filename is not provided
    """
    model = DeepNetwork()
    if filename:
        filenames = [filename]
    else:
        keras_filenames = glob.glob("examples/deepnetwork/*.pmml")
        torch_filenames = glob.glob("examples/deepnetwork/*.pmml")
        filenames = keras_filenames + torch_filenames
    for filepath in filenames:
        print("Validating {0}".format(filepath))
        if model.validate_pmml(filepath):
            print("PMML File is VALID\n")
        else:
            print("PMML File is INVALID\n")
            model.read_pmml(filepath) # Force error


def predict(model, input_file):
    """
    Return a prediction from a model
    The input_file is either an image or a json file describing the input
    """
    if input_file.endswith(".json"):
        with open(input_file,"w") as fd:
            data = json.loads(input_file)
    else:
        data = imread(input_file)
    result = model.predict(data)
    print("Model predicted class: %s"%result)
    return result



parser = argparse.ArgumentParser(description='Main entry point for PMML package.')
parser.add_argument('--model', help='The PMML file to load')
parser.add_argument('--input', default='', help='The path to the input file for testing')
parser.add_argument('--runserver', default=False, help='Run a server')
subparsers = parser.add_subparsers(dest='operation', help='One of [predict, runserver, validate, ...]')

parser_validate = subparsers.add_parser('validate',
    help='Usage: main.py validate [--filename filename]')

parser_validate.add_argument('--filename', type=str,
    help='PMML file to validate')



if __name__=="__main__":
    args = parser.parse_args()

    if args.operation.lower()=="build_keras_examples":
        build_keras_examples()

    elif args.operation.lower()=="build_pytorch_examples":
        build_keras_examples()

    elif args.operation=="validate":
        validate_models_using_schema(args.filename)

    elif args.operation.lower()=="predict":
        model = load_pmml(args.model)
        prediction = predict(model, args.input)

    elif args.operation.lower()=="runserver":
        model = load_pmml(args.model)

    else:
        raise ValueError("Unknown operation %s"%args.operation)
