"""
Read PMML files and make predictions 

Example usage:
    python pmml.py predict \
        --model=examples/deepnetwork/VGG16/model.pmml \
        --input=test/assets/cat.jpg

    python pmml.py runserver \
        --model=examples/deepnetwork/VGG16/model.pmml

    python pmml.py build_examples
"""
import json
import argparse
from lxml import etree
from scipy.misc import imread
from models.gpr.parser import GaussianProcessParser
from models.deepnetwork.core.intermediate import DeepNetwork 
from models.deepnetwork.core.utils import strip_namespace 
from models.deepnetwork.generate_models import build_models



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



def build_examples():
    """
    Automatically build examples from publically available models
    """
    build_models([
        "VGG_16",
        "VGG_19",
        "RESNET_50",
        "MOBILENET",
        #"INCEPTION_V3",
        #"INCEPTION_RESNET",
        #"DENSENET_121",
        #"DENSENET_169",
        #"DENSENET_201"])
        ])



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
parser.add_argument('operation', type=str, help='One of [predict, runserver, or build_examples]')
parser.add_argument('--model', help='The PMML file to load')
parser.add_argument('--input', default='', help='The path to the input file for testing')
parser.add_argument('--runserver', default=False, help='Run a server')



if __name__=="__main__":
    args = parser.parse_args()
    if args.operation.lower()=="build_examples":
        build_examples()

    elif args.operation.lower()=="predict":
        model = load_pmml(args.model)
        prediction = predict(model, args.input)

    elif args.operation.lower()=="runserver":
        model = load_pmml(args.model)

    else:
        raise ValueError("Unknown operation %s"%args.operation)
