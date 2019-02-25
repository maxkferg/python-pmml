"""
An intermediate representation of a DeepNeuralNetwork
Serves as an intermediate between PMML and DL frameworks like Keras
"""
import re
import os
import onnx
import warnings
import numpy as np
import tensorflow as tf
import urllib.request
from lxml import etree
from datetime import datetime
from . import layers
from .utils import read_array, to_bool, strip_namespace, url_exists
from .layers import InputLayer, get_layer_class_by_name
from onnx_tf.backend import prepare

DEBUG = False


class ONNX_Model():
    ns = "{http://www.dmg.org/PMML-4_5}"

    def __init__(self, filename=None, class_map={}, description=None, copyright=None, username="NIST"):
        """
        Create a new PMML Model
        Optionally load layers from a PMML file
        @class_map: A map in the form {class_id: class_name}
        """
        self.layers = []
        self.description = description
        self.copyright = copyright
        self.keras_model = None
        self.weights_file = None
        self.class_map = class_map
        self.filename = filename

        if filename is not None:
            tree = etree.iterparse(filename)
            tree = strip_namespace(tree)
            self.load_metadata(tree.root)
            self.load_pmml(tree.root)

        if self.description is None:
            self.description = "Neural Network"

        if self.copyright is None:
            year = datetime.now().year
            self.copyright = "Copyright (c) {0} {1}".format(year,username)


    def load_metadata(self, root_element):
        """
        Load copyright information etc
        """
        header = root_element.find("Header")
        if "description" in header.attrib:
            self.description = header.attrib["description"]
        if "copyright" in header.attrib:
            self.copyright = header.attrib["copyright"]


    def get_default_weights_file(self):
        """
        Return the absolute path to the default weights file
        """
        return re.sub('.pmml$', '.h5', self.filename)


    def generate_root_tag(self):
        PMML_version = "4.5"
        xmlns = "http://www.dmg.org/PMML-4_5"
        PMML = etree.Element('PMML', xmlns=xmlns, version=PMML_version)
        header = etree.SubElement(PMML, "Header", copyright=self.copyright, description=self.description)
        timestamp = etree.SubElement(header, "Timestamp")
        timestamp.text = datetime.now().strftime("%Y-%M-%d %X")
        return PMML


    def generate_data_dictionary(self):
        """Generate the data dictionary which describes the input"""
        attrib = {'numberOfFields': str(1+len(self.class_map))}
        dictionary = etree.Element("DataDictionary", attrib=attrib)
        image = etree.SubElement(dictionary, "DataField", dataType="image", name="I", height="300", width="300", channels="3")
        # Add the categorical output variables
        categorical = etree.SubElement(dictionary, "DataField", dataType="string", name="class", optype="categorical")
        for class_id in sorted(self.class_map.keys()):
            etree.SubElement(categorical, "Value", value=self.class_map[class_id])
        return dictionary


    def generate_mining_schema(self):
        """Generate the data dictionary which describes the input"""
        schema = etree.Element("MiningSchema")
        etree.SubElement(schema, "MiningField", name="image", usageType="active")
        etree.SubElement(schema, "MiningField", name="class", usageType="predicted")
        return schema




class ONNX_Model(PMML_Model):


    def load_pmml(self, root_element):
        """
        Load the model from PMML
        """
        dnn_element = root_element.find("ONNX_Model")
        if dnn_element.tag != "ONNX_Model":
            raise ValueError("Element must have tag type ONNX_Model. Got %s"%dnn_element.tag)
        layers = dnn_element.findall("NetworkLayer")

        # Load the weights file (Exactly as it is in PMML)
        weights_file = dnn_element.find("Weights").attrib['href']
        if type(weights_file) is not str:
            raise ValueError("Expected weights file to be a string. Got %s"%type(weights_file))
        if weights_file.lower().startswith("http"):
            self.weights_file = weights_file
            if not url_exists(self.weights_file):
                raise ValueError("No weights file at url:", self.weights_file)
        else:
            # Weights files are relative to the current PMML location
            dirname = os.path.dirname(self.filename)
            self.weights_file = os.path.join(dirname, weights_file)
            if not os.path.exists(self.weights_file):
                raise ValueError("No such file:", self.weights_file)

        # Load the ONNX model file
     	if self.weights_file.lower().startswith('http'):
            local_cache = self.get_default_weights_file()
            if os.path.exists(local_cache):
                self.weights_file = local_cache
            else:
                print("Downloading weights from %s"%self.weights_file)
                urllib.request.urlretrieve(self.weights_file, local_cache)
                self.weights_file = local_cache

        print("Loading weights from %s... "%self.weights_file, end="", flush=True)
        model = onnx.load(self.weights_file) # Load the ONNX file
        onnx_model.load_weights(self.weights_file)
        print("Done")
        return onnx_model


    def save_pmml(self, filename, username="NIST", weights_path=None, save_weights=True):
        """
        Save the model to a PMML representation
        @weights_path (optional): Absolute path or url to weights file.
        @save_weights (optional): If True, weights will be saved to @weights file
        """
        self.filename = filename

        PMML = self.generate_root_tag()
        dictionary = self.generate_data_dictionary()
        PMML.append(dictionary)

        # DeepNeuralNetworkLevel
        dnn = etree.SubElement(PMML, "DeepNetwork")
        dnn.set("modelName", "Deep Neural Network")
        dnn.set("functionName", "classification")
        dnn.set("numberOfLayers", str(len(self.layers)))

        # MiningSchema
        schema = self.generate_mining_schema()
        dnn.append(schema)

        # Outputs
        outputs = etree.SubElement(dnn, "Outputs")
        etree.SubElement(outputs, "OutputField", dataType="string", feature="topClass")

        # Save the weights to a file
        if save_weights:
            if weights_path is not None and weights_path.lower().startswith("http"):
                raise ValueError("Can not save weights to url: %s"%weights_path)
            if weights_path is None:
                # Automatically generate the weights path based on the model filename
                weights_path = self.get_default_weights_file()
            self.keras_model.save_weights(weights_path)

        # Generate the link to the weights file
        if weights_path.lower().startswith("http"):
            relative_weights_path = weights_path
        else:
            model_dir = os.path.dirname(filename)
            relative_weights_path = os.path.relpath(weights_path, model_dir)
        etree.SubElement(dnn, "Weights", href=relative_weights_path, encoding="hdf5")

        # Write to file
        tree = etree.ElementTree(PMML)
        tree.write(filename, pretty_print=True, xml_declaration=True, encoding="utf-8")
        print('Wrote PMML file to %s'%filename)


    def predict(self, input_img, tpu_worker=None):
        """
        Generate prediction using PMML representation
        """
        if self.keras_model is None:
            self.keras_model = self.get_keras_model(tpu_worker=tpu_worker)
        batch = np.stack(1*[input_img])
        scores = self.keras_model.predict(batch)
        class_id = np.argmax(scores)
        class_name = self.class_map[class_id]
        return class_name

