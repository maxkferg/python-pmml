"""
An intermediate representation of a DeepNeuralNetwork
Serves as an intermediate between PMML and DL frameworks like Keras
"""
import re
import os
import keras
import urllib
import numpy as np
from datetime import datetime
from lxml import etree
from . import layers
from .utils import read_array, to_bool, strip_namespace, url_exists
from .layers import InputLayer, get_layer_class_by_name
DEBUG = False


class PMML_Model():
    ns = "{http://www.dmg.org/PMML-4_5}"

    def __init__(self, filename=None, class_map={}, description=None, copyright=None, username="NIST"):
        """
        Create a new DeepNeuralNetwork
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
            print(filename)
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


class DeepNetwork(PMML_Model):

    def load_pmml(self, root_element):
        """
        Load the model from PMML
        """
        dnn_element = root_element.find("DeepNetwork")
        if dnn_element.tag != "DeepNetwork":
            raise ValueError("Element must have tag type DeepNetwork. Got %s"%dnn_element.tag)
        layers = dnn_element.findall("NetworkLayer")
        for layer_element in layers:
            config = dict(layer_element.attrib)
            layer_type = config.pop('layerType')
            # Convert config into correct datatype
            if "momentum" in config:
                config["momentum"] = float(config["momentum"])
            if "epsilon" in config:
                config["epsilon"] = float(config["epsilon"])
            if "axis" in config:
                config["axis"] = int(config["axis"])
            if "channels" in config:
                config["channels"] = int(config["channels"])
            if "depth_multiplier" in config:
                config["depth_multiplier"] = int(config["depth_multiplier"])
            if "center" in config:
                config["center"] = to_bool(config["center"])   
            if "threshold" in config:
                config["threshold"] = float(config["threshold"])   
            if "max_value" in config:
                config["max_value"] = float(config["max_value"])   
            if "negative_slope" in config:
                config["negative_slope"] = float(config["negative_slope"])                
            # Read Attributes
            strides = layer_element.find("Strides")
            padding = layer_element.find("Padding")
            pool_size = layer_element.find("PoolSize")
            convolutional_kernel = layer_element.find("ConvolutionalKernel")
            inbound_nodes = layer_element.find("InboundNodes")
            input_size = layer_element.find("InputSize")
            target_shape = layer_element.find("TargetShape")
            
            if convolutional_kernel is not None:
                kernel_size = convolutional_kernel.find("KernelSize")
                kernel_strides = convolutional_kernel.find("KernelStride")
                dilation_rate = convolutional_kernel.find("DilationRate")
                config["kernel_size"] = read_array(kernel_size)
                config["strides"] = read_array(kernel_strides)
                if dilation_rate is not None:
                    config["dilation_rate"] = read_array(dilation_rate)
                    if len(config["dilation_rate"])==1:
                        config["dilation_rate"] = config["dilation_rate"][0]
                if "channels" in convolutional_kernel.attrib:
                    config["channels"] = int(convolutional_kernel.attrib["channels"])
            if target_shape is not None:
                config["target_shape"] = read_array(target_shape)
            if input_size is not None:
                config["input_size"] = read_array(input_size)
            if pool_size is not None:
                config["pool_size"] = read_array(pool_size)
            if strides is not None:
                config["strides"] = read_array(strides)
            if inbound_nodes is not None:
                config["inbound_nodes"] = read_array(inbound_nodes)
            if padding is not None:
                array = read_array(padding)
                if len(array)==1:
                    config["padding"] = array[0]
                elif len(array)==2:
                    config["padding"] = tuple(array)
                elif len(array)==4:
                    config["padding"] = (tuple(array[0:2]),tuple(array[2:4]))
            # Enforce some sanity checks
            if layer_type=="Conv2D":
                assert("kernel_size" in config)
                assert("strides" in config)
            if layer_type=="ZeroPadding2D":
                assert("padding" in config)
            if layer_type!="InputLayer":
                if "inbound_nodes" not in config:
                    raise ValueError("Layer type %s requires argument inbound_nodes"%layer_type)
            # Create the intermediate class representation
            layer_class = get_layer_class_by_name(layer_type)
            new_layer = layer_class(**config)
            self.layers.append(new_layer)
        # Load the categorical output variables
        elements = root_element.findall("./DataDictionary/DataField[@optype='categorical']/Value")
        for i,element in enumerate(elements):
            self.class_map[i] = element.attrib['value']

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

        # Layer
        for layer in self.layers:
            dnn.append(layer.to_pmml())

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


    def predict(self, input_img):
        """
        Generate prediction using PMML representation
        """
        if self.keras_model is None:
            self.keras_model = self.get_keras_model()
        batch = input_img[None,:]
        scores = self.keras_model.predict(batch)
        class_id = np.argmax(scores)
        class_name = self.class_map[class_id]
        return class_name


    def generate_mining_schema(self):
        """Generate the data dictionary which describes the input"""
        schema = etree.Element("MiningSchema")
        etree.SubElement(schema, "MiningField", name="image", usageType="active")
        etree.SubElement(schema, "MiningField", name="class", usageType="predicted")
        return schema


    def _append_layer(self,layer):
        """
        Append a layer to the neural network
        """
        self.layers.append(layer)


    def get_keras_model(self):
        """
        Return the network as a keras model which can be trained
        or used for scoring
        """
        graph = {}
        for i, layer in enumerate(self.layers):
            keras_tensor = layer.to_keras(graph)
            graph[layer.name] = keras_tensor
            graph["prev_layer"] = keras_tensor
            if keras_tensor is None:
                raise ValueError("%s layer returned None"%layer.name)
            if type(layer) is InputLayer:
                graph["input_layer"] = keras_tensor

        # Build the keras model
        inputs = graph["input_layer"]
        outputs = graph["prev_layer"]
        keras_model = keras.models.Model(inputs=inputs, outputs=outputs)
        print("Completed building keras model: %s"%self.description)

        if DEBUG:
            print(keras_model.summary())

        if self.weights_file.lower().startswith('http'):
            local_cache = self.get_default_weights_file()
            if os.path.exists(local_cache):
                self.weights_file = local_cache
            else:
                print("Downloading weights from %s"%self.weights_file)
                localfile = urllib.URLopener()
                localfile.retrieve(self.weights_file, local_cache)

        print("Loading weights from %s... "%self.weights_file, end="", flush=True)
        keras_model.load_weights(self.weights_file)
        print("Done")

        return keras_model





