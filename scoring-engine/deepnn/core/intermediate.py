"""
An intermediate representation of a DeepNeuralNetwork
Serves as an intermediate between PMML and DL frameworks like Keras
"""
import os
import keras
import datetime
import numpy as np
from lxml import etree
from . import layers
from .utils import read_array
from .layers import InputLayer, get_layer_class_by_name
DEBUG = False

class PMML_Model():

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

        if filename is not None:
            self.load_pmml(filename)

        if self.description is None:
            self.description = "Neural Network"

        if self.copyright is None:
            year = datetime.datetime.now().year
            self.copyright = "Copyright (c) {0} {1}".format(year,username)


    def load_pmml(self, filename):
        """
        Load the intermediate represenation from PMML
        """
        pass


    def generate_header(self):
        PMML_version = "4.5"
        xmlns = "http://www.dmg.org/PMML-4_5"
        PMML = etree.Element('PMML', xmlns=xmlns, version=PMML_version)
        header = etree.SubElement(PMML, "Header", copyright=self.copyright, description=self.description)
        return header


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


class DeepNeuralNetwork(PMML_Model):

    def load_pmml(self,filename):
        """
        Load the model from PMML
        """
        tree = etree.parse(filename)
        root = tree.getroot()
        DNN = root.find("DeepNeuralNetwork")
        layers = DNN.findall("Layer")
        if "description" in root.attrib:
            self.description = root.attrib["description"]
        if "copyright" in root.attrib["copyright"]:
            self.copyright = root.attrib["copyright"]
        for layer_element in layers:
            config = dict(layer_element.attrib)
            layer_type = config.pop('type')
            # Convert config into correct datatype
            if "momentum" in config:
                config["momentum"] = float(config["momentum"])
            if "axis" in config:
                config["axis"] = int(config["axis"])
            if "channels" in config:
                config["channels"] = int(config["channels"])
            # Read Attributes
            strides = layer_element.find("Strides")
            padding = layer_element.find("Padding")
            pool_size = layer_element.find("PoolSize")
            convolutional_kernel = layer_element.find("ConvolutionalKernel")
            inbound_nodes = layer_element.find("InboundNodes")
            input_size = layer_element.find("InputSize")
            if convolutional_kernel is not None:
                kernel_size = convolutional_kernel.find("KernelSize")
                kernel_strides = convolutional_kernel.find("KernelStride")
                dilation_rate = convolutional_kernel.find("DilationRate")
                config["channels"] = int(convolutional_kernel.attrib["channels"])
                config["kernel_size"] = read_array(kernel_size)
                config["strides"] = read_array(kernel_strides)
                config["dilation_rate"] = read_array(dilation_rate)
                if len(config["dilation_rate"])==1:
                    config["dilation_rate"] = config["dilation_rate"][0]
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
        elements = root.findall("./DataDictionary/DataField[@optype='categorical']/Value")
        for i,element in enumerate(elements):
            self.class_map[i] = element.attrib['value']

        # Load the weights file
        dirname = os.path.dirname(filename)
        weights_file = DNN.find("Weights").attrib['href']
        self.weights_file = os.path.join(dirname, weights_file)
        if not os.path.exists(self.weights_file):
            raise ValueError("No such file:",self.weights_file)


    def save_pmml(self, filename, username="NIST"):
        """
        Save the model to a PMML representation
        """
        PMML = self.generate_header()
        dictionary = self.generate_data_dictionary()
        PMML.append(dictionary)

        # DeepNeuralNetworkLevel
        DNN = etree.SubElement(PMML,"DeepNeuralNetwork")
        DNN.set("modelname","Deep Neural Network")
        DNN.set("functionname","regression")
        for layer in self.layers:
            DNN.append(layer.to_pmml())

        # Save the weights file, and add a link in the PMML
        model_dir = os.path.dirname(filename)
        weights_file = os.path.join(model_dir,"weights.h5")
        self.keras_model.save_weights(weights_file)
        relpath = os.path.relpath(weights_file, model_dir)
        etree.SubElement(DNN, "Weights", href=relpath, encoding="hdf5")

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
            if type(layer) is InputLayer:
                graph["input_layer"] = keras_tensor

        # Build the keras model
        inputs = graph["input_layer"]
        outputs = graph["prev_layer"]
        keras_model = keras.models.Model(inputs=inputs, outputs=outputs)
        print("Completed building keras model: %s"%self.description)

        if DEBUG:
            print(keras_model.summary())

        print("Loading weights from %s... "%self.weights_file, end="", flush=True)
        keras_model.load_weights(self.weights_file, by_name=True)
        print("Done")

        return keras_model





