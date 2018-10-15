"""
Intermediate representations of Neural Network Layers
Each layer can be converted to PMML or the corrosponding Keras layer
"""
import keras.layers as k
import lxml.etree as et
from lxml import etree
from utils import Array, to_bool
DEBUG = False


class Layer():

	def __init__(self,**kwarfs):
		self.name = name

	def to_pmml(self):
		"""
		Return a etree representation of this layer
		"""
		pass

	def to_keras(self, graph):
		"""
		Return the keras representation of this layer
		"""
		pass



class Flatten(Layer):
	"""
	A 2D Convolutional Layer
	"""

	def __init__(self,name=None):
		self.name = name


	def to_pmml(self):
		"""
		Return an elemenTree item corrosponding to this 
		"""
		layer =  et.Element("Layer", type="Flatten", name=self.name)
		return layer


	def to_keras(self, graph):
		config = {
			"name": self.name,
		}
		if DEBUG:
			print("Creating Flatten layer with config",config)
		prev_layer = graph["prev_layer"]
		return k.Flatten(**config)(prev_layer)


class Activation(Layer):
	"""
	An activation layer
	"""

	def __init__(self, activation="relu", name=None):
		self.name = name
		self.activation = activation

	def to_pmml(self):
		"""
		Return an elemenTree item corrosponding to this 
		"""
		attrib = {
			"name": self.name,
			"activation": self.activation
		}
		layer =  et.Element("Layer", type="Activation", attrib=attrib)
		return layer

	def to_keras(self, graph):
		config = {
			"name": self.name,
			"activation": self.activation
		}
		if DEBUG:
			print("Creating Activation layer with config",config)
		prev_layer = graph["prev_layer"]
		return k.Activation(**config)(prev_layer)


class Merge(Layer):
	"""
	An activation layer
	"""
	def __init__(self, operator="add", inbound_nodes=[], name=None):
		self.name = name
		self.operator = operator
		self.inbound_nodes = inbound_nodes
		operators = ["add","subtract","dot","concatenate"]
		if operator not in operators:
			raise ValueError("Unknown operator %s"%operator)
		if len(inbound_nodes) < 2:
			raise ValueError("Merge layers must have at least 2 inbound nodes")


	def to_pmml(self):
		"""
		Return an elemenTree item corrosponding to this 
		"""
		attrib = {
			"name": self.name,
			"operator": self.operator,
		}
		layer =  et.Element("Layer", type="Merge", attrib=attrib)

		# Strides Element with array Subelement
		inputs = etree.SubElement(layer, "Inputs")
		inputs.append(Array(children=self.inbound_nodes, dtype="string"))

		return layer


	def to_keras(self, graph):
		# Find the input tensors in the graph
		inbound_layers = []
		for inbound_node in self.inbound_nodes:
			if inbound_node not in graph:
				raise ValueError("Could not find layer %s in graph"%inbound_node)
			inbound_layers.append(graph[inbound_node])
		operator = self._get_keras_operator()
		if DEBUG:
			print("Creating Merge({}) layer with inbound {}".format(self.operator, inbound_layers))
		return operator(inbound_layers, name=self.name)


	def _get_keras_operator(self):
		"""
		Return the keras layer matching self.operator
		"""
		keras_map = {
			"add": k.add,
			"subtract": k.subtract,
			"multipy": k.multiply,
			"average": k.average,
			"concatenate": k.concatenate,
			"dot": k.dot,
		}
		return keras_map[self.operator]


 
class BatchNormalization(Layer):
	"""
	A BatchNormalization
	"""

	def __init__(self, axis=-1, momentum=0.99, epsilon=0.001, center=True, name=None):
		self.axis = axis
		self.momentum = momentum
		self.epsilon = epsilon
		self.center = center
		self.name = name

	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this 
		"""
		attrib = {
			"name": self.name,
			"momentum": str(self.momentum),
			"center": str(self.center),
			"axis": str(self.axis)
		}

		layer = etree.Element("Layer", type="BatchNormalization", attrib=attrib)
		return layer

	def to_keras(self, graph):
		"""
		Return the equivalent keras layer
		"""
		config = {
			"name": self.name,
			"momentum": self.momentum,
			"center": self.center,
			"axis": self.axis,
			"name": self.name
		}

		if DEBUG:
			print("Creating BatchNormalization layer with config:\n",config)
		prev_layer = graph["prev_layer"]
		return k.BatchNormalization(**config)(prev_layer)


class GlobalAveragePooling2D(Layer):
	"""
	A GlobalAveragePooling2D layer
	"""

	def __init__(self, name=None):
		self.name = name

	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this 
		"""
		layer = etree.Element("Layer", type="GlobalAveragePooling2D", name=self.name)
		return layer

	def to_keras(self, graph):
		"""
		Return the equivalent keras layer
		"""
		config = {}
		if DEBUG:
			print("Creating GlobalAveragePooling2D layer with config:\n",config)
		prev_layer = graph["prev_layer"]
		return k.GlobalAveragePooling2D(**config)(prev_layer)


class InputLayer(Layer):
	"""
	The first layer in any network
	"""

	def __init__(self, input_size, name=None):
		self.input_size = input_size
		self.name = name

	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this 
		"""
		layer = etree.Element("Layer", type="InputLayer", name=self.name)

		# PoolSize Element with array Subelement
		input_size = etree.SubElement(layer, "InputSize")
		input_size.append(Array(children=self.input_size)) 

		return layer

	def to_keras(self, graph):
		"""
		Return the equivalent keras layer
		"""
		config = {
			'name': self.name,
			'shape': self.input_size,
		}
		if DEBUG:
			print("Creating InputLayer layer with config:\n",config)
		return k.Input(**config)


class MaxPooling2D(Layer):
	"""
	A MaxPoolingLayer
	"""

	def __init__(self, pool_size=(3,3), strides=(1,1), name=None):
		self.pool_size = pool_size
		self.strides = strides
		self.name = name

	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this 
		"""
		layer = etree.Element("Layer", type="MaxPooling2D", name=self.name)

		# PoolSize Element with array Subelement
		pool_size = etree.SubElement(layer, "PoolSize")
		pool_size.append(Array(children=self.pool_size)) 

		# Strides Element with array Subelement
		strides = etree.SubElement(layer, "Strides")
		strides.append(Array(children=self.strides))

		return layer

	def to_keras(self, graph):
		"""
		Return the equivalent keras layer
		"""
		config = {
			'name': self.name,
			'strides': self.strides,
			'pool_size': self.pool_size,
		}
		if DEBUG:
			print("Creating MaxPooling2D layer with config:\n",config)
		prev_layer = graph["prev_layer"]
		return k.MaxPooling2D(**config)(prev_layer)



class AveragePooling2D(Layer):
	"""
	An AveragePooling2D layer
	"""

	def __init__(self, pool_size=(3,3), strides=(1,1), name=None):
		self.pool_size = pool_size
		self.strides = strides
		self.name = name

	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this 
		"""
		layer = etree.Element("Layer", type="AveragePooling2D", name=self.name)

		# PoolSize Element with array Subelement
		pool_size = etree.SubElement(layer, "PoolSize")
		pool_size.append(Array(children=self.pool_size)) 

		# Strides Element with array Subelement
		strides = etree.SubElement(layer, "Strides")
		strides.append(Array(children=self.strides))

		return layer

	def to_keras(self, graph):
		"""
		Return the equivalent keras layer
		"""
		config = {
			'name': self.name,
			'strides': self.strides,
			'pool_size': self.pool_size,
		}
		if DEBUG:
			print("Creating AveragePooling2D layer with config:\n",config)
		prev_layer = graph["prev_layer"]
		return k.AveragePooling2D(**config)(prev_layer)



class ZeroPadding2D(Layer):
	"""
	A zero padding layer
	"""
	def __init__(self, padding=(3,3), name=None):
		self.padding = padding
		self.name = name

	def to_pmml(self):
		"""
		Return an elemenTree item corrosponding to this 
		"""
		layer = etree.Element("Layer", type="ZeroPadding2D", name=self.name)

		# Padding size Element with array Subelement
		if type(self.padding) is int:
			pool_size = etree.SubElement(layer, "Padding")
			pool_size.append(Array(children=[self.padding]))
		elif type(self.padding[0]) is int:
			pool_size = etree.SubElement(layer, "Padding")
			pool_size.append(Array(children=[self.padding]))
		elif type(self.padding[0]) is tuple:
			children = self.padding[0] + self.padding[1]
			pool_size = etree.SubElement(layer, "Padding")
			pool_size.append(Array(children=children)) 
		else:
			raise ValueError("Unknown padding format"+str(self.padding))

		return layer

	def to_keras(self, graph, input_shape=None):
		"""
		Return the equivalent keras layer
		"""
		config = {
			'name': self.name,
			'padding': self.padding,
		}
		if input_shape is not None:
			config['input_shape'] = input_shape
		
		if DEBUG:
			print("Creating ZeroPadding2D layer with config:\n",config)
		
		prev_layer = graph["prev_layer"]
		return k.ZeroPadding2D(**config)(prev_layer)



class Conv2D(Layer):
	"""
	A 2D Convolutional Layer
	"""

	def __init__(self, channels, kernel_size, strides, padding, activation='relu', dilation_rate=1, use_bias=True, inbound_node=None, name=None):
		self.channels = channels
		self.kernel_size = kernel_size
		self.strides = strides
		self.activation = activation
		self.dilation_rate = dilation_rate
		self.padding = padding
		self.name = name
		self.inbound_node = inbound_node
		self.use_bias = to_bool(use_bias)

		# Enforce types
		if type(self.kernel_size) is list:
			self.kernel_size = tuple(self.kernel_size)

		if type(self.strides) is list:
			self.strides = tuple(self.strides)

		if type(self.padding) is list:
			self.padding = tuple(self.padding)

		if type(self.dilation_rate) is list:
			self.dilation_rate = tuple(self.dilation_rate)

	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this 
		"""
		attrib = {
			"type": "Conv2D",
			"activation": self.activation,
			"padding": self.padding,
			"use_bias": str(self.use_bias),
			"inbound_node": str(self.inbound_node)
		}
		if self.name is not None:
			attrib['name'] = self.name

		layer = et.Element("Layer", attrib)
		kernel = et.SubElement(layer, "ConvolutionalKernel",
			attrib={
				"channels": str(self.channels)
			})

		if type(self.dilation_rate) is int:
			self.dilation_rate = [self.dilation_rate]

		dilation_rate = etree.SubElement(kernel, "DilationRate")
		dilation_rate.append(Array(children=self.dilation_rate))

		# Generate <KernelSize><Array>2,2</Array></KernelSize>
		kernel_size = etree.SubElement(kernel, "KernelSize")
		kernel_size.append(Array(children=self.kernel_size))

		kernel_stride = etree.SubElement(kernel, "KernelStride")
		kernel_stride.append(Array(children=self.strides))

		return layer


	def to_keras(self, graph, input_shape=None):
		config = {
			"name": self.name,
			"filters": self.channels,
			"kernel_size": self.kernel_size,
			"strides": self.strides, 
			"padding": self.padding, 
			"activation": self.activation,
			"dilation_rate": self.dilation_rate, 
			"use_bias": to_bool(self.use_bias),
		}
		
		if input_shape is not None:
			config['input_shape'] = input_shape
		
		if DEBUG:
			print("Creating Conv2D layer with config:\n", config)

		prev_layer = graph["prev_layer"]
		if self.inbound_node is not None:
			prev_layer = graph[self.inbound_node]

		return k.Conv2D(**config)(prev_layer)



class Dense(Layer):
	"""
	A 2D Convolutional Layer
	"""

	def __init__(self, channels=128, activation='relu', use_bias=True, name=None):
		self.channels = int(channels)
		self.activation = activation
		self.use_bias = to_bool(use_bias)
		self.name = name


	def to_pmml(self):
		"""
		Return an elemenTree item corrosponding to this 
		"""
		layer =  etree.Element("Layer",
			attrib={
				"name": self.name,
				"type": "Dense",
				"channels": str(self.channels),
				"activation": str(self.activation),
			})
		return layer


	def to_keras(self,graph):
		"""
		Return the keras representation
		"""
		config = {
			"name": self.name,
			"units": self.channels,
			"activation": self.activation
		}
		if DEBUG:
			print("Creating Dense layer with config:\n", config)
		prev_layer = graph["prev_layer"]
		return k.Dense(**config)(prev_layer)


def get_layer_class_by_name(layer_type):
	type_map = {
		"InputLayer": InputLayer,
		"Conv2D": Conv2D,
		"Merge": Merge,
		"Activation": Activation,
		"Dense": Dense,
		"Flatten": Flatten,
		"BatchNormalization": BatchNormalization,
		"MaxPooling2D": MaxPooling2D,
		"ZeroPadding2D": ZeroPadding2D,
		"AveragePooling2D": AveragePooling2D,
		"GlobalAveragePooling2D": GlobalAveragePooling2D
	}
	return type_map[layer_type]
