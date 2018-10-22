"""
Intermediate representations of Neural Network Layers
Each layer can be converted to PMML or the corrosponding Keras layer
"""
import keras.layers as k
import lxml.etree as et
from lxml import etree
from .utils import Array, to_bool
DEBUG = False


class Layer():

	def __init__(self,inbound_nodes,**kwargs):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
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

	def _get_inbound_nodes_element(self):
		"""
		Return an Element representing the inbound nodes
		"""
		inbound_nodes = etree.Element("InboundNodes")
		inbound_nodes.append(Array(children=self.inbound_nodes, dtype="string"))
		return inbound_nodes

	def _get_inbound_nodes_from_graph(self, graph):
		"""
		Return the inbound nodes for this layer
		The return value is a list of tensors with the same number of elements as self.inbound_nodes
		Raise KeyError if the inbound node does not exist in the graph
		"""
		inbound_nodes = []
		for node in self.inbound_nodes:
			if node not in graph:
				raise KeyError("Could not find node %s in graph"%inbound_node)
			inbound_nodes.append(graph[node])
		return inbound_nodes



class Flatten(Layer):
	"""
	A 2D Convolutional Layer
	"""

	def __init__(self, inbound_nodes, name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
		self.name = name


	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this
		"""
		layer =  et.Element("NetworkLayer", layerType="Flatten", name=self.name)
		layer.append(self._get_inbound_nodes_element())
		return layer


	def to_keras(self, graph):
		config = {
			"name": self.name,
		}
		if DEBUG:
			print("Creating Flatten layer with config",config)
		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.Flatten(**config)(inbound_node)


class Activation(Layer):
	"""
	An activation layer
	"""

	def __init__(self, inbound_nodes, activation="relu", threshold=None, max_value=None, negative_slope=None, name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
		self.name = name
		self.activation = activation
		self.threshold = threshold
		self.max_value = max_value
		self.negative_slope = negative_slope

	def to_pmml(self):
		"""
		Return an elemenTree item corrosponding to this
		"""
		attrib = {
			"name": self.name,
			"activation": self.activation
		}

		if self.threshold is not None:
			attrib["threshold"] = str(self.threshold)
		if self.max_value is not None:
			attrib["max_value"] = str(self.max_value)
		if self.negative_slope is not None:
			attrib["negative_slope"] = str(self.negative_slope)

		layer = et.Element("NetworkLayer", layerType="Activation", attrib=attrib)
		layer.append(self._get_inbound_nodes_element())
		return layer

	def to_keras(self, graph):
		config = {
			"name": self.name,
			"activation": self.activation
		}
		# Decide whether to use Relu layer
		use_relu = False

		if self.threshold is not None:
			config["threshold"] = self.threshold
			use_relu = True
		if self.max_value is not None:
			config["max_value"] = self.max_value
			use_relu = True
		if self.negative_slope is not None:
			config["negative_slope"] = self.negative_slope
			use_relu = True

		if DEBUG:
			print("Creating Activation layer with config",config)
		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]

		if self.activation.lower()=='relu' and use_relu:
			config.pop('activation')
			return k.ReLU(**config)(inbound_node)
		return k.Activation(**config)(inbound_node)


class Merge(Layer):
	"""
	An activation layer
	"""
	def __init__(self, inbound_nodes, operator="add", axis=None, name=None):
		self.name = name
		self.operator = operator
		self.axis = axis
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
		if self.axis is not None:
			attrib["axis"] = str(self.axis)
		layer =  et.Element("NetworkLayer", layerType="Merge", attrib=attrib)
		layer.append(self._get_inbound_nodes_element())
		return layer


	def to_keras(self, graph):
		# Find the input tensors in the graph
		operator = self._get_keras_operator()
		if DEBUG:
			print("Creating Merge({}) layer with inbound {}".format(self.operator, inbound_layers))
		inbound_nodes = self._get_inbound_nodes_from_graph(graph)
		if self.axis is not None:
			return operator(inbound_nodes, axis=self.axis, name=self.name)	
		return operator(inbound_nodes, name=self.name)


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

	def __init__(self, inbound_nodes, axis=-1, momentum=0.99, epsilon=0.001, center=True, name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
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
			"epsilon": str(self.epsilon),
			"center": str(self.center),
			"axis": str(self.axis)
		}

		layer = etree.Element("NetworkLayer", layerType="BatchNormalization", attrib=attrib)
		layer.append(self._get_inbound_nodes_element())
		return layer

	def to_keras(self, graph):
		"""
		Return the equivalent keras layer
		"""
		config = {
			"name": self.name,
			"momentum": self.momentum,
			"epsilon": self.epsilon,
			"center": self.center,
			"axis": self.axis,
			"name": self.name
		}

		if DEBUG:
			print("Creating BatchNormalization layer with config:\n",config)
		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.BatchNormalization(**config)(inbound_node)


class GlobalAveragePooling2D(Layer):
	"""
	A GlobalAveragePooling2D layer
	"""

	def __init__(self, inbound_nodes, name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
		self.name = name

	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this
		"""
		layer = etree.Element("NetworkLayer", layerType="GlobalAveragePooling2D", name=self.name)
		layer.append(self._get_inbound_nodes_element())
		return layer

	def to_keras(self, graph):
		"""
		Return the equivalent keras layer
		"""
		config = {
			'name': self.name,
		}

		if DEBUG:
			print("Creating GlobalAveragePooling2D layer with config:\n",config)
		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.GlobalAveragePooling2D(**config)(inbound_node)


class InputLayer(Layer):
	"""
	The first layer in any network
	"""

	def __init__(self, input_size, name=None):
		self.input_size = input_size
		self.name = name

		if self.input_size[0] is None:
			raise ValueError("Model must specify input size")

	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this
		"""
		layer = etree.Element("NetworkLayer", layerType="InputLayer", name=self.name)

		# PoolSize Element with array Subelement
		input_size = etree.SubElement(layer, "InputSize")
		input_size.append(Array(children=self.input_size))

		return layer

	def to_keras(self, graph, batch_size=None):
		"""
		Return the equivalent keras layer
		"""
		batch_shape = [batch_size] + self.input_size
		config = {
			'name': self.name,
			'batch_shape': batch_shape,
		}
		if DEBUG:
			print("Creating InputLayer layer with config:\n",config)
		return k.Input(**config)


class MaxPooling2D(Layer):
	"""
	A MaxPoolingLayer
	"""

	def __init__(self, inbound_nodes, pool_size=(3,3), strides=(1,1), name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
		self.pool_size = pool_size
		self.strides = strides
		self.name = name

	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this
		"""
		layer = etree.Element("NetworkLayer", layerType="MaxPooling2D", name=self.name)
		layer.append(self._get_inbound_nodes_element())

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
		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.MaxPooling2D(**config)(inbound_node)



class AveragePooling2D(Layer):
	"""
	An AveragePooling2D layer
	"""

	def __init__(self, inbound_nodes, pool_size=(3,3), strides=(1,1), name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
		self.pool_size = pool_size
		self.strides = strides
		self.name = name

	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this
		"""
		layer = etree.Element("NetworkLayer", layerType="AveragePooling2D", name=self.name)
		layer.append(self._get_inbound_nodes_element())

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
		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.AveragePooling2D(**config)(inbound_node)



class ZeroPadding2D(Layer):
	"""
	A zero padding layer
	"""
	def __init__(self, inbound_nodes, padding=(3,3), name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
		self.padding = padding
		self.name = name

	def to_pmml(self):
		"""
		Return an elemenTree item corrosponding to this
		"""
		layer = etree.Element("NetworkLayer", layerType="ZeroPadding2D", name=self.name)
		layer.append(self._get_inbound_nodes_element())

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

		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.ZeroPadding2D(**config)(inbound_node)


class Reshape(Layer):
	"""
	A reshape layer
	"""
	def __init__(self, inbound_nodes, target_shape, name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
		self.target_shape = target_shape
		self.name = name

	def to_pmml(self):
		"""
		Return an elemenTree item corrosponding to this
		"""
		layer = etree.Element("NetworkLayer", layerType="Reshape", name=self.name)
		layer.append(self._get_inbound_nodes_element())

		# Padding size Element with array Subelement
		target_shape = etree.SubElement(layer, "TargetShape")
		target_shape.append(Array(children=self.target_shape))

		return layer

	def to_keras(self, graph, input_shape=None):
		"""
		Return the equivalent keras layer
		"""
		config = {
			'name': self.name,
			'target_shape': self.target_shape,
		}
		if input_shape is not None:
			config['input_shape'] = input_shape

		if DEBUG:
			print("Creating ZeroPadding2D layer with config:\n",config)

		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.Reshape(**config)(inbound_node)



class Conv2D(Layer):
	"""
	A 2D Convolutional Layer
	"""

	def __init__(self, inbound_nodes, channels, kernel_size, strides, padding, activation='relu', dilation_rate=1, use_bias=True, name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
		self.channels = channels
		self.kernel_size = kernel_size
		self.strides = strides
		self.activation = activation
		self.dilation_rate = dilation_rate
		self.padding = padding
		self.name = name
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
			"layerType": "Conv2D",
			"activation": self.activation,
			"padding": self.padding,
			"use_bias": str(self.use_bias),
		}
		if self.name is not None:
			attrib['name'] = self.name

		layer = et.Element("NetworkLayer", attrib)
		layer.append(self._get_inbound_nodes_element())

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

		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.Conv2D(**config)(inbound_node)


class DepthwiseConv2D(Layer):
	"""
	A 2D DepthwiseConv2D Layer
	"""

	def __init__(self, inbound_nodes, kernel_size, strides, padding, depth_multiplier, activation='relu', use_bias=True, name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
		self.kernel_size = kernel_size
		self.strides = strides
		self.activation = activation
		self.depth_multiplier = depth_multiplier
		self.padding = padding
		self.name = name
		self.use_bias = to_bool(use_bias)

		# Enforce types
		if type(self.kernel_size) is list:
			self.kernel_size = tuple(self.kernel_size)

		if type(self.strides) is list:
			self.strides = tuple(self.strides)

		if type(self.padding) is list:
			self.padding = tuple(self.padding)


	def to_pmml(self):
		"""
		Return an elementTree item corrosponding to this
		"""
		attrib = {
			"layerType": "DepthwiseConv2D",
			"activation": self.activation,
			"padding": self.padding,
			"use_bias": str(self.use_bias),
			"depth_multiplier": str(self.depth_multiplier)
		}
		if self.name is not None:
			attrib['name'] = self.name

		layer = et.Element("NetworkLayer", attrib)
		layer.append(self._get_inbound_nodes_element())

		kernel = et.SubElement(layer, "ConvolutionalKernel")
		kernel_size = etree.SubElement(kernel, "KernelSize")
		kernel_size.append(Array(children=self.kernel_size))
		kernel_stride = etree.SubElement(kernel, "KernelStride")
		kernel_stride.append(Array(children=self.strides))

		return layer


	def to_keras(self, graph, input_shape=None):
		config = {
			"name": self.name,
			"kernel_size": self.kernel_size,
			"strides": self.strides,
			"padding": self.padding,
			"activation": self.activation,
			"depth_multiplier": self.depth_multiplier,
			"use_bias": to_bool(self.use_bias),
		}

		if input_shape is not None:
			config['input_shape'] = input_shape

		if DEBUG:
			print("Creating DepthwiseConv2D layer with config:\n", config)

		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.DepthwiseConv2D(**config)(inbound_node)




class Dense(Layer):
	"""
	A Dense Layer
	"""

	def __init__(self, inbound_nodes, channels=128, activation='relu', use_bias=True, name=None):
		assert(len(inbound_nodes)==1)
		self.inbound_nodes = inbound_nodes
		self.channels = int(channels)
		self.activation = activation
		self.use_bias = to_bool(use_bias)
		self.name = name


	def to_pmml(self):
		"""
		Return an elemenTree item corrosponding to this
		"""
		attrib = {
				"name": self.name,
				"layerType": "Dense",
				"channels": str(self.channels),
				"activation": str(self.activation),
		}
		layer =  etree.Element("NetworkLayer", attrib=attrib)
		layer.append(self._get_inbound_nodes_element())

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
		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.Dense(**config)(inbound_node)


class Dropout(Layer):
	"""
	A Dropout Layer
	"""

	def __init__(self, inbound_nodes, name=None):
		self.inbound_nodes = inbound_nodes
		self.name = name
		self.rate = 0.001


	def to_pmml(self):
		"""
		Return an elemenTree item corrosponding to this
		"""
		attrib = {
				"name": self.name,
				"layerType": "Dropout",
		}
		layer =  etree.Element("NetworkLayer", attrib=attrib)
		layer.append(self._get_inbound_nodes_element())

		return layer


	def to_keras(self,graph):
		"""
		Return the keras representation
		"""
		config = {
			"rate": self.rate,
			"name": self.name,
		}
		if DEBUG:
			print("Creating Dropout layer with config:\n", config)
		inbound_node = self._get_inbound_nodes_from_graph(graph)[0]
		return k.Dropout(**config)(inbound_node)



def get_layer_class_by_name(layer_type):
	type_map = {
		"InputLayer": InputLayer,
		"Conv2D": Conv2D,
		"DepthwiseConv2D": DepthwiseConv2D,
		"Merge": Merge,
		"Activation": Activation,
		"Dropout": Dropout,
		"Dense": Dense,
		"Reshape": Reshape,
		"Flatten": Flatten,
		"BatchNormalization": BatchNormalization,
		"MaxPooling2D": MaxPooling2D,
		"ZeroPadding2D": ZeroPadding2D,
		"AveragePooling2D": AveragePooling2D,
		"GlobalAveragePooling2D": GlobalAveragePooling2D
	}
	return type_map[layer_type]
