"""
Convert Keras models to PMML
"""
import keras
from intermediate import DeepNeuralNetwork
from layers import InputLayer, Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dense, BatchNormalization, Merge, Activation


def get_inbound_nodes(layer_inbound_nodes):
	return [node[0] for node in layer_inbound_nodes[0]]


def convert(keras_model, description="Neural Network Model"):
	"""
	Convert a keras model to PMML
	@model. The keras model object
	@description. A short description of the model
	Returns a DeepNeuralNetwork object which can be exported to PMML
	"""
	pmml = DeepNeuralNetwork(description=description)
	pmml.keras_model = keras_model
	config = keras_model.get_config()

	for layer in config['layers']:
		layer_class = layer['class_name']
		layer_config = layer['config']
		layer_inbound_nodes = layer['inbound_nodes']
		# Input
		if layer_class is "InputLayer":
			pmml._append_layer(InputLayer(
				name=layer_config['name'],
				input_size=layer_config['batch_input_shape'][1:]
			))
		# Conv2D
		elif layer_class is "Conv2D":
			pmml._append_layer(Conv2D(
				name=layer_config['name'],
				channels=layer_config['filters'],
				kernel_size=layer_config['kernel_size'],
				dilation_rate=layer_config['dilation_rate'],
				use_bias=layer_config['use_bias'],
				activation=layer_config['activation'],
				strides=layer_config['strides'],
				padding=layer_config['padding'],
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes),
			))
		# MaxPooling
		elif layer_class is "MaxPooling2D":
			pmml._append_layer(MaxPooling2D(
				name=layer_config['name'],
				pool_size=layer_config['pool_size'],
				strides=layer_config['strides'],
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes),
			))
		elif layer_class is "AveragePooling2D":
			pmml._append_layer(AveragePooling2D(
				name=layer_config['name'],
				pool_size=layer_config['pool_size'],
				strides=layer_config['strides'],
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes),
			))
		elif layer_class is "GlobalAveragePooling2D":
			pmml._append_layer(GlobalAveragePooling2D(
				name=layer_config['name'],
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes),
			))
		# Flatten
		elif layer_class is "Flatten":
			pmml._append_layer(Flatten(
				name=layer_config['name'],
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes),
			))
		# Dense
		elif layer_class is "Dense":
			pmml._append_layer(Dense(
				name=layer_config['name'],
				channels=layer_config['units'],
				use_bias=layer_config['use_bias'],
				activation=layer_config['activation'],
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes),
			))
		# Zero padding layer
		elif layer_class is "ZeroPadding2D":
			pmml._append_layer(ZeroPadding2D(
				name=layer_config['name'],
				padding=layer_config['padding'],
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes),
			))
		# Batch Normalization
		elif layer_class is "BatchNormalization":
			pmml._append_layer(BatchNormalization(
				name=layer_config['name'],
				axis=layer_config['axis'],
				momentum=layer_config['momentum'],
				epsilon=layer_config['epsilon'],
				center=layer_config['center'],
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes),
			))
		elif layer_class is "Add":
			pmml._append_layer(Merge(
				name=layer_config['name'],
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes)
			))
		elif layer_class is "Subtract":
			pmml._append_layer(Merge(
				name=layer_config['name'],
				operator='subtract',
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes)
			))
		elif layer_class is "Dot":
			pmml._append_layer(Merge(
				name=layer_config['name'],
				operator='dot',
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes)
			))
		elif layer_class is "Concatenate":
			pmml._append_layer(Merge(
				name=layer_config['name'],
				operator='concatenate',
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes)
			))
		elif layer_class is "Activation":
			pmml._append_layer(Activation(
				name=layer_config['name'],
				activation=layer_config['activation'],
				inbound_nodes=get_inbound_nodes(layer_inbound_nodes),
			))
		# Unknown layer
		else:
			raise ValueError("Unknown layer type:",layer_class)
	return pmml
