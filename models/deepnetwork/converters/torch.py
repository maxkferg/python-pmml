"""
Convert Torch models to PMML
"""
import torch.nn as nn
from torchvision.models.resnet import Bottleneck
from core.intermediate import DeepNetwork
from core.layers import InputLayer, Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dense, BatchNormalization, Dropout, Reshape, DepthwiseConv2D, Merge, Activation


def convert(torch_model, class_map, description="Neural Network Model"):
	"""
	Convert a torch model to PMML
	@model. The torch model object
	@class_map. A map in the form {class_id: class_name}
	@description. A short description of the model
	Returns a DeepNeuralNetwork object which can be exported to PMML
	"""
	pmml = DeepNetwork(description=description, class_map=class_map)
	pmml.torch_model = torch_model
	pmml.model_name = str(torch_model)
	first_layer = convert_input()
	last_layer = convert_recursive(torch_model, first_layer, pmml)



def convert_recursive(spec, previous_layer, model):
	for layer in spec.children():
		print(layer.__class__)
		# Go deeper into this parent layer
		if layer.__class__ in [nn.Sequential, Bottleneck]:
			previous_layer = convert_recursive(layer, previous_layer, model)
		# Convert this child layer
		elif layer.__class__ in CONVERSIONS:
			converter = CONVERSIONS[layer.__class__]
			previous_layer = converter(layer, previous_layer)
			model._append_layer(previous_layer)
		else:
			raise ValueError("Unknown layer type:", layer.__class__)

	return previous_layer




def convert_input():
	return InputLayer(
		name="Input Layer",
		input_size=(224,224,3)
	)


def convert_conv(spec, previous_layer):
	print(spec)
	print(dir(spec))
	print(spec.state_dict())
	return Conv2D(
		name=spec._get_name(),
		channels=spec.in_channels,
		kernel_size=spec.kernel_size,
		dilation_rate=spec.dilation,
		use_bias=True,
		activation="Relu",
		strides=spec.stride,
		padding=spec.padding,
		inbound_nodes=[previous_layer],
	)


def convert_depthwise_conv(spec, previous_layer):
	return DepthwiseConv2D(
		name=spec._get_name(),
		kernel_size=spec.kernel_size,
		depth_multiplier=spec.depth_multiplier,
		use_bias=spec.use_bias,
		activation=spec.activation,
		strides=spec.strides,
		padding=spec.padding,
		inbound_nodes=[previous_layer],
	)


def convert_max_pooling(spec, previous_layer):
	print(dir(spec))
	return MaxPooling2D(
		name=spec._get_name(),
		pool_size=spec.kernel_size,
		strides=spec.stride,
		inbound_nodes=[previous_layer],
	)


def convert_average_pooling(spec, previous_layer):
	return AveragePooling2D(
		name=spec._get_name(),
		pool_size=spec.kernel_size,
		strides=spec.stride,
		inbound_nodes=[previous_layer],
	)


def convert_global_average_pooling(spec, previous_layer):
	return GlobalAveragePooling2D(
		name=spec._get_name(),
		inbound_nodes=[previous_layer],
	)


def convert_faltten(spec, previous_layer):
	return Flatten(
		name=spec._get_name(),
		inbound_nodes=[previous_layer],
	)


def convert_dense(spec, previous_layer):
	return Dense(
		name=spec._get_name(),
		channels=spec.units,
		use_bias=spec.use_bias,
		activation=spec.activation,
		inbound_nodes=[previous_layer],
	)


def convert_zero_padding2d(spec, previous_layer):
	return ZeroPadding2D(
		name=spec._get_name(),
		padding=spec.padding,
		inbound_nodes=[previous_layer],
	)


def convert_reshape(spec, previous_layer):
	return Reshape(
		name=spec._get_name(),
		target_shape=spec.target_shape,
		inbound_nodes=[previous_layer],
	)


def convert_dropout(spec, previous_layer):
	return Dropout(
		name=spec._get_name(),
		inbound_nodes=[previous_layer],
	)


def convert_batch_norm(spec, previous_layer):
	print(dir(spec))
	return BatchNormalization(
		name=spec._get_name(),
		axis=0, #spec.axis,
		momentum=spec.momentum,
		epsilon=0,#spec.epsilon,
		center=0,#spec.center,
		inbound_nodes=[previous_layer],
	)


def convert_add(spec, previous_layer):
	return Merge(
		name=spec._get_name(),
		inbound_nodes=[previous_layer]
	)


def convert_substract(spec, previous_layer):
	return Merge(
		name=spec._get_name(),
		operator='subtract',
		inbound_nodes=[previous_layer]
	)


def convert_dot(spec, previous_layer):
	return Merge(
		name=spec._get_name(),
		operator='dot',
		inbound_nodes=[previous_layer]
	)


def convert_concatenate(spec, previous_layer):
	return Merge(
		name=spec._get_name(),
		axis=spec.axis,
		operator='concatenate',
		inbound_nodes=[previous_layer]
	)


def convert_activation(spec, previous_layer):
	return Activation(
		name=spec._get_name(),
		activation=spec.activation,
		inbound_nodes=[previous_layer],
	)


def convert_relu(spec, previous_layer):
	return Activation(
		name=spec._get_name(),
		activation='relu',
		threshold=spec.threshold,
		max_value=6, #spec.max_value,
		negative_slope=0, #spec.negative_slope,
		inbound_nodes=[previous_layer]
)



CONVERSIONS = {
	nn.Conv2d: convert_conv,
	nn.BatchNorm2d: convert_batch_norm,
	nn.ReLU: convert_relu,
	nn.AvgPool2d: convert_average_pooling,
	nn.MaxPool2d: convert_max_pooling,
	nn.Linear: convert_dense,
}