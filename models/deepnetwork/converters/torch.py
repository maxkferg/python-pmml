"""
Convert Torch models to PMML
"""
import torch
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
	layers = serialize_layers(torch_model,'')
	# Attach the first layer
	previous_layer = convert_input() 
	pmml._append_layer(previous_layer)
	# Attach the following layers
	flattened = False
	for identifier, layer in layers:
		print(layer.__class__)
		if layer.__class__ in CONVERSIONS:
			if layer.__class__ == nn.Linear and not flattened:
				previous_layer = convert_flatten(previous_layer)
				pmml._append_layer(previous_layer)
				flattened = True
			converter = CONVERSIONS[layer.__class__]
			previous_layer = converter(identifier, layer, previous_layer)
			pmml._append_layer(previous_layer)
		elif layer.__class__ == Bottleneck:
			pmml, previous_layer = convert_bottleneck(pmml, identifier, layer, previous_layer)
		else:
			pass
			# raise ValueError("Unknown layer type:", layer.__class__)
	return pmml



# If Bottleneck:
# If downsample? or not
# No downsample: x > conv1 > bn1 > relu > conv2 > bn2 > relu > conv3 > bn3 > "merge(x,prev)" > relu
# Yes downsample: x > conv1 > bn1 > relu > conv2 > bn2 > relu > conv3 > bn3 > "downsample(x)" > "merge(x,prev)" > relu
def convert_bottleneck(pmml, identifier, bottleneck, previous_layer):
	x = previous_layer
	conv1 = convert_conv('{0}{1}_{2}'.format(identifier,0,'bottleneck_conv1'), bottleneck.conv1, previous_layer)
	bn1 = convert_batch_norm('{0}{1}_{2}'.format(identifier,1,'bottleneck_bn1'), bottleneck.bn1, conv1)
	relu1 = convert_relu('{0}{1}_{2}'.format(identifier,2,'bottleneck_relu1'), bottleneck.relu, bn1)
	conv2 = convert_conv('{0}{1}_{2}'.format(identifier,3,'bottleneck_conv2'), bottleneck.conv2, relu1)
	bn2 = convert_batch_norm('{0}{1}_{2}'.format(identifier,4,'bottleneck_bn2'), bottleneck.bn2, conv2)
	relu2 = convert_relu('{0}{1}_{2}'.format(identifier,5,'bottleneck_relu2'), bottleneck.relu, bn2)
	conv3 = convert_conv('{0}{1}_{2}'.format(identifier,6,'bottleneck_conv3'), bottleneck.conv3, relu2)
	bn3 = convert_batch_norm('{0}{1}_{2}'.format(identifier,7,'bottleneck_bn3'), bottleneck.bn3, conv3)
	pmml._append_layer(conv1)
	pmml._append_layer(bn1)
	pmml._append_layer(relu1)
	pmml._append_layer(conv2)
	pmml._append_layer(bn2)
	pmml._append_layer(relu2)
	pmml._append_layer(conv3)
	pmml._append_layer(bn3)
	if bottleneck.downsample:
		downsample_conv = convert_conv('{0}{1}_{2}'.format(identifier,8,'bottleneck_downsample_conv'), bottleneck.downsample[0], x)
		downsample_bn = convert_batch_norm('{0}{1}_{2}'.format(identifier,9,'bottleneck_downsample_bn'), bottleneck.downsample[1], downsample_conv)
		merge = convert_add('{0}{1}_{2}'.format(identifier,10,'bottleneck_merge'), [bn3, downsample_bn])
		relu3 = convert_relu('{0}{1}_{2}'.format(identifier,11,'bottleneck_relu3'), bottleneck.relu, merge)
		pmml._append_layer(downsample_conv)
		pmml._append_layer(downsample_bn)
		pmml._append_layer(merge)
		pmml._append_layer(relu3)
	else:
		merge = convert_add('{0}{1}_{2}'.format(identifier,8,'bottleneck_merge'), [bn3, x])
		relu3 = convert_relu('{0}{1}_{2}'.format(identifier,9,'bottleneck_relu3'), bottleneck.relu, merge)
		pmml._append_layer(merge)
		pmml._append_layer(relu3)
	return pmml, relu3



def serialize_layers(torch_model,prefix):
	layers = []
	for i,l in enumerate(torch_model.children()):
		if l.__class__ in [nn.Sequential]:
			layers += serialize_layers(l,'{0}_'.format(i))
		elif l.__class__ in [Bottleneck]:
			layers.append(('{0}{1}_'.format(prefix,i),l))
		else:
			layers.append(('{0}{1}_{2}'.format(prefix,i,l.__class__.__name__),l))
	return layers



# def convert_recursive(spec, previous_layer, model):
# 	i = 0
# 	for layer in spec.children():
# 		if i == 1:
# 			continue
# 		i += 1
# 		print(layer.__class__)
# 		# Go deeper into this parent layer
# 		if layer.__class__ in [nn.Sequential, Bottleneck]:
# 			previous_layer = convert_recursive(layer, previous_layer, model)
# 		# Convert this child layer
# 		elif layer.__class__ in CONVERSIONS:
# 			converter = CONVERSIONS[layer.__class__]
# 			previous_layer = converter(layer, previous_layer)
# 			model._append_layer(previous_layer)
# 		else:
# 			raise ValueError("Unknown layer type:", layer.__class__)

# 	return previous_layer

	# last_layer = convert_recursive(torch_model, first_layer, pmml)



# def convert_recursive(spec, previous_layer, model):
# 	i = 0
# 	for layer in spec.children():
# 		if i == 1:
# 			continue
# 		i += 1
# 		print(layer.__class__)
# 		# Go deeper into this parent layer
# 		if layer.__class__ in [nn.Sequential, Bottleneck]:
# 			previous_layer = convert_recursive(layer, previous_layer, model)
# 		# Convert this child layer
# 		elif layer.__class__ in CONVERSIONS:
# 			converter = CONVERSIONS[layer.__class__]
# 			previous_layer = converter(layer, previous_layer)
# 			model._append_layer(previous_layer)
# 		else:
# 			raise ValueError("Unknown layer type:", layer.__class__)

# 	return previous_layer




def convert_input():
	return InputLayer(
		name="input_1",
		input_size=(224,224,3)
	)


def convert_conv(identifier, spec, previous_layer):
	return Conv2D(
		name=identifier,
		channels=spec.out_channels,
		kernel_size=spec.kernel_size,
		dilation_rate=spec.dilation,
		use_bias=True,
		activation="linear",
		strides=spec.stride,
		padding='same' if spec.padding==(0,0) else 'valid',
		inbound_nodes=[previous_layer.name],
	)


# def convert_depthwise_conv(spec, previous_layer):
# 	return DepthwiseConv2D(
# 		name=spec._get_name(),
# 		kernel_size=spec.kernel_size,
# 		depth_multiplier=spec.depth_multiplier,
# 		use_bias=spec.use_bias,
# 		activation=spec.activation,
# 		strides=spec.strides,
# 		padding=spec.padding,
# 		inbound_nodes=[previous_layer],
# 	)


def convert_max_pooling(identifier, spec, previous_layer):
	return MaxPooling2D(
		name=identifier,
		pool_size=(spec.kernel_size, spec.kernel_size),
		strides=(spec.stride, spec.stride),
		inbound_nodes=[previous_layer.name],
	)


def convert_average_pooling(identifier, spec, previous_layer):
	return AveragePooling2D(
		name=identifier,
		pool_size=(spec.kernel_size, spec.kernel_size),
		strides=(spec.stride, spec.stride),
		inbound_nodes=[previous_layer.name],
	)


# def convert_global_average_pooling(spec, previous_layer):
# 	return GlobalAveragePooling2D(
# 		name=spec._get_name(),
# 		inbound_nodes=[previous_layer],
# 	)


def convert_flatten(previous_layer):
	return Flatten(
		name='flatten',
		inbound_nodes=[previous_layer.name],
	)


def convert_dense(identifier, spec, previous_layer):
	return Dense(
		name=identifier,
		channels=spec.out_features,
		use_bias=True,
		activation='linear',
		inbound_nodes=[previous_layer.name],
	)


# def convert_zero_padding2d(spec, previous_layer):
# 	return ZeroPadding2D(
# 		name=spec._get_name(),
# 		padding=spec.padding,
# 		inbound_nodes=[previous_layer],
# 	)


# def convert_reshape(spec, previous_layer):
# 	return Reshape(
# 		name=spec._get_name(),
# 		target_shape=spec.target_shape,
# 		inbound_nodes=[previous_layer],
# 	)


# def convert_dropout(spec, previous_layer):
# 	return Dropout(
# 		name=spec._get_name(),
# 		inbound_nodes=[previous_layer],
# 	)


def convert_batch_norm(identifier, spec, previous_layer):
	return BatchNormalization(
		name=identifier,
		axis=0, #spec.axis,
		momentum=spec.momentum,
		epsilon=0, #spec.epsilon,
		center=0,#spec.center,
		inbound_nodes=[previous_layer.name],
	)


def convert_add(identifier, previous_layer):
	return Merge(
		name=identifier,
		inbound_nodes=[l.name for l in previous_layer]
	)


# def convert_substract(spec, previous_layer):
# 	return Merge(
# 		name=spec._get_name(),
# 		operator='subtract',
# 		inbound_nodes=[previous_layer]
# 	)


# def convert_dot(spec, previous_layer):
# 	return Merge(
# 		name=spec._get_name(),
# 		operator='dot',
# 		inbound_nodes=[previous_layer]
# 	)


# def convert_concatenate(spec, previous_layer):
# 	return Merge(
# 		name=spec._get_name(),
# 		axis=spec.axis,
# 		operator='concatenate',
# 		inbound_nodes=[previous_layer]
# 	)


# def convert_activation(spec, previous_layer):
# 	return Activation(
# 		name=spec._get_name(),
# 		activation=spec.activation,
# 		inbound_nodes=[previous_layer],
# 	)


def convert_relu(identifier, spec, previous_layer):
	return Activation(
		name=identifier,
		activation='relu',
		threshold=spec.threshold,
		negative_slope=0,
		inbound_nodes=[previous_layer.name]
)



CONVERSIONS = {
	nn.Conv2d: convert_conv,
	nn.BatchNorm2d: convert_batch_norm,
	nn.ReLU: convert_relu,
	nn.AvgPool2d: convert_average_pooling,
	nn.MaxPool2d: convert_max_pooling,
	nn.Linear: convert_dense,
}