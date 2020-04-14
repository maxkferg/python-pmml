"""
python tests.py gdxray train --dataset=~/data/GDXray/Castings --backbone=vgg16
python tests.py gdxray eval --dataset=~/data/GDXray/Castings
"""
import os
import argparse
from segmentation_models import Unet
from tests.gdxray.train import train_gdxray
from tests.gdxray.eval import eval_gdxray
from tests.gdxray.dataloader import KerasDataset
from models.deepnetwork.converters.keras import convert
from models.deepnetwork.core.intermediate import DeepNetwork


parser = argparse.ArgumentParser(description='Process some integers.')

subparsers = parser.add_subparsers(
	help='Run GDXray tests',
	dest='command')

parser_validate = subparsers.add_parser('validate',
	help='Usage: test.py validate [--filename filename]')

parser_validate.add_argument('--filename', type=str,
                    help='PMML file to validate')

parser_gdxray = subparsers.add_parser('gdxray',
	help='Usage: test.py gdxray <train|eval>')

parser_gdxray.add_argument('operation', type=str,
	help='train or eval')

parser_gdxray.add_argument('--dataset', type=str,
                    help='Location of the GDXRay dataset')

parser_gdxray.add_argument('--backbone', type=str,
					default='resnet34',
                    help='Location of the GDXRay dataset')



BACKBONES = [
	"vgg16",
	"vgg19",
	"resnet18",
	"resnet34",
	"resnet50",
	"resnext101",
	"densenet121",
	"densenet169",
	"senet154",
	"mobilenet",
	"inceptionv3",
	"efficientnetb0",
]


def callback(model, backbone):
	"""Save the model to PMML"""
	#pmml = convert_keras_to_pmml(model)
	class_map = {}
	weights_path = "examples/deepnetwork/weights/UNet-{0}.h5".format(backbone)
	output_path = "examples/deepnetwork/UNet-{0}.pmml".format(backbone)
	description = "UNet-{0} model trained to classify casting defects".format(backbone)
	pmml = convert(model, class_map=class_map, description=description)
	pmml.save_pmml(output_path, weights_path=weights_path, save_weights=True)



def test_gdxray_train(args):
	"""
	Test that a GDXRay model can be trained and saved to PMML
	"""
	print("GDXRay train")
	backbone = args.backbone.lower()
	model = Unet(backbone, classes=2, input_shape=(384, 384, 3), encoder_weights=None, activation='softmax')
	train_dataset = KerasDataset(os.path.expanduser(args.dataset))
	save_callback = lambda model: callback(model, backbone)
	train_gdxray(model, train_dataset, save_callback=save_callback)



def test_gdxray_eval(args):
	"""
	Test that a GDXray model can be loaded from PMML and evaluated on images
	"""
	print("GDXRay eval")
	eval_gdxray(model, dataset)




if __name__=="__main__":
	args = parser.parse_args()
	if args.command=='gdxray':
		if args.operation=="train":
			test_gdxray_train(args)
		elif args.operation=="eval":
			test_gdxray_eval(args)
	elif args.command is None:
		print("No tests to run")
	else:
		print("Unknown test:", args.command)
