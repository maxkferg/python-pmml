"""
python tests.py gdxray train --dataset=~/data/GDXray/Castings
python tests.py gdxray eval --dataset=~/data/GDXray/Castings
"""
import os
import argparse
from segmentation_models import Unet
from tests.gdxray.train import train_gdxray
from tests.gdxray.eval import eval_gdxray
from tests.gdxray.dataloader import KerasDataset
from models.deepnetwork.converters.keras import convert


parser = argparse.ArgumentParser(description='Process some integers.')

subparsers = parser.add_subparsers(help='Run GDXray tests')

parser_gdxray = subparsers.add_parser('gdxray',
	help='Usage: test.py gdxray <train|eval>')

parser_gdxray.add_argument('operation', type=str,
	help='train or eval')

parser_gdxray.add_argument('--dataset', type=str,
                    help='Location of the GDXRay dataset')



def save_callback(model):
	"""Save the model to PMML"""
	#pmml = convert_keras_to_pmml(model)
	class_map = {}
	weights_path = "examples/deepnetwork/UNet.h5"
	output_path = "examples/deepnetwork/UNet.pmml"
	description = "UNet model trained to classify casting defects"
	pmml = convert(model, class_map=class_map, description=description)
	pmml.save_pmml(output_path, weights_path=weights_path, save_weights=False)



def test_gdxray_train(args):
	"""
	Test that a GDXRay model can be trained and saved to PMML
	"""
	print("GDXRay train")
	model = Unet('resnet34', classes=2, input_shape=(384, 384, 3), encoder_weights=None, activation='softmax')
	train_dataset = KerasDataset(os.path.expanduser(args.dataset))
	train_gdxray(model, train_dataset, save_callback=save_callback)



def test_gdxray_eval(args):
	"""
	Test that a GDXray model can be loaded from PMML and evaluated on images
	"""
	print("GDXRay eval")
	eval_gdxray(model, dataset)


if __name__=="__main__":
	args = parser.parse_args()
	if args.operation=="train":
		test_gdxray_train(args)
	elif args.operation=="eval":
		test_gdxray_eval(args)


