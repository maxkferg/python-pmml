"""
python tests.py gdxray train --dataset=~/data/GDXray/Castings
python tests.py gdxray eval --dataset=~/data/GDXray/Castings
"""
import os
import argparse
from tests.gdxray.train import train_gdxray
from tests.gdxray.eval import eval_gdxray
from tests.gdxray.dataset import GDXrayDataset
from models.deepnetwork.converters.torch import convert
import segmentation_models_pytorch as smp



parser = argparse.ArgumentParser(description='Process some integers.')

subparsers = parser.add_subparsers(help='Run GDXray tests')

parser_gdxray = subparsers.add_parser('gdxray',
	help='Usage: test.py gdxray <train|eval>')

parser_gdxray.add_argument('operation', type=str,
	help='train or eval')

parser_gdxray.add_argument('--dataset', type=str,
                    help='Location of the GDXRay dataset')



OUTPUT_PATH = "examples/deepnetwork/torch/UNet.pmml"
WEIGHTS_PATH = "examples/deepnetwork/torch/UNet.pmml"

def save_pmml(net):
	print("Saving net")
	pmml = convert(net,{})
	pmml.save_pmml(OUTPUT_PATH, weights_path=WEIGHTS_PATH, save_weights=False)


def test_gdxray_train(args):
	"""
	Test that a GDXRay model can be trained and saved to PMML
	"""
	model = smp.Unet('resnet34', encoder_weights='imagenet', classes=2)
	dataset = GDXrayDataset(os.path.expanduser(args.dataset))
	train_gdxray(model, dataset, save_pmml)


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


