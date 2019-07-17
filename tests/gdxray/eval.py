import segmentation_models_pytorch as smp




def eval_gdxray():
	model = smp.Unet()
	dataloader = "dataloader"