import sys
import os
import numpy as np
from keras import optimizers
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score



def train_gdxray(model,
          train_dataset,
          val_dataset=None,
          epochs=10,
          batch_size=16,
          val_percent=0.05,
          save_cp=True,
          gpu=False,
          lr=1e-2,
          img_scale=0.5,
          save_callback=None):

    #if args.load:
    #    net.load_state_dict(torch.load(args.load))
    #    print('Model loaded from {}'.format(args.load))

    # Parameters
    params = {'batch_size': 64,
              'shuffle': True}

    # All parameter gradients will be clipped to
    # a maximum norm of 1.
    sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

    # Compile SGD optimizer
    model.compile(optimizer=sgd, loss="binary_crossentropy")

    # Generators
    if val_dataset is None:
        val_dataset = train_dataset

    if save_callback is not None:
        save_callback(model)

    # Train model on dataset
    model.fit_generator(
                      generator=train_dataset,
                      validation_data=val_dataset,
                      use_multiprocessing=True,
                      epochs=1,
                      workers=6)

