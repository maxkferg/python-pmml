import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim


def train_gdxray(net,
          dataset,
          epochs=10,
          batch_size=16,
          val_percent=0.05,
          save_cp=True,
          gpu=False,
          lr=1e-2,
          img_scale=0.5):

    #if args.load:
    #    net.load_state_dict(torch.load(args.load))
    #    print('Model loaded from {}'.format(args.load))

    if gpu:
        net.cuda()
    try:
        train_net(net=net,
                  dataset=dataset,
                  lr=lr,
                  gpu=False,
                  epochs=epochs,
                  batch_size=batch_size,
                  save_cp=save_cp,
                  val_percent=0.05,
                  img_scale=img_scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




def train_net(net, dataset, epochs, batch_size, lr, val_percent, save_cp, gpu, img_scale):

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(dataset), str(save_cp), str(gpu)))

    N_train = len(dataset['train'])


    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    train_loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)


    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        epoch_loss = 0

        for (inputs, labels) in train_loader:

            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            masks_pred = net(inputs)
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))
