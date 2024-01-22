from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from time import time
import numpy as np
import torch
from torch.autograd import Variable as V

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from networks.qmenet import QME_Net_cenet
from scripts.loss import  *
from scripts.data import ImageFolder
from scripts.quant import *
os.environ['CUDA_VISIBLE_DEVICES'] = "1"







def QME_Net_Train(net,flagQ, parameter_mask, old_weight_dict,  image_path,  save_path,  alpha, epoch, batchsize, lr, NUM_UPDATE_LR, INITAL_EPOCH_LOSS):

    print(epoch)
    NAME = 'QME-Net_' + image_path.split('/')[-1]
    no_optim = 0
    total_epoch = epoch
    train_epoch_best_loss = INITAL_EPOCH_LOSS
    batchsize = batchsize

    dataset = ImageFolder(root_path=image_path, datasets='DRIVE')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss = dice_bce_loss()
    diversityloss = diversity_Loss()

    tic = time()
    for epoch in range(1, total_epoch + 1):
        if flagQ:
            update_weight(net, old_weight_dict, parameter_mask)

        train_epoch_loss = 0
        train_epoch_segmentation_loss = 0
        train_epoch_diversity_loss = 0
        train_epoch_segmentation_loss1 = 0
        train_epoch_segmentation_loss2 = 0
        train_epoch_segmentation_loss3 = 0
        train_epoch_segmentation_loss0 = 0

        data_loader_iter = iter(data_loader)

        for img, mask in data_loader_iter:
            img = V(img.cuda(), volatile=False)
            mask = V(mask.cuda(), volatile=False)
            optimizer.zero_grad()
            pred1, pred2, pred3, pred\
                , n1e1, n2e1, n3e1 \
                , n1e2, n2e2, n3e2 \
                , n1e3, n2e3, n3e3 \
                , n1e4, n2e4, n3e4 \
                , n1e5, n2e5, n3e5 \
                , n1d1, n2d1, n3d1 \
                , n1d2, n2d2, n3d2 \
                , n1d3, n2d3, n3d3 \
                , n1d4, n2d4, n3d4 \
                = net.forward(img)

            segmentation_loss1 = loss(mask, pred1)
            segmentation_loss2 = loss(mask, pred2)
            segmentation_loss3 = loss(mask, pred3)
            segmentation_loss0 = loss(mask, pred)

            diversity_losse1 = diversityloss(n1e1, n2e1, n3e1)
            diversity_losse2 = diversityloss(n1e2, n2e2, n3e2)
            diversity_losse3 = diversityloss(n1e3, n2e3, n3e3)
            diversity_losse4 = diversityloss(n1e4, n2e4, n3e4)
            diversity_losse5 = diversityloss(n1e5, n2e5, n3e5)
            diversity_lossd1 = diversityloss(n1d1, n2d1, n3d1)
            diversity_lossd2 = diversityloss(n1d2, n2d2, n3d2)
            diversity_lossd3 = diversityloss(n1d3, n2d3, n3d3)
            diversity_lossd4 = diversityloss(n1d4, n2d4, n3d4)

            diversity_loss = diversity_losse1 + diversity_losse2 + diversity_losse3 + diversity_losse4+ \
                          diversity_lossd1 +diversity_lossd2 +diversity_lossd3 +diversity_lossd4

            segmentation_loss = segmentation_loss1 + segmentation_loss2 + segmentation_loss3 + segmentation_loss0

            train_loss = segmentation_loss - alpha*diversity_loss

            train_loss.backward()
            optimizer.step()


            train_epoch_loss += train_loss
            train_epoch_diversity_loss += (diversity_loss+ diversity_losse5)
            train_epoch_segmentation_loss += segmentation_loss
            train_epoch_segmentation_loss1 += segmentation_loss1
            train_epoch_segmentation_loss2 += segmentation_loss2
            train_epoch_segmentation_loss3 += segmentation_loss3
            train_epoch_segmentation_loss0 += segmentation_loss0



        train_epoch_loss = train_epoch_loss / len(data_loader_iter)
        train_epoch_diversity_loss = train_epoch_diversity_loss / len(data_loader_iter)
        train_epoch_segmentation_loss = train_epoch_segmentation_loss / len(data_loader_iter)
        train_epoch_segmentation_loss1 = train_epoch_segmentation_loss1 / len(data_loader_iter)
        train_epoch_segmentation_loss2 = train_epoch_segmentation_loss2 / len(data_loader_iter)
        train_epoch_segmentation_loss3 = train_epoch_segmentation_loss3 / len(data_loader_iter)
        train_epoch_segmentation_loss0 = train_epoch_segmentation_loss0 / len(data_loader_iter)


        print(' epoch: ', epoch, ' time:', int(time() - tic), ' loss1: ', round(train_epoch_segmentation_loss1.item(), 6), \
              ' loss2: ', round(train_epoch_segmentation_loss2.item(), 6), ' loss3: ', round(train_epoch_segmentation_loss3.item(), 6), \
              ' loss0: ', round(train_epoch_segmentation_loss0.item(), 6) , ' segmentation_loss: ', round(train_epoch_segmentation_loss.item(), 6),  ' Diversity: ',
              round(train_epoch_diversity_loss.item(), 6))


        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_segmentation_loss
            torch.save(net.state_dict(), save_path + NAME + '.th')
        # 
        if no_optim > NUM_UPDATE_LR:
            net.load_state_dict(torch.load(save_path + NAME + '.th'))
            lr = lr / 2

    print('Finish!')
    return net























if __name__ == '__main__':
    print(torch.__version__)


    parser = ArgumentParser(description="Training script for QME-Net models",formatter_class=ArgumentDefaultsHelpFormatter)


    parser.add_argument('--backbone', default='cenet', type=str, help='Neural network architecture')
    parser.add_argument('--input-size', default=448, type=int, help='Images input size')
    parser.add_argument('--image-path', default='./dataset/DRIVE', type=str, help='DIRVE dataset path')
    parser.add_argument('--save-path', default='weights/', type=str, help='store the trained model')
    parser.add_argument('--alpha', default=0.005, type=int, help='the empirical coefficient for diversity')
    parser.add_argument('--epoch', default=1200, type=int, help='Training epochs for full-precision stage')
    parser.add_argument('--epochQ', default=300, type=int, help='Training epochs for quantization stage')
    parser.add_argument('--batchsize', default=4, type=int, help='Batch per GPU')
    parser.add_argument('--lr', default=2e-4, type=int, help='learning rate')
    parser.add_argument('--update-lr', default=10, type=int, help='learning rate decay')
    parser.add_argument('--epochloss-init', default=10000, type=int, help='learning rate decay')
    parser.add_argument('--Q0', default=5, type=int, help='Quantization bits for meta learner B0')
    parser.add_argument('--Q1', default=5, type=int, help='Quantization bits for base learner B1')
    parser.add_argument('--Q2', default=5, type=int, help='Quantization bits for base learner B2')
    parser.add_argument('--Q3', default=5, type=int, help='Quantization bits for base learner B3')
    parser.add_argument('--quant', default=[0.3,0.3,0.2,0.2], nargs = '+', type = int, help=' ')

    args = parser.parse_args()

    net = QME_Net_cenet().cuda()


    print('--------------------------------------------------------------')
    print('--------------------Full-precision stage----------------------')
    print('--------------------------------------------------------------')

    flagQ = False
    net = QME_Net_Train(net, flagQ, [], [],   args.image_path,  args.save_path, args.alpha, args.epoch, args.batchsize, args.lr,args.update_lr, args.epochloss_init )

    # model_path = './weights/QME-Net_DRIVE.th'
    # net = loadweight(net, model_path)

    print('--------------------------------------------------------------')
    print('--------------------Quantization stage------------------------')
    print('--------------------------------------------------------------')
    flagQ = True

    parameter_mask = {}
    paremeter_dict = net.state_dict()
    model_dict = net.state_dict()
    for k, v in net.state_dict().items():
        model_dict[k] = v
        flag = np.ones(v.shape)
        flag = flag > 0
        parameter_mask[k] = flag


    print('----------------INQ 1--------------')
    old_parameter_mask, new_parameter_mask = two_sage_partition(net, parameter_mask, 1, args.quant)
    new_qunat_weight_dict = quant_weight_dict(args.backbone, net, old_parameter_mask, new_parameter_mask, args.Q1, args.Q2, args.Q3, args.Q0)
    net.load_state_dict(new_qunat_weight_dict)
    save_pathQ = args.save_path + 'NQ1/'
    net = QME_Net_Train(net, flagQ, new_parameter_mask, new_qunat_weight_dict,  args.image_path,  save_pathQ, args.alpha, args.epochQ, args.batchsize, args.lr,  args.update_lr, args.epochloss_init )

    print('----------------INQ 2--------------')
    old_parameter_mask, new_parameter_mask = two_sage_partition(net, new_parameter_mask, 2, args.quant)
    new_qunat_weight_dict = quant_weight_dict(args.backbone, net, old_parameter_mask, new_parameter_mask, args.Q1, args.Q2, args.Q3, args.Q0)
    net.load_state_dict(new_qunat_weight_dict)
    save_pathQ = args.save_path + 'NQ2/'
    net = QME_Net_Train(net, flagQ, new_parameter_mask, new_qunat_weight_dict,  args.image_path,  save_pathQ, args.alpha, args.epochQ, args.batchsize, args.lr,  args.update_lr, args.epochloss_init )

    print('----------------INQ 3--------------')
    old_parameter_mask, new_parameter_mask = two_sage_partition(net, new_parameter_mask, 3, args.quant)
    new_qunat_weight_dict = quant_weight_dict(args.backbone, net, old_parameter_mask, new_parameter_mask, args.Q1, args.Q2, args.Q3, args.Q0)
    net.load_state_dict(new_qunat_weight_dict)
    save_pathQ = args.save_path + 'NQ3/'
    net = QME_Net_Train(net, flagQ, new_parameter_mask, new_qunat_weight_dict, args.image_path,  save_pathQ, args.alpha, args.epochQ, args.batchsize, args.lr,  args.update_lr, args.epochloss_init )

    print('----------------INQ 4--------------')
    last_parameter_mask = {}
    for k, v in net.state_dict().items():
        flag = np.zeros(v.shape)
        flag = flag > 0
        last_parameter_mask[k] = flag

    new_qunat_weight_dict = quant_weight_dict(args.backbone, net, new_parameter_mask, last_parameter_mask, args.Q1, args.Q2, args.Q3, args.Q0)
    net.load_state_dict(new_qunat_weight_dict)
    save_pathQ = args.save_path + 'NQ4/'
    net = QME_Net_Train(net, flagQ, new_parameter_mask, new_qunat_weight_dict, args.image_path,  save_pathQ, args.alpha, args.epochQ, args.batchsize, args.lr,  args.update_lr, args.epochloss_init )














