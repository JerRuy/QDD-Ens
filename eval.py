import cv2
import os
import numpy as np
from PIL import Image
import warnings
import torch
from torch.autograd import Variable as V
import sklearn.metrics as metrics

warnings.filterwarnings('ignore')

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from networks.qmenet import QME_Net_cenet

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

BATCHSIZE_PER_CARD = 1

def calculate_auc_test(prediction, label):
    result_1D = prediction.flatten()
    label_1D = label.flatten()
    label_1D = label_1D / 255
    auc = metrics.roc_auc_score(label_1D, result_1D)
    return auc


def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    vessel_IoU = TP / (FN + TP + FP)
    background_IoU = TN / (FN + TP + TN)
    IoU = (vessel_IoU+background_IoU)/2

    return acc, IoU


class TTAFrame():
    def __init__(self, net):
        self.net = net

    def test_one_img_from_path_8a1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))
        imgg = np.array(np.rot90(img))

        ########################
        img1 = img[None]
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img1)
        maskb, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img2)
        maskc, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img3)
        maskd, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img4)

        maska = maska.squeeze().cpu().data.numpy()
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = maskd.squeeze().cpu().data.numpy()

        #########################
        imgg1 = imgg[None]
        imgg2 = np.array(imgg1)[:, ::-1]
        imgg3 = np.array(imgg1)[:, :, ::-1]
        imgg4 = np.array(imgg2)[:, :, ::-1]

        imgg1 = imgg1.transpose(0, 3, 1, 2)
        imgg2 = imgg2.transpose(0, 3, 1, 2)
        imgg3 = imgg3.transpose(0, 3, 1, 2)
        imgg4 = imgg4.transpose(0, 3, 1, 2)

        imgg1 = V(torch.Tensor(np.array(imgg1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg2 = V(torch.Tensor(np.array(imgg2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg3 = V(torch.Tensor(np.array(imgg3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg4 = V(torch.Tensor(np.array(imgg4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maskaa, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg1)
        maskbb, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg2)
        maskcc, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg3)
        maskdd, mask, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg4)

        maskaa = maskaa.squeeze().cpu().data.numpy()
        maskbb = maskbb.squeeze().cpu().data.numpy()
        maskcc = maskcc.squeeze().cpu().data.numpy()
        maskdd = maskdd.squeeze().cpu().data.numpy()

        mask1 = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]
        mask2 = maskaa + maskbb[::-1] + maskcc[:, ::-1] + maskdd[::-1, ::-1]

        mask2 = np.rot90(mask2)[::-1, ::-1]
        mask = mask1 + mask2

        return mask

    def test_one_img_from_path_8a2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))
        imgg = np.array(np.rot90(img))

        ########################
        img1 = img[None]
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask, maska, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img1)
        mask, maskb, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img2)
        mask, maskc, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img3)
        mask, maskd, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img4)

        maska = maska.squeeze().cpu().data.numpy()
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = maskd.squeeze().cpu().data.numpy()

        #########################
        imgg1 = imgg[None]
        imgg2 = np.array(imgg1)[:, ::-1]
        imgg3 = np.array(imgg1)[:, :, ::-1]
        imgg4 = np.array(imgg2)[:, :, ::-1]

        imgg1 = imgg1.transpose(0, 3, 1, 2)
        imgg2 = imgg2.transpose(0, 3, 1, 2)
        imgg3 = imgg3.transpose(0, 3, 1, 2)
        imgg4 = imgg4.transpose(0, 3, 1, 2)

        imgg1 = V(torch.Tensor(np.array(imgg1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg2 = V(torch.Tensor(np.array(imgg2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg3 = V(torch.Tensor(np.array(imgg3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg4 = V(torch.Tensor(np.array(imgg4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask, maskaa, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg1)
        mask, maskbb, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg2)
        mask, maskcc, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg3)
        mask, maskdd, mask, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg4)

        maskaa = maskaa.squeeze().cpu().data.numpy()
        maskbb = maskbb.squeeze().cpu().data.numpy()
        maskcc = maskcc.squeeze().cpu().data.numpy()
        maskdd = maskdd.squeeze().cpu().data.numpy()

        mask1 = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]
        mask2 = maskaa + maskbb[::-1] + maskcc[:, ::-1] + maskdd[::-1, ::-1]

        mask2 = np.rot90(mask2)[::-1, ::-1]
        mask = mask1 + mask2

        return mask

    def test_one_img_from_path_8a3(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))
        imgg = np.array(np.rot90(img))

        ########################
        img1 = img[None]
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask, mask, maska, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img1)
        mask, mask, maskb, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img2)
        mask, mask, maskc, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img3)
        mask, mask, maskd, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img4)

        maska = maska.squeeze().cpu().data.numpy()
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = maskd.squeeze().cpu().data.numpy()

        #########################
        imgg1 = imgg[None]
        imgg2 = np.array(imgg1)[:, ::-1]
        imgg3 = np.array(imgg1)[:, :, ::-1]
        imgg4 = np.array(imgg2)[:, :, ::-1]

        imgg1 = imgg1.transpose(0, 3, 1, 2)
        imgg2 = imgg2.transpose(0, 3, 1, 2)
        imgg3 = imgg3.transpose(0, 3, 1, 2)
        imgg4 = imgg4.transpose(0, 3, 1, 2)

        imgg1 = V(torch.Tensor(np.array(imgg1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg2 = V(torch.Tensor(np.array(imgg2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg3 = V(torch.Tensor(np.array(imgg3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg4 = V(torch.Tensor(np.array(imgg4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask, mask, maskaa, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg1)
        mask, mask, maskbb, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg2)
        mask, mask, maskcc, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg3)
        mask, mask, maskdd, mask\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg4)

        maskaa = maskaa.squeeze().cpu().data.numpy()
        maskbb = maskbb.squeeze().cpu().data.numpy()
        maskcc = maskcc.squeeze().cpu().data.numpy()
        maskdd = maskdd.squeeze().cpu().data.numpy()

        mask1 = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]
        mask2 = maskaa + maskbb[::-1] + maskcc[:, ::-1] + maskdd[::-1, ::-1]

        mask2 = np.rot90(mask2)[::-1, ::-1]
        mask = mask1 + mask2

        return mask

    def test_one_img_from_path_8l(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))
        imgg = np.array(np.rot90(img))

        ########################
        img1 = img[None]
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask, mask, mask, maska\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img1)
        mask, mask, mask, maskb\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img2)
        mask, mask, mask, maskc\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img3)
        mask, mask, mask, maskd\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(img4)

        maska = maska.squeeze().cpu().data.numpy()
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = maskd.squeeze().cpu().data.numpy()

        #########################
        imgg1 = imgg[None]
        imgg2 = np.array(imgg1)[:, ::-1]
        imgg3 = np.array(imgg1)[:, :, ::-1]
        imgg4 = np.array(imgg2)[:, :, ::-1]

        imgg1 = imgg1.transpose(0, 3, 1, 2)
        imgg2 = imgg2.transpose(0, 3, 1, 2)
        imgg3 = imgg3.transpose(0, 3, 1, 2)
        imgg4 = imgg4.transpose(0, 3, 1, 2)

        imgg1 = V(torch.Tensor(np.array(imgg1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg2 = V(torch.Tensor(np.array(imgg2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg3 = V(torch.Tensor(np.array(imgg3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        imgg4 = V(torch.Tensor(np.array(imgg4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask, mask, mask, maskaa\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg1)
        mask, mask, mask, maskbb\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg2)
        mask, mask, mask, maskcc\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4 = self.net.forward(imgg3)
        mask, mask, mask, maskdd\
            ,  n1e1, n2e1, n3e1 \
            , n1e2, n2e2, n3e2 \
            , n1e3, n2e3, n3e3 \
            , n1e4, n2e4, n3e4 \
            , n1e5, n2e5, n3e5 \
            , n1d1, n2d1, n3d1 \
            , n1d2, n2d2, n3d2 \
            , n1d3, n2d3, n3d3 \
            , n1d4, n2d4, n3d4  = self.net.forward(imgg4)

        maskaa = maskaa.squeeze().cpu().data.numpy()
        maskbb = maskbb.squeeze().cpu().data.numpy()
        maskcc = maskcc.squeeze().cpu().data.numpy()
        maskdd = maskdd.squeeze().cpu().data.numpy()

        mask1 = maska + maskb[::-1] + maskc[:, ::-1] + maskd[::-1, ::-1]
        mask2 = maskaa + maskbb[::-1] + maskcc[:, ::-1] + maskdd[::-1, ::-1]

        mask2 = np.rot90(mask2)[::-1, ::-1]
        mask = mask1 + mask2

        return mask

    # def load(self, path):
    #     model_dict = self.net.state_dict()
    #     checkpoint = torch.load(path)
    #     for k, v in checkpoint.items():
    #         # print
    #         # name = k[7:]  # remove `module.`
    #         name = k
    #         model_dict[name] = v
    #     self.net.load_state_dict(model_dict)
    #

def test_assistant(name, mask):
    gt_root = './dataset/DRIVE/test/1st_manual'
    threshold = 5
    disc = 20

    new_mask = mask.copy()
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)

    ground_truth_path = os.path.join(gt_root, name.split('_')[0] + '_manual1.gif')
    ground_truth = np.array(Image.open(ground_truth_path))

    mask = cv2.resize(mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))

    new_mask = cv2.resize(new_mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))

    predi_mask = np.zeros(shape=np.shape(mask))
    predi_mask[mask > disc] = 1
    gt = np.zeros(shape=np.shape(ground_truth))
    gt[ground_truth > 0] = 1

    acc, IoU = accuracy(predi_mask[:, :, 0], gt)
    auc = calculate_auc_test(new_mask / 8., ground_truth)

    return acc, IoU,  auc










def loadweight(net, path):
    model_dict = net.state_dict()
    checkpoint = torch.load(path)

    for k, v in checkpoint.items():
        name = k
        model_dict[name] = v
    net.load_state_dict(model_dict)
    return net




def test_ce_net_vessel(net, path):
    source = './dataset/DRIVE/test/images/'
    val = os.listdir(source)

    solver = TTAFrame(net)

    total_acc = []
    total_acc1 = []
    total_acc2 = []
    total_acc3 = []
    total_acc0 = []

    total_iou = []
    total_iou1 = []
    total_iou2 = []
    total_iou3 = []
    total_iou0 = []

    total_auc1 = []
    total_auc2 = []
    total_auc3 = []
    total_auc = []
    total_auc0 = []

    for i, name in enumerate(val):
        image_path = os.path.join(source, name.split('.')[0] + '.tif')
        print(image_path)
        mask1 = solver.test_one_img_from_path_8a1(image_path)
        mask2 = solver.test_one_img_from_path_8a2(image_path)
        mask3 = solver.test_one_img_from_path_8a3(image_path)
        mask0 = solver.test_one_img_from_path_8l(image_path)

        mask = (mask1 + mask2 + mask3 + mask0) / 4

        acc0, IoU0, auc0 = test_assistant(name, mask0)
        acc1, IoU1, auc1 = test_assistant(name, mask1)
        acc2, IoU2, auc2 = test_assistant(name, mask2)
        acc3, IoU3, auc3 = test_assistant(name, mask3)
        acc ,  IoU, auc = test_assistant(name, mask)

        total_acc1.append(acc1)
        total_acc2.append(acc2)
        total_acc3.append(acc3)
        total_acc0.append(acc0)
        total_acc.append(acc)

        total_iou1.append(IoU1)
        total_iou2.append(IoU2)
        total_iou3.append(IoU3)
        total_iou0.append(IoU0)
        total_iou.append(IoU)

        total_auc1.append(auc1)
        total_auc2.append(auc2)
        total_auc3.append(auc3)
        total_auc0.append(auc0)
        total_auc.append(auc)

        print(i + 1, 'Base learner1: Accuracy: ', acc1,  ' IoU: ', IoU1, ' AUC: ', auc1)
        print(i + 1, 'Base learner2: Accuracy: ', acc2,  ' IoU: ', IoU2, ' AUC: ', auc2)
        print(i + 1, 'Base learner3: Accuracy: ', acc3,  ' IoU: ', IoU3, ' AUC: ', auc3)
        print(i + 1, 'Meta learner: Accuracy: ', acc0,  ' IoU: ', IoU0, ' AUC: ', auc0)

        print(i + 1, 'ensemble prediction: Accuracy: ', acc,  ' IoU: ', IoU, ' AUC: ', auc)




    print(' Base learner1: Accuracy: ', np.mean(total_acc1), ' IoU: ', np.mean(total_iou1), ' AUC: ', np.mean(total_auc1))
    print(' Base learner2: Accuracy: ', np.mean(total_acc2), ' IoU: ', np.mean(total_iou2), ' AUC: ', np.mean(total_auc2))
    print(' Base learner3: Accuracy: ', np.mean(total_acc3), ' IoU: ', np.mean(total_iou3), ' AUC: ', np.mean(total_auc3))
    print(' Meta learner : Accuracy: ', np.mean(total_acc0), ' IoU: ', np.mean(total_iou0), ' AUC: ', np.mean(total_auc0))

    print(' Ensemble prediction: Accuracy: ', np.mean(total_acc), ' IoU: ', np.mean(total_iou), ' AUC: ', np.mean(total_auc))

if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluation script for QME-Net models",formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-path', default='weights/NQ4/', type=str, help='store the trained model')

    args = parser.parse_args()

    net = QME_Net_cenet().cuda()
    net.eval()
    net = loadweight(net, args.model_path+'QME-Net_DRIVE.th')

    test_ce_net_vessel(net,  args.model_path)


