"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""
# remember to change paths

import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io
import gc
# import argparse

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import copy
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('./TensorBoardX/')


# Convolution module
# use conv to capture local features, instead of postion embedding.
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class deepConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize,  *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias= False, max_norm = 2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride = (1,3))
            )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(1,outF, kernalSize, padding = 0, max_norm = 2, *args, **kwargs),
                Conv2dWithConstraint(25, 25, (nChan, 1), padding = 0, bias= False, max_norm = 2),
                nn.BatchNorm2d(outF),
                nn.ELU(),
                nn.MaxPool2d((1,3), stride = (1,3))
                )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(inF, outF, kernalSize, max_norm = 0.5,*args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 2, dropoutP = 0.25, *args, **kwargs):
        super(deepConvNet, self).__init__()

        kernalSize = (1,10)
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
            for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))
        
        self.projection = nn.Sequential(
#             nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):

        x = self.allButLastLayers(x)
        x = self.projection(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=120):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 120, (1, 15), (1, 1)),
            nn.Conv2d(120, 120, (64, 1), (1, 1)),
            nn.BatchNorm2d(120),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(120, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            # nn.Linear(44720, 2440),
            # nn.ELU(),
            nn.Linear(8600, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=120, depth=6, n_classes=4, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

class DeepConformer(nn.Sequential):
    def __init__(self, deepconv_args, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(

            deepConvNet(**deepconv_args),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

class EEGDataset(Dataset):
    def __init__(self, data_root, subs, batch_size, augment=True):
        self.root = data_root
        self.subs = subs
        self.batch_size = batch_size
        self.augment = augment
        self.allData=None
        self.allLabel=None
        self.create_dataset()
    
    def get_source_data(self, sub):
        total_data = scipy.io.loadmat(os.path.join(self.root, '%d.mat' % sub))
        train_data = total_data['data']
        train_label = total_data['label']

        train_data = np.transpose(train_data, (2, 1, 0))
        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)

        allData = train_data
        allLabel = train_label[0]

        shuffle_num = np.random.permutation(len(allData))
        allData = allData[shuffle_num, :, :, :]
        allLabel = allLabel[shuffle_num]

        # standardize
        target_mean = np.mean(allData)
        target_std = np.std(allData)
        allData = (allData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return allData, allLabel

    def create_dataset(self):
        for sub in self.subs:
            allData, allLabel = self.get_source_data(sub)
            if(self.allData is None):
                self.allData = allData
                self.allLabel = allLabel
            else:
                self.allData = np.concatenate((self.allData, allData), axis=0)
                self.allLabel = np.concatenate((self.allLabel, allLabel), axis=0)
        
        self.allData = torch.from_numpy(self.allData)
        self.allLabel = torch.from_numpy(self.allLabel)
        
        if(self.augment):
            aug_data, aug_label = self.interaug(self.allData, self.allLabel)
            self.allData = torch.cat((self.allData, aug_data))
            self.allLabel = torch.cat((self.allLabel, aug_label))

    def __len__(self):
        return len(self.allData)
    
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 64, 480))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 60:(rj + 1) * 60] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 60:(rj + 1) * 60]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data)
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label)
        aug_label = aug_label.long()
        return aug_data, aug_label

    def __getitem__(self, idx):
        return self.allData[idx], self.allLabel[idx]

class ExP():
    def __init__(self, deepconv_args, train_subs, val_subs, test_subs, randomFolder, fold_num, patience=50, mdl=False, mdl_ratio=0.5):
        super(ExP, self).__init__()
        self.batch_size = 64
        self.n_epochs = 1000
        self.n_finetune_epochs = 100
        self.c_dim = 4
        self.lr = 0.004 
        self.weight_decay=0.005
        # batch_size = 32
        # epochs = 100
        self.momentum = 0.8
        # self.lr = 0.0002
        # self.b1 = 0.5
        # self.b2 = 0.999
        # self.dimension = (190, 50)
        self.train_subs = train_subs
        self.val_subs = val_subs
        self.test_subs = test_subs
        self.fold_num = fold_num
        self.mdl = mdl
        self.mdl_ratio = mdl_ratio
        self.load_best_model = True
        self.patience = patience
        self.finetune_split = 0.2

        self.start_epoch = 0
        self.root = '/home/msai/anishmad001/codes/EEG-Conformer/data/phisionet'

        
        self.results_folder = os.path.join("./results", randomFolder, str(self.fold_num))

        if(not os.path.exists(self.results_folder)):
            os.makedirs(self.results_folder)

        if(self.mdl):
            # self.log_write_file = os.path.join(self.results_folder, "log_subject%d_mdl.txt" % self.test_sub)
            pass
        else:
            self.log_write_file = os.path.join(self.results_folder, "train_log.txt")


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        # self.model = Conformer().cuda()
        self.model = DeepConformer(deepconv_args, emb_size=200).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        self.model_save_path = self.results_folder
    
    def log_write(self, line):
        with open(self.log_write_file, 'a') as f:
            f.write(line)

    # Segmentation and Reconstruction (S&R) data augmentation
    # def interaug(self, timg, label):  
    #     aug_data = []
    #     aug_label = []
    #     for cls4aug in range(4):
    #         cls_idx = np.where(label == cls4aug)
    #         tmp_data = timg[cls_idx]
    #         tmp_label = label[cls_idx]

    #         tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 64, 480))
    #         for ri in range(int(self.batch_size / 4)):
    #             for rj in range(8):
    #                 rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
    #                 tmp_aug_data[ri, :, :, rj * 60:(rj + 1) * 60] = tmp_data[rand_idx[rj], :, :,
    #                                                                   rj * 60:(rj + 1) * 60]

    #         aug_data.append(tmp_aug_data)
    #         aug_label.append(tmp_label[:int(self.batch_size / 4)])
    #     aug_data = np.concatenate(aug_data)
    #     aug_label = np.concatenate(aug_label)
    #     aug_shuffle = np.random.permutation(len(aug_data))
    #     aug_data = aug_data[aug_shuffle, :, :]
    #     aug_label = aug_label[aug_shuffle]

    #     aug_data = torch.from_numpy(aug_data).cuda()
    #     aug_data = aug_data.float()
    #     aug_label = torch.from_numpy(aug_label).cuda()
    #     aug_label = aug_label.long()
    #     return aug_data, aug_label

    def get_source_data(self, sub):
        # train data
        total_data = scipy.io.loadmat(os.path.join(self.root, '%d.mat' % sub))
        train_data = total_data['data']
        train_label = total_data['label']

        train_data = np.transpose(train_data, (2, 1, 0))
        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)

        allData = train_data
        allLabel = train_label[0]

        shuffle_num = np.random.permutation(len(allData))
        allData = allData[shuffle_num, :, :, :]
        allLabel = allLabel[shuffle_num]

        # standardize
        target_mean = np.mean(allData)
        target_std = np.std(allData)
        allData = (allData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return allData, allLabel

    def save_plots(self, train_accs, val_accs, train_losses, val_losses, model_path, prefix=''):
        # Saving train-val loss curve
        plt.figure()
        plt.title("Loss vs. Number of Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(range(1,len(train_losses)+1),train_losses,label="Train Loss")
        plt.plot(range(1,len(val_losses)+1),val_losses,label="Validation Loss")
        plt.legend(loc='upper right')
        file_name = prefix + 'loss.png'
        save_path = os.path.join(model_path, file_name)
        plt.savefig(save_path)

        # Saving train-val accuracy curve
        plt.figure()
        plt.title("Accuracy vs. Number of Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1,len(train_accs)+1),train_accs,label="Train Accuracy")
        plt.plot(range(1,len(val_accs)+1),val_accs,label="Validation Accuracy")
        plt.legend(loc='upper right')
        file_name = prefix + 'acc.png'
        save_path = os.path.join(model_path, file_name)
        plt.savefig(save_path)
    
    def validate(self):
        loss_vals = []
        correct_sum = 0
        num = 0
        self.model.eval()
        with torch.no_grad():
            print("Starting validation")
            for val_img, val_img_label in self.val_dataloader:
                val_img = Variable(val_img.cuda().type(self.Tensor))
                val_img_label = Variable(val_img_label.cuda().type(self.LongTensor))
                Tok, Cls = self.model(val_img)
                loss_val = self.criterion_cls(Cls, val_img_label)
                loss_vals.append(loss_val.detach().cpu().numpy())
                y_pred = torch.max(Cls, 1)[1]
                correct_sum += (y_pred == val_img_label).cpu().numpy().astype(int).sum()
                num += val_img_label.size(0)
            
        print("Finished validation")

        val_loss = sum(loss_vals) / len(loss_vals)
        val_acc = correct_sum / num
        return val_loss, val_acc
    
    def test(self):
        loss_vals = []
        correct_sum = 0
        num = 0
        self.model.eval()
        with torch.no_grad():
            print("Starting testing")
            for test_img, test_img_label in self.test_dataloader:
                test_img = Variable(test_img.cuda().type(self.Tensor))
                test_img_label = Variable(test_img_label.cuda().type(self.LongTensor))
                Tok, Cls = self.model(test_img)
                loss_val = self.criterion_cls(Cls, test_img_label)
                loss_vals.append(loss_val.detach().cpu().numpy())
                y_pred = torch.max(Cls, 1)[1]
                correct_sum += (y_pred == test_img_label).cpu().numpy().astype(int).sum()
                num += test_img_label.size(0)
            
        print("Finished testing")

        test_loss = sum(loss_vals) / len(loss_vals)
        test_acc = correct_sum / num
        return test_loss, test_acc
    
    def get_unseen_test_data(self):
        self.unseen_test_sub = random.choice(self.test_subs)
        self.log_write(f"Unseen test subject: {self.unseen_test_sub}\n")
        self.unseenTestData, self.unseenTestLabel = self.get_source_data(self.unseen_test_sub)


    def train(self, mdl=False):
        dataset = EEGDataset(data_root=self.root, subs=self.train_subs, batch_size=self.batch_size)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = EEGDataset(data_root=self.root, subs=self.val_subs, batch_size=self.batch_size, augment=False)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_dataset = EEGDataset(data_root=self.root, subs=self.test_subs, batch_size=self.batch_size, augment=False)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        # Optimizers
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        

        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)

        bestAcc = 0
        averAcc = 0
        num = 0
        val_accs = []
        val_losses = []
        train_accs = []
        train_losses = []
        no_imp = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        # model_path = os.path.join(self.model_save_path, str(self.fold_num))
        model_path = self.model_save_path
        if(not os.path.exists(model_path)):
            os.makedirs(model_path)
        
        best_model_path = os.path.join(model_path, 'model_best_val_acc.pth')

        for e in range(self.n_epochs):
            if(no_imp >= self.patience):
                print("Stopping training as validation accuracy has not improved for", self.patience, "epochs. Trained for ", e, "epochs.")
                self.log_write('Stopping training as validation accuracy has not improved for ' + str(self.patience) + ' epochs. Trained for ' + str(e) + ' epochs.\n')
                break
            
            # in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # validation process
            print("Staring eval")
            val_loss, acc = self.validate()
            train_pred = torch.max(outputs, 1)[1]
            train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

            print(f"Epoch {e+1} : Train loss: {loss.detach().cpu().numpy()} Train accuracy: {train_acc} Val loss: {val_loss} Val accuracy: {acc}")
            
            train_losses.append(loss.detach().cpu().numpy())
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(acc)

            self.log_write("Epoch " + str(e+1) + ": Validation Accuracy: " + str(acc) + "\n")

            num = num + 1
            averAcc = averAcc + acc
            if acc > bestAcc:
                bestAcc = acc
                torch.save(self.model, best_model_path)
                no_imp = 0
            else:
                no_imp += 1
        
        
        if(mdl):
            torch.save(self.model, os.path.join(model_path, 'model_mdl.pth'))
        else:
            torch.save(self.model, os.path.join(model_path, 'model.pth'))
        
        self.save_plots(train_accs, val_accs, train_losses, val_losses, model_path)
        
        averAcc = averAcc / num
        print('The average val accuracy is:', averAcc)
        print('The best val accuracy is:', bestAcc)
        self.log_write('The average val accuracy is: ' + str(averAcc) + "\n")
        self.log_write('The best val accuracy is: ' + str(bestAcc) + "\n")

        if(self.load_best_model):
            self.model = torch.load(best_model_path)
        
        test_loss, unseen_acc = self.test()
        self.log_write("Unseen subjects test accuracy: " + str(unseen_acc) + "\n")
        self.log_write("Unseen subjects test loss: " + str(test_loss) + "\n")
        print(f"Unseen subjects test loss: {test_loss}")
        print(f"Unseen subjects test accuracy: {unseen_acc}")

        return unseen_acc, bestAcc, averAcc
    

    def finetune(self):
        self.get_unseen_test_data()
        idx = int(self.unseenTestData.shape[0] * self.finetune_split)
        img = self.unseenTestData[:idx]
        label = self.unseenTestLabel[:idx]
        test_data = self.unseenTestData[idx:]
        test_label = self.unseenTestLabel[idx:]

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        val_accs = []
        val_losses = []
        train_accs = []
        train_losses = []

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        model_path = self.model_save_path
        if(not os.path.exists(model_path)):
            os.makedirs(model_path)

        for e in range(self.n_finetune_epochs):
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                # aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                # img = torch.cat((img, aug_data))
                # label = torch.cat((label, aug_label))


                tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # test process
            if (e + 1) % 1 == 0:
                with torch.no_grad():
                    self.model.eval()
                    Tok, Cls = self.model(test_data)


                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Finetune Epoch:', e,
                    '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                    '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                    '  Train accuracy %.6f' % train_acc,
                    '  Test accuracy is %.6f' % acc)
                
                train_losses.append(loss.detach().cpu().numpy())
                val_losses.append(loss_test.detach().cpu().numpy())
                train_accs.append(train_acc)
                val_accs.append(acc)

                self.log_write("Finetune Epoch " + str(e+1) + ": Test Accuracy: " + str(acc) + "\n")

                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    torch.save(self.model, os.path.join(model_path, 'model_finetuned_sub_' + str(self.unseen_test_sub) + '_best_acc.pth'))
        
        torch.save(self.model, os.path.join(model_path, 'model_finetuned_sub_' + str(self.unseen_test_sub) + '.pth'))

        self.save_plots(train_accs, val_accs, train_losses, val_losses, model_path, prefix='finetuned_')

        averAcc = averAcc / num
        print('The average finetuned test accuracy is:', averAcc)
        print('The best finetuned test accuracy is:', bestAcc)
        self.log_write('The average finetuned test accuracy is: ' + str(averAcc) + "\n")
        self.log_write('The best finetuned test accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc

def getLosoSplit(subs, cv=5):
    shuffle_num = np.random.permutation(len(subs))
    subs = np.array(subs)
    shuffled_subs = subs[shuffle_num]
    folds = np.array_split(shuffled_subs, cv)
    print("Number of folds:", len(folds))
    return folds

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    best = 0
    aver = 0
    sub_spec_sum = 0
    cross_sub_sum = 0
    num_exps = 5
    BAD_SUBJECTS_EEGBCI = [87, 89, 91, 99]
    subs = list(i for i in range(109) if i not in BAD_SUBJECTS_EEGBCI)
    
    # subs = list(range(1,10))
    prev_test = None
    skip_list = []
    num = 0
    mdl = False
    randomFolder = str(time.strftime("%Y-%m-%d--%H-%M", time.localtime()))+ '-'+str(random.randint(1,1000))
    print("Saving results in", randomFolder)
    done_list = []

    deepconv_args = {'nChan': 22, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass': 4, 'doWeightNorm': True}
    
    folds = getLosoSplit(subs)

    # print("Done")
    # exit(0)
    
    for i in range(len(folds)):
        # if(i == 0):
        #     print("First iteration")
        starttime = datetime.datetime.now()

        # folds_copy = copy.deepcopy(folds)

        test_idx = i
        test_subs = folds[test_idx].tolist()

        print("Number of test subjects:", len(test_subs))
        print("Test subjects:", test_subs)

        if(i-1<0):
            val_idx = random.choice(list(range(len(folds))))
        else:
            val_idx = i-1
        
        val_subs = folds[val_idx].tolist()
        print("Number of validation subjects:", len(val_subs))
        print("Validation subs:", val_subs)

        train_subs = []
        for fold_idx in range(len(folds)):
            if(fold_idx == val_idx or fold_idx == test_idx):
                continue
            train_subs.extend(folds[fold_idx].tolist())

        print("Number of subjects for training:", len(train_subs))
        print("Train subs:", train_subs)

        # exit(0)

        # sub = random.choice(subs)
        # while(sub in skip_list or sub in done_list):
        #     sub = random.choice(subs)
        
        

        seed_n = np.random.randint(2023)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        # train_subs, val_sub, test_subs = getLosoSplit(subs)
        # print('Test Subject %d' % (sub))
        # exit(0)

        exp = ExP(deepconv_args, train_subs, val_subs, test_subs, randomFolder, fold_num=i+1, mdl=mdl)
        if(i == 0):
            summary(exp.model, (1, 64, 480))
        
        print("Number of trainable parameters:", count_parameters(exp.model))

        # exit(0)

        unseen_acc, bestAcc, averAcc = exp.train(mdl)
        # sub_spec_sum += test_acc
        cross_sub_sum += unseen_acc

        if(not mdl):
            result_write_file = os.path.join(exp.results_folder, "sub_result.txt")
        else:
            result_write_file = os.path.join(exp.results_folder, "sub_result_mdl.txt")


        with open(result_write_file, 'a') as result_write:
            result_write.write('Seed is: ' + str(seed_n) + "\n")
            result_write.write('The best val accuracy is: ' + str(bestAcc) + "\n")
            result_write.write('The average val accuracy is: ' + str(averAcc) + "\n")
            # result_write.write('Test Subject ' + str(sub) + ' : ' + 'Subject specific test accuracy is: ' + str(test_acc) + "\n")
            result_write.write('Unseen Subject test accuracy is: ' + str(unseen_acc) + "\n")
        
        if(mdl):
            endtime = datetime.datetime.now()
            # print("Summary for subject", sub)
            print("Average validation accuracy:", averAcc)
            # print("Subject specific accuracy:", test_acc)
            print('Cross subject accuracy: ' + str(unseen_acc))
            # print('subject %d duration: '%(sub) + str(endtime - starttime))

            best = best + bestAcc
            aver = aver + averAcc
            num+=1
            # prev_test = sub
            del exp.model
            del exp
            gc.collect()
            torch.cuda.empty_cache()
            # done_list.append(sub)
            continue
        
        # Finetune and test
        for name, param in exp.model.named_parameters():
            param.requires_grad = True if("clshead" in name or "fc" in name) else False

        print("Number of trainable parameters for finetuning:", count_parameters(exp.model))

        bestAccFt, averAccFt = exp.finetune()
        with open(result_write_file, 'a') as result_write:
            result_write.write('Seed is: ' + str(seed_n) + "\n")
            result_write.write('The best finetuned accuracy is: ' + str(bestAccFt) + "\n")
            result_write.write('The average finetuned accuracy is: ' + str(averAccFt) + "\n")
        
        endtime = datetime.datetime.now()
        # print("Summary for subject", sub)
        print("Average validation accuracy:", averAcc)
        # print("Subject specific accuracy:", test_acc)
        print('Cross subject accuracy: ' + str(unseen_acc))
        print('Cross subject accuracy after finetuning for subject ' + str(exp.unseen_test_sub) + ': ' + str(bestAccFt))
        print('fold %d duration: '%(i+1) + str(endtime - starttime))

        best = best + bestAccFt
        aver = aver + averAccFt
        num+=1
        # prev_test = sub
        # done_list.append(sub)

    # sub_spec_avg = sub_spec_sum / num
    cross_sub_avg = cross_sub_sum / num
    best = best / num
    aver = aver / num

    with open(result_write_file, 'a') as result_write:
        # result_write.write('**The average sub-specific accuracy is: ' + str(sub_spec_avg) + "\n")
        if(mdl):
            result_write.write("Average cross-subject test accuracy with MDL:", cross_sub_avg)
        else:
            result_write.write('The average average cross-subject test accuracy after finetuning is: ' + str(aver) + "\n")
            result_write.write('The average best cross-subject test accuracy after finetuning is: ' + str(best) + "\n")
    
    print('**FINAL SUMMARY**')
    # print('The average sub-specific accuracy is: ' + str(sub_spec_avg) + "\n")
    if(mdl):
        print("Average cross-subject test accuracy with MDL: " + str(cross_sub_avg))
    else:
        print('The average average cross-subject test accuracy after finetuning is: ' + str(aver) + "\n")
        print('The average best cross-subject test accuracy after finetuning is: ' + str(best) + "\n")


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))