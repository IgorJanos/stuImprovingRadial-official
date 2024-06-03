#------------------------------------------------------------------------------
#
#   Improving Radial ...
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
import yaml
import torch
import numpy as np
import hydra
import cv2
import matplotlib.pyplot as plt
import torchvision.io as io
import torchvision.transforms as transforms
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from src.backstage.trainer import Trainer, DataSource
from src.backstage.utils import getTqdm, undistort
from src.backstage.loggers import predict_both


import torchvision.transforms.functional as TF

TO_FLOAT = transforms.ToTensor()

class Evaluator:
    def __init__(self, config_filename):

        # CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load config
        config = yaml.safe_load(Path(config_filename).read_text())
        config = OmegaConf.create(config)

        # Load trainer
        run_folder = Path(config_filename).parent.parent
        self.trainer = Trainer(config, output_folder=run_folder.as_posix())

        # Load best checkpoint
        self.trainer.load_checkpoint("checkpoint.pt")
        print("Done loading")

     
    def evaluate(self, hf=False):
        dataX = []
        dataY = []
        dataX2 = []
        dataY2 = []

        model = torch.nn.DataParallel(self.trainer.model)
        model.eval()

        # Accumulator
        mdld_acc = 0.0
        mdld_count = 0

        with torch.no_grad():
            progress = getTqdm(self.trainer.dsVal.loader)
            for (x, k) in progress:
                x = x.to(self.device)
                k = k.to(self.device)
                x = self.trainer.dsTrain.transform_val(x).contiguous()

                # Compute the prediction
                k_hat, comp_hat, comp_target = model(x, k)

                if (hf):
                    # Flip images
                    x_flip = TF.hflip(x)
                    k_hat2, _, _ = model(x_flip, k)
                    # Average the predictions
                    k_hat = (k_hat + k_hat2) / 2

                if (self.trainer.nOutput == 1):
                    error_k1 = (k_hat[:,0:1] - k[:,0:1])
                elif (self.trainer.nOutput == 2):
                    error_k1 = (k_hat[:,0:1] - k[:,0:1])
                    error_k2 = (k_hat[:,1:2] - k[:,1:2])

                # Get MDLD
                mdld = self.trainer.criterionMDLD(k_hat, k).mean()
                mdld_acc += mdld.item()
                mdld_count += 1

               # Store data in the result lists
                dataX.append(k[:,0:1].cpu().detach().numpy())
                dataY.append(error_k1[:,0:1].cpu().detach().numpy())

                if (self.trainer.nOutput == 2):
                    dataX2.append(k[:,1:2].cpu().detach().numpy())
                    dataY2.append(error_k2[:,0:1].cpu().detach().numpy())

        # MDLD average
        mdld_avg = mdld_acc / mdld_count
        print("MDLD: ", mdld_avg)

        # Concatenate the results
        dataX = np.concatenate(dataX, axis=0)
        dataY = np.concatenate(dataY, axis=0)
        dataX2 = np.concatenate(dataX2, axis=0)
        dataY2 = np.concatenate(dataY2, axis=0)
        return dataX[:,0], dataY[:,0], dataX2[:,0], dataY2[:,0]
    
    def getImages(self, images):
        result = []
        for i in images:
            result.append(self.trainer.dsVal.ds[i][0].unsqueeze(0))
        result = torch.concat(result, dim=0)
        return result
    
    def arrangeImages(self, x, shape):
        B,C,H,W = x.shape
        TH,TW = shape

        result = np.zeros(shape=(C,TH*H,TW*W), dtype=np.uint8)

        for ty in range(TH):
            for tx in range(TW):
                idx = ty*TW + tx
                dx = tx*W
                dy = ty*H
                image = x[idx].cpu().detach().numpy()
                image = (255*image).astype(np.uint8)
                result[:,dy:dy+H,dx:dx+W] = image

        result = np.transpose(result, (1,2,0))
        return result

    def rectifyImage(self, file_name):
        # Load image
        x = cv2.imread(file_name, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        # Make float tensor
        x_float = x.astype(np.float32) / 255.0
        x_float = np.transpose(x_float, (2,0,1))
        x_float = torch.from_numpy(x_float)

        # network input
        net_input = self.trainer.dsVal.transform_val(x_float).unsqueeze(0).to(self.device)
        model = self.trainer.model
        model.eval()

        # With flip
        with torch.no_grad():
            _, k_hat = predict_both(model, net_input)
        
        undistorted = undistort(x, k_hat[0].cpu().detach().numpy(), scaleShape=(1920,1080))
        return undistorted




def showImage(x):
    H,W,C = x.shape
    x = cv2.resize(x,(int(W*16.0/9.0),H), interpolation=cv2.INTER_AREA)
    fig = plt.figure(figsize=(32,32), facecolor="white")
    plt.axis("off")
    plt.imshow(x)
    plt.show()
    plt.close()

def showScatterPlot(k1, error_k1, name, ylabel):
    params = {
            'font.size': 20,
            'legend.loc': 'upper right'
        }
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(14,8), facecolor='white')
    axes = fig.subplots(1, 1)
    axes.set_ylim(top=0.15, bottom=-0.15)
    plt.scatter(k1, error_k1, s=1.0, label=name)
    plt.axhline(y=0.0, color="r", linestyle="--", linewidth=2.0)
    axes.set_xlabel("$k_1$")
    axes.set_ylabel(ylabel)
    #plt.legend()
    plt.show()
    plt.close()


def split_into_subintervals(data, key, value, intervals):
    result = []
    result_intervals = []
    for i in range(len(intervals)-1):
        range_low = intervals[i]
        range_high = intervals[i+1]
        selection = data[ (data[key] > range_low) & (data[key] <= range_high) ]
        result.append(selection[value].to_numpy())
        interval_name = f"$({range_low}, \; {range_high} ]$"
        result_intervals.append(interval_name)
    return result, result_intervals

def showBoxPlot(data, data_labels, name, ylabel): 
    fig = plt.figure(figsize=(14,8), facecolor='white')
    axes = fig.subplots(1, 1)
    plt.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0)
    axes.set_ylim(top=0.05, bottom=-0.05)
    bp = axes.boxplot(data, 0, '')
    for whisker in bp["whiskers"]:
        whisker.set(linewidth=2, linestyle="--")
    for cap in bp["caps"]:
        cap.set(linewidth=2)
    for median in bp["medians"]:
        median.set(linewidth=2, color="red")
    for box in bp["boxes"]:
        box.set(linewidth=2)

    axes.set_xticklabels(data_labels)
    axes.set_xlabel(name)
    axes.set_ylabel(ylabel)

    #plt.legend()
    plt.show()
    plt.close()
