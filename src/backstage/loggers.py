#------------------------------------------------------------------------------
#
#   Improving Radial ...
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
import wandb
import torch
import cv2

import yaml
import numpy as np
import matplotlib.pyplot as plt

from .utils import rescale, undistort, toNumpyImage, getTqdm, k2FromK1
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from .losses import MDLD

import multiprocessing

#------------------------------------------------------------------------------
#   Logger baseclass
#------------------------------------------------------------------------------

class Logger:
    def trainingStart(self):
        pass

    def trainingEnd(self):
        pass

    def epochStart(self, epoch, stats):
        pass

    def epochEnd(self, epoch, stats):
        pass

#------------------------------------------------------------------------------
#   Loggers baseclass
#------------------------------------------------------------------------------

class Loggers(Logger):
    def __init__(self, loggers=[]):
        self.loggers = loggers

    def trainingStart(self):
        for l in self.loggers:
            l.trainingStart()

    def trainingEnd(self):
        for l in self.loggers:
            l.trainingEnd()

    def epochStart(self, epoch, stats):
        for l in self.loggers:
            l.epochStart(epoch, stats)

    def epochEnd(self, epoch, stats):
        for l in self.loggers:
            l.epochEnd(epoch, stats)


class ConfigLogger(Logger):
    def __init__(self, trainer):
        self.trainer = trainer

    def trainingStart(self):
        fn = os.path.join(self.trainer.outputFolder, "config.yaml")
        s = yaml.dump(self.trainer.conf)
        with open(fn, "w") as f:
            f.write(s)


#------------------------------------------------------------------------------
#   CsvLogger baseclass
#------------------------------------------------------------------------------

class CsvLogger(Logger):
    def __init__(self, trainer, filename):
        self.trainer = trainer
        self.filename = os.path.join(self.trainer.outputFolder, filename)
        self.file = None
        self.lines = 0
        self.separator = ','

    def trainingStart(self):
        self.lines = 0
        self.file = open(self.filename, "w")

    def trainingEnd(self):
        if (self.file is not None):
            self.file.close()
            self.file = None

    def epochEnd(self, epoch, stats):
        if (self.file is not None):
            line = ""
            s = stats.getAvg()

            if (self.lines == 0):
                # CSV header
                line = self.separator.join(["epoch", "lr"] + list(s.keys()))
                self.file.write(line + "\n")

            # Join all values
            values = [ "{:.10f}".format(s[k]) for k in s.keys() ]
            values = [ 
                "{}".format(epoch+1),
                "{:.10f}".format(self.trainer.opt.param_groups[0]['lr'])
            ] + values
            line = self.separator.join(values)
            self.file.write(line + "\n")
            self.file.flush()
            self.lines += 1


#------------------------------------------------------------------------------
#   WandBLogger 
#------------------------------------------------------------------------------

class WandBLogger(Logger):
    def __init__(self, trainer, config, wandbConfigFile):
        self.trainer = trainer

        with open(wandbConfigFile, "r") as f:
            yamlConfig = yaml.safe_load(f)

        self.config = config
        self.kwargs = yamlConfig["wandb"]
        self.run = None

    def trainingStart(self):
        wandb.config = self.config
        self.run = wandb.init(
            config=self.config, 
            name=self.trainer.name,
            **self.kwargs
            )

    def trainingEnd(self):
        if (self.run is not None):
            wandb.finish()
            self.run = None

    def epochEnd(self, epoch, stats):
        s = stats.getAvg()
        self.run.log(s)



#------------------------------------------------------------------------------
#   ModelCheckpoint 
#------------------------------------------------------------------------------

class ModelCheckpoint(Logger):
    def __init__(self, trainer, singleFile=True, bestName=None):
        self.trainer = trainer
        self.folder = trainer.outputFolder
        self.singleFile = singleFile
        self.bestName = bestName
        self.bestValue = None

    def trainingEnd(self):
        self.saveCheckpoint(os.path.join(self.folder, "model-end.pt"))

    def epochEnd(self, epoch, stats):
        s = stats.getAvg()
        
        isBest = False
        if (self.bestName is None):
            isBest = True
        else:
            value = s[self.bestName]
            if (self.bestValue is None):
                isBest = True
                self.bestValue = value
                print("  new best metric: {}: {:.10f}".format(self.bestName, self.bestValue))
            else:
                if (value < self.bestValue):
                    self.bestValue = value
                    print("  new best metric: {}: {:.10f}".format(self.bestName, self.bestValue))
                    isBest = True

        if (isBest):
            if (self.singleFile):
                fn = os.path.join(self.folder, "checkpoint.pt")
            else:
                fn = os.path.join(self.folder, "checkpoint-{:04d}.pt".format(epoch+1))

            self.saveCheckpoint(fn)
            self.trainer.bestCheckpoint = fn

    def saveCheckpoint(self, fn):
        checkpoint = {
            "model": self.trainer.model.state_dict(),
            "opt": self.trainer.opt.state_dict()
        }
        torch.save(checkpoint, fn)



#------------------------------------------------------------------------------
#   ResultSampler
#------------------------------------------------------------------------------

def predict(model, x):
    return model(x)

def predict_flip(model, x):
    k_hat = torch.cat([
            model(x).unsqueeze(0),
            model(torch.flip(x, dims=(3,))).unsqueeze(0)
            ], dim=0)
    k_hat = torch.mean(k_hat, dim=0)
    return k_hat

def predict_both(model, x):
    k_hat = torch.cat([
            model(x).unsqueeze(0),
            model(torch.flip(x, dims=(3,))).unsqueeze(0)
            ], dim=0)
    k_hat_noflip = k_hat[0]
    k_hat_flip = torch.mean(k_hat, dim=0)
    return k_hat_noflip, k_hat_flip


class ResultSampler(Logger):
    def __init__(self, trainer, scaleShape, indices=None):
        self.trainer = trainer
        self.folder = os.path.join(self.trainer.outputFolder, "samples")
        self.scaleShape = scaleShape

        # Standard prediction function
        self.predict_fn = predict

        os.makedirs(self.folder, exist_ok=True)

        # Get a few samples
        lImages = []
        lK = []
        if (indices is not None):
            for i in indices:
                x, k = self.trainer.dsVal.ds[i]
                lImages.append(x.unsqueeze(0))
                lK.append(torch.from_numpy(k).unsqueeze(0))

        # Store the images & labels
        self.images = torch.cat(lImages, dim=0)
        self.k = torch.cat(lK, dim=0)

    def trainingEnd(self):

        if (self.trainer.bestCheckpoint is not None):
            if (os.path.isfile(self.trainer.bestCheckpoint)):
                print("Loading best checkpoint: ", self.trainer.bestCheckpoint)
                checkpoint = torch.load(self.trainer.bestCheckpoint, map_location=torch.device("cpu"))
                self.trainer.model.load_state_dict(checkpoint["model"])
                self.trainer.model = self.trainer.model.to(self.trainer.device)
            else:
                print("No best checkpoint.")
                return None

        print("Sampling results: ")

        model = self.trainer.model
        model.eval()
        with torch.no_grad():
            images = self.images.to(self.trainer.device)
            images = self.trainer.dsTrain.transform_val(images)
            khat = self.predict_fn(model, images)

        # Undistore & save
        for i in range(len(self.images)):
            imgOriginal = toNumpyImage(self.images[i])
            k = self.k[i].cpu().detach().numpy()
            kh = khat[i].cpu().detach().numpy()

            imgOriginal = cv2.cvtColor(imgOriginal, cv2.COLOR_RGB2BGR)

            # Undistortneme obrazky
            undistortedLabel = undistort(imgOriginal, k, scaleShape=self.scaleShape)
            undistortedEstimate = undistort(imgOriginal, kh, scaleShape=self.scaleShape)

            # Zmensime ...
            imgOriginal = rescale(imgOriginal, scaleShape=self.scaleShape)
            undistortedLabel = rescale(undistortedLabel, scaleShape=self.scaleShape)
            undistortedEstimate = rescale(undistortedEstimate, scaleShape=self.scaleShape)

            cv2.imwrite(os.path.join(self.folder, "image-{}-original.png".format(i)), imgOriginal)
            cv2.imwrite(os.path.join(self.folder, "image-{}-label.png".format(i)), undistortedLabel)
            cv2.imwrite(os.path.join(self.folder, "image-{}-estimate.png".format(i)), undistortedEstimate)


#------------------------------------------------------------------------------
#   PSNR_SSIM_Sampler
#------------------------------------------------------------------------------

class PSNR_SSIM_Sampler(Logger):
    def __init__(self, trainer):
        self.trainer = trainer
        self.fn = os.path.join(self.trainer.outputFolder, "psnr_ssim.csv")


    def trainingEnd(self):
        print("")
        print("Sampling PSNR, SSIM: ")

        model = torch.nn.DataParallel(self.trainer.model)
        model.eval()

        result = {
            "psnr": 0.0,
            "ssim": 0.0,
            "mdld": 0.0
        }
        result_flip = {
            "psnr": 0.0,
            "ssim": 0.0,
            "mdld": 0.0
        }
        nItems = 0

        # Multiprocessing pool
        pool = multiprocessing.Pool(12)


        with torch.no_grad():
            progress = getTqdm(self.trainer.dsVal.loader)
            progress.set_description("PSNR/SSIM")
            for (x, k) in progress:
                x = x.to(self.trainer.device)
                k = k.to(self.trainer.device)
                x = self.trainer.dsTrain.transform_val(x)

                # Compute the prediction
                k_hat_noflip, k_hat_flip = predict_both(model, x)
                
                if (self.trainer.nOutput == 1):
                    k_hat_noflip = k2FromK1(k_hat_noflip)     # compute k2 coefficient
                    k_hat_flip = k2FromK1(k_hat_flip)     # compute k2 coefficient
                    
                # Evaluate minibatch
                nItems += self.evalMinibatch(pool, x, k, k_hat_noflip, result)
                self.evalMinibatch(pool, x, k, k_hat_flip, result_flip)

        # Zapiseme do CSVcka
        csv = open(self.fn, "w")
        csv.write("psnr,ssim,mdld" + "\n")
        csv.write("{},{},{}".format(
            result["psnr"] / (nItems),
            result["ssim"] / (nItems),
            result["mdld"] / (nItems)
        ) + "\n")
        csv.write("{},{},{}".format(
            result_flip["psnr"] / (nItems),
            result_flip["ssim"] / (nItems),
            result_flip["mdld"] / (nItems)
        ) + "\n")
        csv.flush()
        csv.close()
        print("  PSNR: {}".format(result["psnr"] / (nItems)))
        print("  SSIM: {}".format(result["ssim"] / (nItems)))
        print("  MDLD: {}".format(result["mdld"] / (nItems)))


    def evalMinibatch(self, pool, inX, inK, inKhat, result):
        '''
            1. Pre vsetky obrazky zbehneme
                - Undistortneme podla labelu
                - Undistortneme podla khat
                - PSNR, SSIM
        '''

        b,c,h,w = inX.shape
        mMdld = MDLD(shape=(h,w), device=self.trainer.device)

        # MDLD
        m = mMdld(inKhat, inK).sum()
        result["mdld"] = result["mdld"] + m.cpu().detach().numpy().item()

        # PSNR, SSIM
        inX = inX.cpu().detach().numpy()
        inK = inK.cpu().detach().numpy()
        inKhat = inKhat.cpu().detach().numpy()

        # Make a batch of all individual items
        batch = [(inX[i], inK[i], inKhat[i]) for i in range(b)]

        # Parallel processing
        results = pool.map(work_item_psnr_ssim, batch)
        for (psnr,ssim) in results:
            result["psnr"] = result["psnr"] + psnr
            result["ssim"] = result["ssim"] + ssim        

        # Sprocesovali sme B obrazkov
        return b

def work_item_psnr_ssim(item):
    x, k, khat = item
    # Skonvertujeme na obrazok
    x = toNumpyImage(x)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    # Undistortneme obrazky
    undistortedLabel = undistort(x, k)
    undistortedEstimate = undistort(x, khat)
    # Spocitame metriky
    psnr = peak_signal_noise_ratio(undistortedLabel, undistortedEstimate)
    ssim = structural_similarity(undistortedLabel, undistortedEstimate, channel_axis=-1)
    return (psnr, ssim)


#------------------------------------------------------------------------------
#   ErrorDistributionLogger
#------------------------------------------------------------------------------

class ErrorDistributionLogger(Logger):
    def __init__(self, trainer):
        self.trainer = trainer
        self.out_folder = self.trainer.outputFolder
        self.nOutput = self.trainer.nOutput
        # Standard prediction function
        self.predict_fn = predict

    def trainingEnd(self):
        print("")
        print("Sampling Error Distribution: ")

        k1, error_k1, k2, error_k2 = self.evaluate()

        image_k1 = os.path.join(self.out_folder, "error_k1.png")
        image_k2 = os.path.join(self.out_folder, "error_k2.png")

        self.showScatterPlot(image_k1, k1, error_k1, "", "Error of $\widehat{k_1}$")
        self.showScatterPlot(image_k2, k2, error_k2, "", "Error of $\widehat{k_2}$")


    def showScatterPlot(self, file_path, k1, error_k1, name, ylabel):
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
        if (name != ""):
            plt.legend()
        plt.show()
        plt.savefig(file_path)
        plt.close()

    def evaluate(self):
        dataX = []
        dataY = []
        dataX2 = []
        dataY2 = []

        model = torch.nn.DataParallel(self.trainer.model)
        model.eval()

        with torch.no_grad():
            progress = getTqdm(self.trainer.dsVal.loader)
            for (x, k) in progress:
                x = x.to(self.trainer.device)
                k = k.to(self.trainer.device)
                x = self.trainer.dsVal.transform_val(x)

                # Compute the prediction
                k_hat = self.predict_fn(model, x)

                if (self.nOutput == 1):
                    error_k1 = (k_hat[:,0:1] - k[:,0:1])
                elif (self.nOutput == 2):
                    error_k1 = (k_hat[:,0:1] - k[:,0:1])
                    error_k2 = (k_hat[:,1:2] - k[:,1:2])


               # Store data in the result lists
                dataX.append(k[:,0:1].cpu().detach().numpy())
                dataY.append(error_k1[:,0:1].cpu().detach().numpy())

                if (self.nOutput == 2):
                    dataX2.append(k[:,1:2].cpu().detach().numpy())
                    dataY2.append(error_k2[:,0:1].cpu().detach().numpy())


        # Concatenate the results
        dataX = np.concatenate(dataX, axis=0)
        dataY = np.concatenate(dataY, axis=0)

        if (len(dataX2) > 0):
            dataX2 = np.concatenate(dataX2, axis=0)
            dataY2 = np.concatenate(dataY2, axis=0)
        else:
            dataX2 = np.zeros_like(dataX)
            dataY2 = np.zeros_like(dataY)
        return dataX[:,0], dataY[:,0], dataX2[:,0], dataY2[:,0]







