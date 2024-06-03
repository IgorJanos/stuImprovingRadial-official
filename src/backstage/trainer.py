#------------------------------------------------------------------------------
#
#   Improving Radial ...
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
import numpy as np
import torch
import datetime
import hydra
from hydra.core.hydra_config import HydraConfig

from .loggers import Loggers
from .dataset import FootballDataset
from .utils import getTqdm, Statistics, k2FromK1, get_k2_from_k1, clone_of
from .losses import MDLD, compute_penalty_factor, square_difference

from torch.utils.data import DataLoader
from torchvision import transforms



#------------------------------------------------------------------------------
#   DataSource class
#------------------------------------------------------------------------------

class DataSource:
    def __init__(
        self, 
        dataset_path, 
        subset, 
        shape, 
        batch_size, 
        num_workers, 
        shuffle=True
    ):
        self.ds = FootballDataset(dataset_path, subset, asTensor=True)

        self.transform_train = transforms.Compose([
            #transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize(shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.loader = DataLoader(self.ds, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers, pin_memory=True
        )
        # Iterable loader
        self.itLoader = None

    def get(self):
        if (self.itLoader is None):
            self.itLoader = iter(self.loader)

        # Loop indefinitely
        try:
            sample = next(self.itLoader)
        except (OSError, StopIteration):
            self.itLoader = iter(self.loader)
            sample = next(self.itLoader)

        return sample


#------------------------------------------------------------------------------
#   
#   Trainer class
#
#------------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg, output_folder = None):

        # CUDA / CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cfg = cfg

        # Setup run name & folder
        if not output_folder:
            self.outputFolder = HydraConfig.get().run.dir
        else:
            self.outputFolder = output_folder
            
        self.name = cfg.exp.id

        # Training params
        self.loss = "mdld"
        self.dp = True

        # Data sources
        self.shape = (cfg.model.input_shape[0], cfg.model.input_shape[1])

        self.dsTrain = DataSource(
            dataset_path=cfg.dataset.folder, 
            subset=cfg.dataset.train_set, 
            shape=self.shape,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.dataset.num_workers,
            shuffle=True
        )
        self.dsVal = DataSource(
            dataset_path=cfg.dataset.folder, 
            subset=cfg.dataset.val_set, 
            shape=self.shape,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.dataset.num_workers,
            shuffle=True
        )
        
        # Number of output coefficients
        self.nOutput = cfg.model.num_outputs

        # Model
        self.model = hydra.utils.instantiate(cfg.model)
        self.model = self.model.to(self.device)
        self.bestCheckpoint = None

        # Store original model parameters
        self.extractor_params = clone_of(self.model.get_extractor_params())

        # Optimizer
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            betas=cfg.training.betas
        )     
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.opt, gamma=cfg.training.lr_decay
        )

        self.log = Loggers()
        self.stats = Statistics([
            'lossT', 'lossV', 
            'mdldT', 'mdldV', 
            'compT', 'compV',
            'dlpT', 'dlpV',
            'exregT'
            ])

        # Loss functions
        self.criterionL2 = torch.nn.MSELoss(reduction="none")
        self.criterionMDLD = MDLD(shape=(640, 360), device=self.device)
        self.criterionBCE = torch.nn.BCEWithLogitsLoss(reduction="none")

        # Print out the sizes of our models
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Trainable model size: {params/1000000.0}M params")


    def load_checkpoint(self, filename):
        file_path = os.path.join(self.outputFolder, filename)

        # Load both model and optimizer
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        if (checkpoint is not None):
            self.model.load_state_dict(checkpoint["model"])
            self.opt.load_state_dict(checkpoint["opt"])

        # move to GPU
        self.model = self.model.to(self.device)


    def setup(self, loggers):
        os.makedirs(self.outputFolder, exist_ok=True)
        self.log = Loggers(loggers)


    def optimize(self):

        # Data paralell !
        if (self.dp):
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model

        self.log.trainingStart()
        for i in range(self.cfg.training.epochs):

            self.stats.reset()
            self.log.epochStart(i, self.stats)

            # Train & Validate
            self.trainingPass(i, model)
            self.validationPass(i, model, self.dsVal.loader)

            self.log.epochEnd(i, self.stats)

            self.scheduler.step()

        self.log.trainingEnd()
        

    def trainingPass(self, epoch, model):
        model.train()

        self.opt.zero_grad()

        progress = getTqdm(
            range(self.cfg.training.it_per_epoch * self.cfg.training.batches_per_it)
        )
        progress.set_description("Train {}".format(epoch+1))
        for it,_ in enumerate(progress):

            # Get next sample
            x, k = self.dsTrain.get()
            x = x.to(self.device)
            k = k.to(self.device)
            x = self.dsTrain.transform_train(x).contiguous()

            # Optimize one step
            k_hat, comp_hat, comp_target = model(x, k)
            k1 = k[:,0:1]
            k1_hat = k_hat[:,0:1]

            # MSE and MDLD losses
            if (self.nOutput == 1):
                loss_l2 = self.criterionL2(k_hat, k[:,0:1]) 
                loss_mdld = self.criterionMDLD(k2FromK1(k_hat), k)
                k2 = get_k2_from_k1(k1)
                k2_hat = k2FromK1(k1_hat)

            elif (self.nOutput == 2):
                loss_l2 = self.criterionL2(k_hat, k[:,0:2])
                loss_mdld = self.criterionMDLD(k_hat, k)
                k2 = k[:,1:2]
                k2_hat = k_hat[:,1:2]
            
            if (self.loss == "mdld"):
                # Disentangle sources of error
                mdld_k1 = self.criterionMDLD(torch.cat([k1_hat, k2], dim=1), torch.cat([k1, k2], dim=1))
                mdld_k2 = self.criterionMDLD(torch.cat([k1, k2_hat], dim=1), torch.cat([k1, k2], dim=1))

                final_loss = mdld_k1 + mdld_k2
            else:
                final_loss = loss_l2

            # Comparator loss
            if (self.cfg.training.lambda_comp > 0.0):
                lossComp = self.criterionBCE(comp_hat, comp_target)
                final_loss = final_loss + self.cfg.training.lambda_comp * lossComp
            else:
                lossComp = torch.zeros_like(final_loss)

            # Penalize inaccuracy of small values
            if (self.cfg.training.lambda_dlp > 0.0):
                lossDLP = loss_l2 * compute_penalty_factor(k[:,0:1], a=10.0)
                final_loss = final_loss + self.cfg.training.lambda_dlp * lossDLP
            else:
                lossDLP = torch.zeros_like(final_loss)

            # Extractor regularization
            if (self.cfg.training.lambda_exreg > 0.0):
                extractor_params = self.model.get_extractor_params()
                lossExReg = self.cfg.training.lambda_exreg * square_difference(extractor_params, self.extractor_params)
                final_loss = final_loss + lossExReg
            else:
                lossExReg = torch.zeros(size=(1,))
            
            # Normalize for accumulated steps
            final_loss = final_loss.mean() / self.cfg.training.batches_per_it           
            final_loss.backward()

            # Accumulate more gradient
            if (((it+1) % self.cfg.training.batches_per_it) == 0 or (it+1) == len(progress)):
                self.opt.step()
                self.opt.zero_grad()

            # Update stats
            self.stats.step("lossT", loss_l2.mean().cpu().detach().item())
            self.stats.step("mdldT", loss_mdld.mean().cpu().detach().item())
            self.stats.step("compT", lossComp.mean().cpu().detach().item())
            self.stats.step("dlpT", lossDLP.mean().cpu().detach().item())
            self.stats.step("exregT", lossExReg.cpu().detach().item())

            # Update progress info
            progress.set_postfix(self.stats.getAvg(["lossT", "mdldT", "compT", "dlpT", "exregT"]))



    def validationPass(self, epoch, model, data):
        model.eval()
        with torch.no_grad():            
            progress = getTqdm(data)
            progress.set_description("Val   {}".format(epoch+1))
            for (x, k) in progress:
                x = x.to(self.device)
                k = k.to(self.device)
                x = self.dsTrain.transform_val(x).contiguous()

                # Compute the prediction
                k_hat, comp_hat, comp_target = model(x, k)

                # Compute metrics
                if (self.nOutput == 1):
                    loss_l2 = self.criterionL2(k_hat, k[:,0:1])
                    mdld = self.criterionMDLD(k2FromK1(k_hat), k)
                elif (self.nOutput == 2):
                    loss_l2 = self.criterionL2(k_hat, k[:,0:2])
                    mdld = self.criterionMDLD(k_hat, k)
                lossComp = self.criterionBCE(comp_hat, comp_target)
                # Penalize inaccuracy of small values
                lossDLP = loss_l2 * compute_penalty_factor(k[:,0:1], a=10.0)

                # Update stats
                self.stats.step("lossV", loss_l2.mean().cpu().detach().item())
                self.stats.step("mdldV", mdld.mean().cpu().detach().item())
                self.stats.step("compV", lossComp.mean().cpu().detach().item())
                self.stats.step("dlpV", lossDLP.mean().cpu().detach().item())

                # Update progress info
                progress.set_postfix(self.stats.getAvg(["lossV", "mdldV", "compV", "dlpV"]))



