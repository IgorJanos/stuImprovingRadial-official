#------------------------------------------------------------------------------
#
#   Improving Radial ...
#
#   Author : Igor Janos
#
#------------------------------------------------------------------------------

import os
from omegaconf import DictConfig, OmegaConf
import hydra


from backstage.trainer import Trainer
from backstage.loggers import WandBLogger, CsvLogger, ModelCheckpoint, \
    ResultSampler, PSNR_SSIM_Sampler, ErrorDistributionLogger



@hydra.main(version_base=None, config_path="config", config_name="default")
def do_train(cfg: DictConfig):

    # Load and execute training
    trainer = Trainer(cfg)

    loggers = [
        CsvLogger(trainer, "training.csv"),
        ModelCheckpoint(trainer, singleFile=True, bestName="mdldV"),
        ResultSampler(trainer, scaleShape=cfg.eval.shape, indices=cfg.eval.image_indices),
        ErrorDistributionLogger(trainer),
    ]

    if (cfg.exp.wandb):
        loggers += [ WandBLogger(trainer, OmegaConf.to_container(cfg), ".wandb.yaml") ]
    if (cfg.exp.psnr_ssim):
        loggers += [ PSNR_SSIM_Sampler(trainer) ]

    trainer.setup(loggers)

    # Go go
    trainer.optimize()



if __name__ == "__main__":
    do_train()

