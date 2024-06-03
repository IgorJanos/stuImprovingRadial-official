from typing import Callable
import torch
import numpy as np
import cv2

import matplotlib.pyplot as plt

from dataclasses import dataclass
from src.backstage.models.timm_models import TimmDistortionModel
from src.backstage.models.baseline import BaselineDistortionModel
from torchvision import transforms

from src.backstage.utils import undistort, get_distortion_map



class NumpyToTensor(Callable):
    def __call__(self, x):
        y = np.transpose(x, axes=(2,0,1))   # HWC -> CHW
        y = torch.from_numpy(y).float()
        y = y / 255.0
        return y                            # <0;1>


class PadWithWhite(Callable):
    def __init__(self, px, py):
        self.px = px
        self.py = py

    def __call__(self, x):
        H,W,C = x.shape
        result = np.ones(shape=(H+self.py, W+self.px, C), dtype=np.uint8) * 255
        result[:H,:W,:] = x
        return result



class Infer:
    def __init__(self, model, input_shape):

        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        self.input_shape = input_shape

        self.to_tensor = NumpyToTensor()
        self.transform = transforms.Compose([
            transforms.Resize(size=input_shape),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
            

    def predict_batch(
        self, 
        x: torch.Tensor
    ) -> np.ndarray :
        """ Predict a batch of images in format B;C;H;W and range <0;1> """
        x = self.transform(x)
        x = x.to(self.device)
        y = self._predict_both(x)
        return y.cpu().detach().numpy()

    def predict_single_image(
        self, 
        x: np.ndarray
    ) -> np.ndarray:
        """ Predict a single image in HWC, RGB-8bit format """
        x_batch = self.to_tensor(x).unsqueeze(0)
        return self.predict_batch(x_batch)[0]


    def load_image_from_file(self, file_path) -> np.ndarray:
        x = cv2.imread(file_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return x

    def resize_to_input(self, x: np.ndarray) -> np.ndarray:
        x = cv2.resize(x, self.input_shape, interpolation=cv2.INTER_AREA)
        return x

    def undistort(
        self, 
        x: np.ndarray, 
        k: np.ndarray
    ) -> np.ndarray:
        """ Undistort a HWC, RGB-8bit source image using the predicted K coefficients """
        scale_shape = (x.shape[1], x.shape[0])      # Width x Height
        return undistort(x, k, scale_shape)
    
    def undistort_using_map(
        self, 
        x: np.ndarray, 
        d: np.ndarray            
    ) -> np.ndarray:
        scale_shape = (x.shape[0], x.shape[1])
        h = x.shape[0]
        _, cx, cy, xx, yy = get_distortion_map(scale_shape, np.array([0.0, 0.0]))
        mapX = (d*xx*h + cx).astype(np.float32)
        mapY = (d*yy*h + cy).astype(np.float32)
        # Undistort image
        result = cv2.remap(x, mapX, mapY, cv2.INTER_LINEAR)
        return result

    def process_file(self, file_path, pad=PadWithWhite(16,16)):
        image = self.load_image_from_file(file_path)
        k = self.predict_single_image(self.resize_to_input(image))
        corrected = self.undistort(image, k)
        return pad(image), pad(corrected)

    def _predict_both(self, x):
        k_hat = torch.cat([
                self.model(x).unsqueeze(0),
                self.model(torch.flip(x, dims=(3,))).unsqueeze(0)
                ], dim=0)
        k_hat_flip = torch.mean(k_hat, dim=0)
        return k_hat_flip
    
    def get_distortion_map(
        self, 
        x: np.ndarray, 
        k: np.ndarray
    ) -> np.ndarray:
        """ Get a map of distortion ratio for every pixel """
        scale_shape = (x.shape[0], x.shape[1])      # Height x Width
        d, _, _, _, _ = get_distortion_map(scale_shape, k)
        return d
    
    def get_mdld(self, d1, d2):
        return np.abs(d1 - d2).mean()



    @staticmethod
    def from_checkpoint(file_path):
        
        @dataclass
        class ModelConfig:
            """ Config for the best training run """
            backbone: str = "tf_efficientnet_lite0"
            num_outputs: int = 2
            num_hidden: int = 24
            num_compare: int = 16
            num_compare_outputs: int = 1
            drop_rate: float = 0.0
            input_shape = (224, 224)

        cfg = ModelConfig()

        model = TimmDistortionModel(
            backbone=cfg.backbone,
            num_outputs=cfg.num_outputs,
            num_hidden=cfg.num_hidden,
            num_compare=cfg.num_compare,
            num_compare_outputs=cfg.num_compare_outputs,
            drop_rate=cfg.drop_rate,
            input_shape=cfg.input_shape
        )

        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model"])
        model.eval()

        return Infer(model, cfg.input_shape)

    @staticmethod
    def from_football360_checkpoint(file_path):
        model = BaselineDistortionModel(
            backbone="efficientnet_b5",
            num_outputs=1,
            num_hidden=256,
            drop_rate=0.0,
            hidden_act="relu",
            input_shape=(224,224)
        )
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return Infer(model, (224,224))



def showImage(x, figsize=(16,9)):
    H,W,C = x.shape
    fig = plt.figure(figsize=figsize, facecolor="white")
    plt.axis("off")
    plt.imshow(x)
    plt.show()
    plt.close()