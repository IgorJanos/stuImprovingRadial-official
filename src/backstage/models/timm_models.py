
import torch
import torch.nn as nn

import numpy as np

import timm


#------------------------------------------------------------------------------
#
#   TimmDistortionModel
#
#------------------------------------------------------------------------------

class TimmDistortionModel(nn.Module):
    def __init__(
        self, 
        backbone, 
        num_outputs,
        num_hidden,
        num_compare,
        num_compare_outputs,
        drop_rate,
        input_shape     
    ):
        super().__init__()

        # Feature extractor
        extractor, num_features = TIMM_MODELS[backbone](backbone)
        self.extractor = nn.Sequential(
            extractor.layer0,
            extractor.layer1,
            extractor.layer2,
            extractor.layer3,
        )

        # Regressor head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(num_features, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_outputs)
        )

        # Comparator head
        layers_comparator = [
            nn.Dropout(drop_rate),
        ]
        num_in = num_features * 2
        if (num_compare > 0):
            layers_comparator += [
                nn.Linear(num_in, num_compare),
                nn.BatchNorm1d(num_compare),
                nn.SiLU(),
                nn.Linear(num_compare, num_compare_outputs)
            ]
        else:
            layers_comparator += [ 
                nn.Linear(num_in * 2, num_compare_outputs) 
            ]
        self.comparator = nn.Sequential(*layers_comparator)
        self.num_compare_outputs = num_compare_outputs


    def forward(self, x, k=None):
        # Extractor
        features = self.extractor(x)
        khat = self.regressor(features)

        if (k is None):
            return khat

        # Shuffle and compare
        f1, k_shuffled = shuffle_indices(features, k)
        comp_hat = self.compare(features, f1)

        if (self.num_compare_outputs == 1):
            comp_target = ((k_shuffled[:,0] > k[:,0]) * 1.0).unsqueeze(1)
        else:
            comp_target = torch.cat([
                ((k_shuffled[:,0] > k[:,0]) * 1.0).unsqueeze(1),
                ((k_shuffled[:,1] > k[:,1]) * 1.0).unsqueeze(1)
                ],
                dim=1
            )

        return khat, comp_hat, comp_target


    def compare(self, f0, f1):
        B = f0.size(0)
        f0 = f0.view(B,-1)
        f1 = f1.view(B,-1)
        f = torch.cat([f0, f1], dim=1)
        return self.comparator(f)

    def get_extractor_params(self):
        return self.extractor.parameters()





def shuffle_indices(x,y):
    n = np.arange(x.shape[0])
    np.random.shuffle(n)
    return x[n], y[n]


def _make_efficientnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, *model.blocks[0:2])
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained


def createTimmEfficientNet(backbone, globalAvgPool=False):

    model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
    pretrained = _make_efficientnet(model)
    numFeatures = 320 * (7*7)
    return pretrained, numFeatures


TIMM_MODELS = {
    "tf_efficientnet_lite0": createTimmEfficientNet,
}

