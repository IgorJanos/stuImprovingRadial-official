
import torch
import torch.nn as nn
import torchvision


#------------------------------------------------------------------------------
#
#   BaselineDistortionModel
#
#------------------------------------------------------------------------------


class BaselineDistortionModel(nn.Module):
    def __init__(
        self, 
        backbone, 
        num_outputs,
        num_hidden,
        drop_rate,
        input_shape,
        hidden_act
    ):
        super().__init__()

        self.input_shape = input_shape
        
        # Feature extractor
        self.extractor, num_features = BASELINE_MODELS[backbone](backbone)

        # Decide activation
        if hidden_act == "relu": activation_layer = nn.ReLU()
        elif hidden_act == "silu": activation_layer = nn.SiLU()
        else: activation_layer = nn.Identity()

        # Regressor head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(num_features, num_hidden),
            nn.BatchNorm1d(num_hidden),
            activation_layer,
            nn.Linear(num_hidden, num_outputs)
        )

        # No comparator head!


    def forward(self, x, k=None):
        # Extractor
        features = self.extractor(x)
        khat = self.regressor(features)

        if (k is None):
            return khat

        # We don't do comparing
        comp_target = torch.zeros_like(k)
        comp_hat = comp_target.clone()

        return khat, comp_hat, comp_target



    def get_extractor_params(self):
        return self.extractor.parameters()



def createDensenet(backbone, globalAvgPool=False):    
    if (backbone == "densenet121"):   model = torchvision.models.densenet121(pretrained=True)
    elif (backbone == "densenet161"): model = torchvision.models.densenet161(pretrained=True)
    elif (backbone == "densenet169"): model = torchvision.models.densenet169(pretrained=True)
    elif (backbone == "densenet201"): model = torchvision.models.densenet201(pretrained=True)
    else:
        return None, 0

    numFeatures = model.classifier.in_features
    model.classifier = nn.Sequential()
    return model, numFeatures

def createResnet(backbone, globalAvgPool=False):
    if (backbone == "resnet18"): model = torchvision.models.resnet18(pretrained=True)
    elif (backbone == "resnet34"): model = torchvision.models.resnet34(pretrained=True)
    elif (backbone == "resnet50"): model = torchvision.models.resnet50(pretrained=True)
    elif (backbone == "resnet101"): model = torchvision.models.resnet101(pretrained=True)
    elif (backbone == "resnet152"): model = torchvision.models.resnet152(pretrained=True)
    else:
        return None, 0

    numFeatures = model.fc.in_features
    model.fc = nn.Sequential()
    return model, numFeatures

def createEfficientnet(backbone, globalAvgPool=False):
    if (backbone == "efficientnet_b0"): model = torchvision.models.efficientnet_b0(pretrained=True)
    elif (backbone == "efficientnet_b1"): model = torchvision.models.efficientnet_b1(pretrained=True)
    elif (backbone == "efficientnet_b2"): model = torchvision.models.efficientnet_b2(pretrained=True)
    elif (backbone == "efficientnet_b3"): model = torchvision.models.efficientnet_b3(pretrained=True)
    elif (backbone == "efficientnet_b4"): model = torchvision.models.efficientnet_b4(pretrained=True)
    elif (backbone == "efficientnet_b5"): model = torchvision.models.efficientnet_b5(pretrained=True)
    elif (backbone == "efficientnet_b6"): model = torchvision.models.efficientnet_b6(pretrained=True)
    elif (backbone == "efficientnet_b7"): model = torchvision.models.efficientnet_b7(pretrained=True)
    else:
        return None, 0

    if (globalAvgPool):
        extractor = nn.Sequential(model.features, model.avgpool)
        numFeatures = model.classifier[1].in_features
    else:
        extractor = model.features
        numFeatures = model.classifier[1].in_features * (7*7)

    return extractor, numFeatures



BASELINE_MODELS = {
    "densenet121": createDensenet,
    "densenet161": createDensenet,
    "densenet169": createDensenet,
    "densenet201": createDensenet,

    "resnet18": createResnet,
    "resnet34": createResnet,
    "resnet50": createResnet,
    "resnet101": createResnet,
    "resnet152": createResnet,

    "efficientnet_b0": createEfficientnet,
    "efficientnet_b1": createEfficientnet,
    "efficientnet_b2": createEfficientnet,
    "efficientnet_b3": createEfficientnet,
    "efficientnet_b4": createEfficientnet,
    "efficientnet_b5": createEfficientnet,
    "efficientnet_b6": createEfficientnet,
    "efficientnet_b7": createEfficientnet,
}



