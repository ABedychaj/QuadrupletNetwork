from torch import nn


class Backbone(nn.Module):
    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


class EmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.embeddings = nn.Linear(512, 128)

    def forward(self, x):
        return self.embeddings(self.backbone(x))
