import torchvision.models as models
from torch import nn
from torch.optim import SGD
from torchvision.transforms.functional import to_tensor

from src.dataset.cifar import CIFAR100_Full, prepare_cifar
from src.hardmining import CustomBatchSampler
from src.model.Embedding import Backbone, EmbeddingModel
from src.training import train
from src.utils.logging import setup_logger
from src.utils.util import find_device

logger = setup_logger()
device = find_device()

logger.info("Using device: \"{}\"".format(device))

# prepare dataset
ds_train = CIFAR100_Full("/tmp/cf100", transform=to_tensor, target_transform=None, download=True)
ds_val = CIFAR100_Full("/tmp/cf100", transform=to_tensor, target_transform=None, download=True)

prepare_cifar(ds_train, 50, True)
prepare_cifar(ds_val, 50, False)

# prepare embedding model
resnet18 = models.resnet18(pretrained=False)
resnet18.fc = nn.Identity()

backbone = Backbone(resnet18)
embedding_model = EmbeddingModel(backbone)

opt = SGD(embedding_model.parameters(), lr=0.0005)

# Paper suggest M hard examples and 2M easier ones;
# batch size from experiment section is 128; we'll use 42+84=126 to have the first assumtion ready
dataloader_train = CustomBatchSampler(ds_train, mOHNM=True, batch_size=500,
                                      hard_batchs_size=42, norm_batchs_size=84,
                                      embedding_network=embedding_model)

accuracies_train, accuracies_val = train(logger=logger,
                                         ds_train=ds_train,
                                         ds_val=ds_val,
                                         epochs=50,
                                         opt=opt,
                                         model=embedding_model,
                                         dataloader=dataloader_train,
                                         device=device)

