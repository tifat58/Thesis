import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
from datasets.idrid_datasets import IDRIDDataset

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


idrid_norm_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize(512, 512),
                                           transforms.Normalize(mean = [0.4557, 0.2588, 0.1325],
                                                                std = [0.2856, 0.1855, 0.1356])
                                     ])

data_dir = '/mnt/sda/haal02-data/IDRID/'

id_train_set = IDRIDDataset(data_dir=data_dir, length=100, train=True, transform=idrid_norm_transform)
id_test_set = IDRIDDataset(data_dir=data_dir, length=100, train=False, transform=idrid_norm_transform)

id_train_loader = torch.utils.data.DataLoader(id_train_set, batch_size=32, shuffle=True)
id_test_loader = torch.utils.data.DataLoader(id_test_set, batch_size=32, shuffle=False)

print("working")

for data in id_train_loader:
    img, lbl = data
    print(img.shape, lbl)