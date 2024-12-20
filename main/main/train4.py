import os, sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__), sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))
from libs import *

from data import ImageDataset
from models import *
from engines import train_fn

train_loaders = {
    "train":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "/workspace/yogiek/gc2/datasets/H-D/train/4/", 
            augment = True, 
        ), 
        num_workers = 8, batch_size = 32, 
        shuffle = True, 
    ), 
    "val":torch.utils.data.DataLoader(
        ImageDataset(
            data_dir = "/workspace/yogiek/gc2/datasets/H-D/valid/4/", 
            augment = False, 
        ), 
        num_workers = 2, batch_size = 32, 
        shuffle = False, 
    ), 
}

FT = torch.load(
    "/workspace/yogiek/gc2/warmup/ckps/H-D/PH2_val/best.ptl", 
    map_location = "cpu", 
)

for parameter in FT.parameters():
    parameter.requires_grad = False

model = {
    "FT":FT, 
    "FS":fcn_resnet50(), 
    "GS":fcn_3x64_gctx(), 
}

# model = fcn_resnet50(
#     num_classes = 2, 
# )
# optimizer = optim.AdamW(
#     model.parameters(), weight_decay = 5e-4, 
#     lr = 1e-4, 
# )
# scheduler = optim.lr_scheduler.StepLR(
#     optimizer, 
#     step_size = 40, gamma = 0.1, 
# )

save_ckps_dir = '/workspace/yogiek/gc2/warmup/ckps/H-D/PH2_val'

if not os.path.exists(save_ckps_dir):
    os.makedirs(save_ckps_dir)

train_fn(
    train_loaders, num_epochs = 500, 
    models = model, 
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
    save_ckps_dir = save_ckps_dir, 
)