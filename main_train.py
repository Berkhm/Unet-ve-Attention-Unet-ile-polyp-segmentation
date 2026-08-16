import torch

from dataset import get_data_loaders
from train import define_loss_and_optimizer, train_model

from model_unetpp import UnetPlusPlus
from model_attention_unet import AttentionUnet


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


MODEL_TYPE = "unetpp"

MODEL_SAVE_DIR = "/content/drive/MyDrive/Unet++ ve Attention Unet ile polyp segmentation"


train_loader, test_loader = get_data_loaders(
    batch_size=8,
    image_size=256
)


if MODEL_TYPE == "unetpp":

    model = UnetPlusPlus(
        in_chann=3,
        num_class=1
    ).to(device)

    model_name = "U-Net++"

    model_path = MODEL_SAVE_DIR + "/unetplusplus_polyp_60epoch.pth"


elif MODEL_TYPE == "attention":

    model = AttentionUnet(
        in_chann=3,
        num_class=1
    ).to(device)

    model_name = "Attention U-Net"

    model_path = MODEL_SAVE_DIR + "/attention_unet_polyp_60epoch.pth"



criterion, optimizer = define_loss_and_optimizer(
    model
)


train_model(
    model=model,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=60,
    model_name=model_name
)


torch.save(
    model.state_dict(),
    model_path
)


print("Model Drive'a kaydedildi:", model_path)