import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from vision_transformer import Trainer , ViTData
from vision_transformer.models.vit_model import ViT
from vision_transformer.utils import Cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training"
    )
    parser.add_argument("--mixup_type",help="Mixup type to choose" ,choices=["none" , "attention" , "key_and_value" , "query"])
    parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
    return parser.parse_args()


def main():
    args = make_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_file = args.config_file
    config_file_dir = str(Path(config_file).resolve().parent)
    config = Cfg(config_file)
    dataset_params = config.get_params(type="dataset")
    dataloader_params = config.get_params(type="dataloader")
    model_params = config.get_params(type="model")
    learning_params = config.get_params(type="learning")

    data = ViTData(name=dataset_params["name"], save_dir=dataset_params["dir"])
    train_loader, valid_loader = data.create_dataloader(
        batch_size=dataloader_params["batch_size"], shuffle=dataloader_params["shuffle"]
    )

    model = ViT(
        in_chans=dataset_params["in_channels"],
        n_classes=dataset_params["num_classes"],
        embed_dim=model_params["embed_dim"],
        patch_size=model_params["patch_size"],
        img_size=dataset_params["image_size"],
        depth=model_params["num_blocks"],
        n_heads=model_params["heads"],
    )
    model.to(device)
    MIXUP_TYPE = args.mixup_type
    DEPTH = model_params["num_blocks"]
    alpha_lm = args.alpha

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=learning_params["learning_rate"], momentum=0.9
    )
    trainer = Trainer(
        epochs=learning_params["epochs"],
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=config_file_dir, mixup_type=MIXUP_TYPE , depth = DEPTH , alpha=alpha_lm
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
