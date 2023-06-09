from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from vision_transformer.utils import AverageMeter, get_logger
import numpy as np

class Trainer:
    def __init__(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: Any,
        optimizer: Any,
        device: Any,
        save_dir: str,mixup_type : str,depth : int , alpha : Any
    ) -> None:
        self.epochs = epochs
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.mixup_type = mixup_type
        self.depth = depth
        self.alpha = alpha
        self.logger = get_logger(str(Path(self.save_dir).joinpath("log.txt")))
        self.best_loss = float("inf")
    
    def mixup_criterion(self , preds, y_a, y_b, lam):
        return lam * self.criterion(preds, y_a) + (1 - lam) * self.criterion(preds, y_b)
    
    def fit(self, model: nn.Module) -> None:
        for epoch in range(self.epochs):
            model.train()
            losses = AverageMeter("train_loss")

            with tqdm(self.train_loader, dynamic_ncols=True) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.epochs}]")

                for tr_data in pbar:
                    block_num = np.random.randint(0, self.depth)
                    lmbda = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
                    tr_X = tr_data[0].to(self.device)
                    tr_y = tr_data[1].to(self.device)
                    batch_size = tr_X.size()[0]
                    index = torch.randperm(batch_size)
                    self.optimizer.zero_grad()
                    out = model(tr_X , tr_X[index , :] , self.mixup_type , lmbda , block_num)
                    # CrossEntropy
                    if self.mixup_type == "none":
                        loss = self.criterion(out, tr_y)
                    else:
                        targetA , targetB = tr_y , tr_y[index]
                        loss = self.mixup_criterion(out , targetA , targetB , lmbda)
                    loss.backward()
                    self.optimizer.step()
                    losses.update(loss.item())
                    pbar.set_postfix(loss=losses.value)

            self.logger.info(f"(train) epoch: {epoch} loss: {losses.avg}")
            self.evaluate(model, epoch)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, epoch: int) -> None:
        model.eval()
        losses = AverageMeter("valid_loss")
        total = 0
        correct = 0
        for va_data in tqdm(self.valid_loader):
            va_X = va_data[0].to(self.device)
            va_y = va_data[1].to(self.device)

            out = model(va_X)
            loss = self.criterion(out, va_y)
            losses.update(loss.item())
            _, predicted = out.max(1)
            total += va_y.size(0)
            correct += predicted.eq(va_y).sum().item()
        acc = 100.*correct/total
        self.logger.info(f"(valid) epoch: {epoch} loss: {losses.avg} acc : {acc}")

        if losses.avg <= self.best_loss:
            self.best_acc = losses.avg
            torch.save(model.state_dict(), Path(self.save_dir).joinpath("best.pth"))
