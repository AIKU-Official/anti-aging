import os
from argparse import ArgumentParser
from datetime import datetime
from shutil import copytree
from typing import Tuple

import torch
from pytz import timezone
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
from tqdm import tqdm
from tqdm.auto import trange

from .datasets import Websites
from .models import AttUNet, R2AttUNet, R2UNet, UNet
from .utils import Downscale, iterate_dataloader, seed_everything

MODELS = {"attunet": AttUNet, "r2attunet": R2AttUNet, "r2unet": R2UNet, "unet": UNet}


def eval_collate(batch: list[tuple[Tensor, Tensor]]):
    inputs, targets = zip(*batch)
    return torch.cat(inputs), torch.cat(targets)


def model_step(model: nn.Module, batch: tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    inputs, targets = batch
    inputs, targets = inputs.cuda(), targets.cuda()
    preds = torch.tanh(model(inputs))
    loss = F.mse_loss(preds, targets - inputs)
    return loss, targets.cpu(), (inputs + preds).cpu()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./data/")
    parser.add_argument("--logs-path", type=str, default="./logs/")
    parser.add_argument("--experiment", type=str, default="unet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--downscaler", type=str, default="blur_5_0.1_5.0")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-steps", type=float, default=10000)
    parser.add_argument("--log-interval", type=float, default=100)
    args = parser.parse_args()

    if args.downscaler.startswith("downscale"):
        splits = args.downscaler.split("_")
        factor = int(splits[1])
        resample = splits[2]
        downscaler = Downscale(factor, resample)
    elif args.downscaler.startswith("blur"):
        splits = args.downscaler.split("_")
        kernel_size = int(splits[1])
        sigma = tuple(map(float, splits[2:4]))
        downscaler = T.GaussianBlur(kernel_size, sigma)
    else:
        raise ValueError()

    train_transform = T.Compose(
        [
            T.RandomResizedCrop((64, 64)),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    eval_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.FiveCrop((64, 64)),
            T.Lambda(lambda crops: [TF.resize(crop, (256, 256)) for crop in crops]),
            T.Lambda(lambda crops: torch.stack(crops)),
        ]
    )

    train_dataset = Websites(
        args.data_path, split="train", downscaler=downscaler, transform=train_transform
    )
    valid_dataset = Websites(
        args.data_path, split="valid", downscaler=downscaler, transform=eval_transform
    )
    test_dataset = Websites(
        args.data_path, split="test", downscaler=downscaler, transform=eval_transform
    )

    train_dataloader = DataLoader(
        train_dataset, args.train_batch_size, shuffle=True, pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_collate,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, args.eval_batch_size, shuffle=False, collate_fn=eval_collate, pin_memory=True
    )
    train_batch = next(iter(train_dataloader))
    valid_batch = next(iter(valid_dataloader))
    test_batch = next(iter(test_dataloader))

    model = MODELS[args.model](3, 3).cuda()
    optimizer = optim.AdamW(model.parameters(), args.learning_rate)

    seed_everything(args.seed)
    current_datetime = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    run_path = os.path.join(args.logs_path, args.experiment, current_datetime)
    os.makedirs(run_path)
    copytree("src/", os.path.join(run_path, "src/"))
    writer = SummaryWriter(run_path)

    train_loss, best_valid_loss = 0.0, None
    print("Training")
    for step in (pbar := trange(args.num_steps)):
        model.train()
        batch = next(iterate_dataloader(train_dataloader))
        loss, _, _ = model_step(model, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (step + 1) % args.log_interval == 0:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch in valid_dataloader:
                    loss, _, _ = model_step(model, batch)
                    valid_loss += loss.item()

                _, _, train_preds = model_step(model, train_batch)
                _, _, valid_preds = model_step(model, valid_batch)

            train_loss /= args.log_interval
            valid_loss /= len(valid_dataloader)
            pbar.set_description("train loss (%.4f) valid loss (%.4f)" % (train_loss, valid_loss))
            writer.add_scalar("train/loss", train_loss, step + 1)
            writer.add_scalar("valid/loss", valid_loss, step + 1)

            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                checkpoint = {
                    "seed": args.seed,
                    "step": step + 1,
                    "best_valid_loss": best_valid_loss,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(run_path, "best.pth"))

            train_inputs = [TF.resize(train_input, (64, 64)) for train_input in train_batch[0]]
            train_targets = [TF.resize(train_target, (64, 64)) for train_target in train_batch[1]]
            train_preds = [TF.resize(train_pred, (64, 64)) for train_pred in train_preds]
            inputs_grid = make_grid(train_inputs, nrow=5, padding=4, normalize=True)
            targets_grid = make_grid(train_targets, nrow=5, padding=4, normalize=True)
            preds_grid = make_grid(train_preds, nrow=5, padding=4, normalize=True)
            writer.add_image("train/inputs", inputs_grid, step + 1)
            writer.add_image("train/targets", targets_grid, step + 1)
            writer.add_image("train/preds", preds_grid, step + 1)

            valid_inputs = [TF.resize(valid_input, (64, 64)) for valid_input in valid_batch[0]]
            valid_targets = [TF.resize(valid_target, (64, 64)) for valid_target in valid_batch[1]]
            valid_preds = [TF.resize(valid_pred, (64, 64)) for valid_pred in valid_preds]
            inputs_grid = make_grid(valid_inputs, nrow=5, padding=4, normalize=True)
            targets_grid = make_grid(valid_targets, nrow=5, padding=4, normalize=True)
            preds_grid = make_grid(valid_preds, nrow=5, padding=4, normalize=True)
            writer.add_image("valid/inputs", inputs_grid, step + 1)
            writer.add_image("valid/targets", targets_grid, step + 1)
            writer.add_image("valid/preds", preds_grid, step + 1)

            train_loss = 0.0

    checkpoint = {
        "seed": args.seed,
        "step": args.num_steps,
        "best_valid_loss": best_valid_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(run_path, "last.pth"))

    model.eval()
    test_loss = 0.0
    print("Testing")
    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader):
            loss, _, _ = model_step(model, inputs, targets)
            test_loss += loss.item()
        _, _, test_preds = model_step(model, test_batch)

    test_loss /= len(test_dataloader)
    print("test loss (%.4f)" % test_loss)
    writer.add_scalar("test/loss", test_loss, args.num_steps)

    test_inputs = [TF.resize(test_input, (64, 64)) for test_input in test_batch[0]]
    test_targets = [TF.resize(test_target, (64, 64)) for test_target in test_batch[1]]
    test_preds = [TF.resize(test_pred, (64, 64)) for test_pred in test_preds]
    inputs_grid = make_grid(test_inputs, nrow=5, padding=4, normalize=True)
    targets_grid = make_grid(test_targets, nrow=5, padding=4, normalize=True)
    preds_grid = make_grid(test_targets, nrow=5, padding=4, normalize=True)
    writer.add_image("test/inputs", inputs_grid, args.num_steps)
    writer.add_image("test/targets", targets_grid, args.num_steps)
    writer.add_image("test/pred", preds_grid, args.num_steps)

    writer.close()
