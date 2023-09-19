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
from tqdm.auto import tqdm, trange

from .datasets import Websites
from .models import AttUNet, R2AttUNet, R2UNet, UNet
from .utils import (
    Downscale,
    GaussianBlur,
    RandomBlur,
    iterate_dataloader,
    seed_everything,
)

MODELS = {"attunet": AttUNet, "r2attunet": R2AttUNet, "r2unet": R2UNet, "unet": UNet}


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
    parser.add_argument("--experiment", type=str, default="gaussianblur_residual_r2attunet")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--downscaler", type=str, default="gaussianblur"
    )  # "downscale" or "randomblur"
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", type=str, default="r2attunet")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-steps", type=float, default=10000)
    parser.add_argument("--log-interval", type=float, default=50)
    args = parser.parse_args()

    if args.downscaler == "downscale":
        downscaler = Downscale(factor=4, resample="bicubic")
    elif args.downscaler == "randomblur":
        downscaler = RandomBlur(kernel_size=[1, 3, 5, 7, 9], sigma=(0.1, 9.0))
    elif args.downscaler == "gaussianblur":
        downscaler = GaussianBlur(kernel_size=5, sigma=3.0)
    else:
        raise ValueError()

    train_transform = T.Compose(
        [
            T.RandomResizedCrop((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    eval_transform = T.Compose(
        [
            T.CenterCrop((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)

    model = MODELS[args.model](3, 3)
    optimizer = optim.AdamW(model.parameters(), args.learning_rate)

    initial_step = 0
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from)
        assert args.seed == checkpoint["seed"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        initial_step = checkpoint["step"]

    model = model.cuda()

    seed_everything(args.seed)
    current_datetime = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    run_path = os.path.join(args.logs_path, args.experiment, current_datetime)
    os.makedirs(run_path)
    copytree("src/", os.path.join(run_path, "src/"))
    writer = SummaryWriter(run_path)

    train_loss = 0.0
    best_valid_loss = None if args.resume_from is None else checkpoint["best_valid_loss"]
    train_batch = next(iter(train_dataloader))
    valid_batch = next(iter(valid_dataloader))
    test_batch = next(iter(test_dataloader))
    print("Training")
    for step in (
        pbar := trange(args.num_steps - initial_step, initial=initial_step, total=args.num_steps)
    ):
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

            inputs_grid = make_grid(train_batch[0][:10], nrow=5, normalize=True)
            targets_grid = make_grid(train_batch[1][:10], nrow=5, normalize=True)
            preds_grid = make_grid(train_preds[:10], nrow=5, normalize=True)
            writer.add_image("train/inputs", inputs_grid, step + 1)
            writer.add_image("train/targets", targets_grid, step + 1)
            writer.add_image("train/preds", preds_grid, step + 1)
            TF.to_pil_image(inputs_grid).save(os.path.join(run_path, "train_inputs.png"))
            TF.to_pil_image(targets_grid).save(os.path.join(run_path, "train_targets.png"))
            TF.to_pil_image(preds_grid).save(os.path.join(run_path, "train_preds.png"))

            inputs_grid = make_grid(valid_batch[0][:10], nrow=5, normalize=True)
            targets_grid = make_grid(valid_batch[1][:10], nrow=5, normalize=True)
            preds_grid = make_grid(valid_preds[:10], nrow=5, normalize=True)
            writer.add_image("valid/inputs", inputs_grid, step + 1)
            writer.add_image("valid/targets", targets_grid, step + 1)
            writer.add_image("valid/preds", preds_grid, step + 1)
            TF.to_pil_image(inputs_grid).save(os.path.join(run_path, "valid_inputs.png"))
            TF.to_pil_image(targets_grid).save(os.path.join(run_path, "valid_targets.png"))
            TF.to_pil_image(preds_grid).save(os.path.join(run_path, "valid_preds.png"))

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
        for batch in tqdm(test_dataloader):
            loss, _, _ = model_step(model, batch)
            test_loss += loss.item()
        _, _, test_preds = model_step(model, test_batch)

    test_loss /= len(test_dataloader)
    print("test loss (%.4f)" % test_loss)
    writer.add_scalar("test/loss", test_loss, args.num_steps)

    inputs_grid = make_grid(test_batch[0][:10], nrow=5, normalize=True)
    targets_grid = make_grid(test_batch[1][:10], nrow=5, normalize=True)
    preds_grid = make_grid(test_preds[:10], nrow=5, normalize=True)
    writer.add_image("test/inputs", inputs_grid, args.num_steps)
    writer.add_image("test/targets", targets_grid, args.num_steps)
    writer.add_image("test/pred", preds_grid, args.num_steps)
    TF.to_pil_image(inputs_grid).save(os.path.join(run_path, "test_inputs.png"))
    TF.to_pil_image(targets_grid).save(os.path.join(run_path, "test_targets.png"))
    TF.to_pil_image(preds_grid).save(os.path.join(run_path, "test_preds.png"))

    writer.close()
