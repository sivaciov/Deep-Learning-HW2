import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch.ao.quantization.pt2e.port_metadata_pass import logger

from .models import ClassificationLoss, load_model, save_model
from .utils import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using mps")
    else:
        device = torch.device("cpu")
        print("Using cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # set up model specific hyperparameters
    if model_name == "linear":
        num_epoch = 50
        lr = 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        #batch_size = 1000
    elif model_name == "mlp":
        num_epoch = 50
        lr = 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif model_name == "mlp_deep":
        num_epoch = 200
        lr = 0.001
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif model_name == "mlp_deep_residual":
        num_epoch = 500
        lr = 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)


    # load data
    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    # optimizer = ...

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        epoch_loss = 0

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # TODO: implement training step
            optimizer.zero_grad() # clear gradients

            # forward pass
            logits = model(img)
            #print(logits.shape)


            loss = loss_func(logits, label)
            epoch_loss += loss.item()

            # backward pass
            loss.backward()
            optimizer.step()

            # log training loss
            logger.add_scalar("train_loss", loss.item(), global_step)

            global_step += 1

        print(f"Epoch {epoch + 1:2d} / {num_epoch:2d}: train_loss={epoch_loss:.4f}")

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                # TODO: compute validation accuracy

                # get the predicted class
                logits = model(img)
                pred = logits.argmax(dim=1)
                acc = (pred == label).float().mean()
                metrics["val_acc"].append(acc)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        logger.add_scalar("val_accuracy", epoch_val_acc, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

        if epoch_val_acc > 0.82:
            break

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
