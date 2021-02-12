"""Main experiments

(1) Data generation
(2) Train INVASE-GNN
(3) Evaluate INVASE on ground truth and prediction
"""

import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from utils import save_checkpoint

def main():

    # get dataset
    if args.task == 'mutag':
        get_data = 0

    # dataloaders


    # instantise model
    idx_details = f"{args.task}_r-{arg.run_id}_l-{args.lamda}_g-{args.n_layer}_t-{args.conv_type}_s-{seed}"

    # train


    # evaluate on test set


def train(model, optimizer, idx_details, loss, device, train_generator, val_generator, epochs):
    """Main training function (universal)
    """
    writer = SummaryWriter(
        log_dir=(f"logs/{idx_details}_{datetime.datetime.now():%d-%m-%Y_%H-%M-%S}"))
    checkpoint_file = f"models/checkpoint_{idx_details}.pth.tar"
    best_file = f"models/best_{idx_details}.pth.tar"

    best_loss, _ = model.evaluate(
        generator=val_generator,
        criterion=loss,
        optimizer=None,
        device=device,
        task="val"
    )

    # try, except to be able to stop training midway through
    try:
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = model.evaluate(
                generator=train_generator,
                criterion=loss,
                optimizer=optimizer,
                device=device,
                task="train"
            )

            # Validation
            with torch.no_grad():
                # evaluate on validation set
                val_loss, val_acc = model.evaluate(
                    generator=val_generator,
                    criterion=loss,
                    optimizer=None,
                    device=device,
                    task="val"
                )

            print("Epoch: [{}/{}]\n"
                "Train      : Loss {:.4f}\t Acc: {:.4f}\t"
                "Validation : Loss {:.4f}\t Acc: {:.4f}\t".format(
                    epoch+1, epochs,
                    train_loss, train_acc, val_loss, val_acc))

            # save model
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss

            checkpoint_dict = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_error": best_loss,
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint_dict,
                is_best,
                checkpoint_file,
                best_file)

            # write actor, critic and baseline loss/accuracy

            writer.add_scalar("loss/train", train_loss, epoch+1)
            writer.add_scalar("loss/validation", val_loss, epoch+1)
            writer.add_scalar("acc/train", train_acc, epoch+1)
            writer.add_scalar("acc/validation", val_acc, epoch+1)

    except KeyboardInterrupt:
        pass

def test():

    # initantise the model

    # load state
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    # eval
    test_acc, test_x, y_true, y_pred = model.evaluate(test_generator, loss, optimiser, device, task="test")
    print(test_acc, y_true[:10], y_pred[:10])


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(
        description="INVASE-GNN")

    parser.add_argument("--task",
                        type=str,
                        default="mutag",
                        metavar="PATH",
                        help="Task to perform (mutag, syn1)")

    parser.add_argument('--lamda',
                        type=float,
                        nargs='?',
                        default=0.1,
                        help='INVASE hyperparameter')

    parser.add_argument('--fea-dim',
                        type=int,
                        nargs='?',
                        default=32,
                        help='Node feature dimension')

    parser.add_argument('--actor_h_dim',
                        help='hidden state dimensions for actor',
                        default=100,
                        type=int)
    parser.add_argument('--critic_h_dim',
                        help='hidden state dimensions for critic',
                        default=200,
                        type=int)

    parser.add_argument('--conv-layers',
                        type=int,
                        nargs='?',
                        default=3,
                        help='Number of Graph Convolution layers')

    parser.add_argument('--conv-type',
                        type=str,
                        nargs='?',
                        default='GCN',
                        help='Type of Graph Convolution layers')

    parser.add_argument("--batch-size", "--bsize",
                        default=256,
                        type=int,
                        metavar="N",
                        help="mini-batch size (default: 256)")

    parser.add_argument("--seed",
                        default=0,
                        type=int,
                        metavar="N",
                        help="seed for random number generator")

    # optimiser inputs
    parser.add_argument("--epochs",
                        default=300,
                        type=int,
                        metavar="N",
                        help="number of total epochs to run")

    parser.add_argument("--loss",
                        default="BCE",
                        type=str,
                        metavar="str",
                        help="choose a Loss Function")

    parser.add_argument("--optim",
                        default="Adam",
                        type=str,
                        metavar="str",
                        help="choose an optimizer; SGD, Adam")

    parser.add_argument("--learning-rate", "--lr",
                        default=0.0001,
                        type=float,
                        metavar="float",
                        help="initial learning rate (default: 1e-4)")

    parser.add_argument("--run-id",
                        default=0,
                        type=int,
                        metavar="N",
                        help="experiment id")

    parser.add_argument("--disable-cuda",
                        action="store_true",
                        help="Disable CUDA")

    args = parser.parse_args(sys.argv[1:])

    args.device = torch.device("cuda") if (not args.disable_cuda) and  \
        torch.cuda.is_available() else torch.device("cpu")

    return args


if __name__ == "__main__":
    args = input_parser()

    print("The model will run on the {} device".format(args.device))

    main()