"""Main experiments

(1) Data generation
(2) Train INVASE-GNN
(3) Evaluate on ground truth and prediction
"""

import sys
import argparse
import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from invase_gnn import InvaseGNN
from utils import save_checkpoint, load_previous_state
from data_processing import mutag_data

def main():

    # get dataset
    if args.task == 'mutag':
        train_dataset, val_dataset, test_dataset = mutag_data(args.seed, args.val_size, args.test_size)
        fea_dim = train_dataset.num_features
        label_dim = train_dataset.num_classes
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # TODO synthetic task
    else:
         NameError(f"task {args.task} not allowed")

    # instantise model
    idx_details = f"{args.model_type}_{args.task}_r-{args.run_id}_ln-{args.node_lamda}_lf-{args.fea_lamda}_g-{args.n_layer}_s-{args.seed}"
    if args.model_type == "INVASE":
        model = InvaseGNN(fea_dim, label_dim, args.actor_h_dim, args.critic_h_dim, args.n_layer, args.node_lamda, args.fea_lamda)
    elif args.model_type == "GAT":
        model = GAT()
    else:
        NameError(f"model type {args.model_type} not allowed")
    model.to(args.device)
    # train
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)

    if not args.evaluate:
        train(model, optimizer, idx_details, loss, args.device, train_loader, val_loader, args.epochs)
    
    model, _, _, _, saved_args = load_previous_state(f"models/checkpoint_{idx_details}.pth.tar", model, args.device, optimizer)
    # evaluate on test set
    critic_test_acc, baseline_test_acc, x_test, \
                    selected_features, selected_nodes, y_trues, y_preds = model.evaluate(test_loader, loss, optimizer, args.device, task="test")
    # get summary results on selected features. 
    
    # save results
    print("TEST")
    print("--------")
    print("Critic Acc {:.3f}\t Baseline Acc {:.3f}".format(critic_test_acc, baseline_test_acc))

    print("example:", x_test[0], selected_features[0], selected_nodes[0], y_trues[0], y_preds[0])


def train(model, optimizer, idx_details, loss, device, train_generator, val_generator, epochs):
    """Main training function (universal)
    """
    writer = SummaryWriter(
        log_dir=(f"logs/{idx_details}_{datetime.datetime.now():%d-%m-%Y_%H-%M-%S}"))
    checkpoint_file = f"models/checkpoint_{idx_details}.pth.tar"
    best_file = f"models/best_{idx_details}.pth.tar"

    _, best_critic_acc, _, _, _ = model.evaluate(
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
            train_actor_loss, train_critic_acc, train_baseline_acc, \
                train_prop_of_feas, train_prop_of_nodes = model.evaluate(
                generator=train_generator,
                criterion=loss,
                optimizer=optimizer,
                device=device,
                task="train"
            )

            # Validation
            with torch.no_grad():
                # evaluate on validation set
                val_actor_loss, val_critic_acc, val_baseline_acc, \
                    val_prop_of_feas, val_prop_of_nodes  = model.evaluate(
                    generator=val_generator,
                    criterion=loss,
                    optimizer=None,
                    device=device,
                    task="val"
                )
            # print actor, critic and baseline accuracy as in INVASE
            if epoch % 10 == 0:
                print("Epoch: [{}/{}]\n"
                  "Train      : Actor Loss {:.4f}\t"
                  "Critic Acc {:.3f}\t Baseline Acc {:.3f}\n"
                  "Validation : Actor Loss {:.4f}\t"
                  "Critic Acc {:.3f}\t Baseline Acc {:.3f}\n"
                  "Prop of Nodes chosen: {:.3f}\t Prop of Features: {:3f}\n".format(
                    epoch+1, epochs,
                    train_actor_loss, train_critic_acc, train_baseline_acc,
                    val_actor_loss, val_critic_acc, val_baseline_acc,
                    val_prop_of_nodes, val_prop_of_feas))

            # save model
            is_best = val_critic_acc < best_critic_acc
            if is_best:
                best_critic_acc = val_critic_acc

            # add seperate accuracy/loss metrics
            checkpoint_dict = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc": best_critic_acc,
                "optimizer": optimizer.state_dict(),
                "args": args
            }
            save_checkpoint(checkpoint_dict,
                is_best,
                checkpoint_file,
                best_file)

            # write metrics to tensorboard
            writer.add_scalar("actor_loss/train", train_actor_loss, epoch+1)
            writer.add_scalar("critic_acc/train", train_critic_acc, epoch+1)
            writer.add_scalar("baseline_acc/train", train_baseline_acc, epoch+1)
            writer.add_scalar("prop_of_feas/training", train_prop_of_feas, epoch+1)
            writer.add_scalar("prop_of_nodes/training", train_prop_of_nodes, epoch+1)
            writer.add_scalar("actor_loss/validation", val_actor_loss, epoch+1)
            writer.add_scalar("critic_acc/validation", val_critic_acc, epoch+1)
            writer.add_scalar("baseline_acc/validation", val_baseline_acc, epoch+1)
            writer.add_scalar("prop_of_feas/validation", val_prop_of_feas, epoch+1)
            writer.add_scalar("prop_of_nodes/validation", val_prop_of_nodes, epoch+1)

    except KeyboardInterrupt:
        pass


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

    parser.add_argument('--node-lamda',
                        type=float,
                        nargs='?',
                        default=0.1,
                        help='INVASE hyperparameter for nodes')

    parser.add_argument('--fea-lamda',
                        type=float,
                        nargs='?',
                        default=0.1,
                        help='INVASE hyperparameter for features')

    parser.add_argument('--actor-h-dim',
                        help='hidden state dimensions for actor',
                        default=128,
                        type=int)

    parser.add_argument('--critic-h-dim',
                        help='hidden state dimensions for critic',
                        default=256,
                        type=int)

    parser.add_argument('--n-layer',
                        type=int,
                        nargs='?',
                        default=3,
                        help='Number of Graph Convolution layers')

    parser.add_argument('--model-type',
                        type=str,
                        nargs='?',
                        default='INVASE',
                        help='Type of model to evaluate (INVASE, GAT)')

    parser.add_argument("--batch-size", "--bsize",
                        default=128,
                        type=int,
                        metavar="N",
                        help="mini-batch size (default: 256)")

    parser.add_argument("--seed",
                        default=0,
                        type=int,
                        metavar="N",
                        help="seed for random number generator")

    parser.add_argument("--val-size",
                        default=0.1,
                        type=float,
                        help="Proportion of data to use for validation")

    parser.add_argument("--test-size",
                        default=0.1,
                        type=float,
                        help="Proportion of data to use for testing")

    # optimiser inputs
    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        metavar="N",
                        help="number of total epochs to run")

    parser.add_argument("--loss",
                        default="CE",
                        type=str,
                        metavar="str",
                        help="choose a Loss Function")

    parser.add_argument("--optim",
                        default="Adam",
                        type=str,
                        metavar="str",
                        help="choose an optimizer; SGD, Adam")

    parser.add_argument("--learning-rate", "--lr",
                        default=0.001,
                        type=float,
                        metavar="float",
                        help="initial learning rate (default: 1e-4)")

    parser.add_argument("--l2",
                        default=0.0,
                        type=float,
                        metavar="float",
                        help="L2 Regularisation")

    parser.add_argument("--run-id",
                        default=0,
                        type=int,
                        metavar="N",
                        help="experiment id")

    parser.add_argument("--evaluate",
                        action="store_true",
                        help="DSkip training and just evaluate on test set")

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