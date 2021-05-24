import argparse
import os
import torch
import gym
from agent import Master
from model import AC


parser = argparse.ArgumentParser(description="PyTorch A3C Algorithm")

parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="learning rate (default: 0.0001)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--num-process",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 4)",
)
parser.add_argument(
    "--cuda", action="store_true", default=False, help="enables CUDA training"
)

parser.add_argument(
    "--env-name",
    type=str,
    default="CartPole-v0",
    help="environment to train on (default: CartPole-v0)",
)

parser.add_argument(
    "--shared_optimizer", default=False, help="use an optimizer without shared momentum"
)

parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="discount factor for rewards (default: 0.99)",
)

parser.add_argument(
    "--gae-lambda",
    type=float,
    default=1.0,
    help="lambda parameter for GAE (default: 1.0)",
)

parser.add_argument(
    "--entropy_coef",
    type=float,
    default=0.01,
    help="entropy term coefficient (default: 0.01)",
)

parser.add_argument(
    "--value-loss-coef",
    type=float,
    default=0.5,
    help="value loss coefficient (default: 0.5)",
)


if __name__ == "__main__":
    # not to use OMP threads in numpy process
    os.environ["OMP_NUM_THREADS"] = "1"

    args = parser.parse_args()

    # cuda settings
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # set torch seed
    torch.manual_seed(args.seed)

    # training
    agent = Master(
        args.num_process,
        args.seed,
        args.env_name,
        args.lr,
        AC,
        args.shared_optimizer,
        args.gamma,
        args.gae_lambda,
        args.entropy_coef,
        args.value_loss_coef,
    )
    agent.train()
    # testing
    agent.test()
