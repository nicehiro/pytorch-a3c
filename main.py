import argparse
import os
from shared_optim import SharedAdam
import torch
import gym
from agent import Worker, Player
from model import AC
from movan import Net


parser = argparse.ArgumentParser(description="PyTorch A3C Algorithm")

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
    "--shared-optimizer", default=True, help="use an optimizer without shared momentum"
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
    "--entropy-coef",
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

parser.add_argument("--episodes-n", type=int, default=3000, help="episode number")


if __name__ == "__main__":
    # not to use OMP threads in numpy process
    os.environ["OMP_NUM_THREADS"] = "1"

    args = parser.parse_args()

    # set torch seed
    torch.manual_seed(args.seed)

    env = gym.make(args.env_name)
    master_model = AC(env.observation_space.shape[0], env.action_space.n)
    master_model.share_memory()

    shared_adam = SharedAdam(master_model.parameters(), lr=args.lr)

    workers = [
        Worker(
            rank,
            args.seed,
            args.env_name,
            args.lr,
            AC,
            master_model,
            shared_adam,
            args.gamma,
            args.gae_lambda,
            args.entropy_coef,
            args.value_loss_coef,
            1,
            args.episodes_n,
        )
        for rank in range(args.num_process)
    ]

    for w in workers:
        w.start()

    # test
    player = Player(args.env_name, args.episodes_n, master_model)
    player.start()

    for w in workers:
        w.join()
    player.join()
