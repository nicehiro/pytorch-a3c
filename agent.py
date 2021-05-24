import torch
import gym
from torch._C import dtype
from model import AC
import torch.nn.functional as F
import torch.optim as optim
import time
import torch.multiprocessing as mp
from shared_optim import SharedAdam


class Worker:
    def __init__(
        self,
        rank,
        seed,
        env_name,
        lr,
        model,
        master_model,
        optimizer,
        gamma,
        gae_lambda,
        entropy_coef,
        value_loss_coef,
        max_grad_norm,
        episodes_n=20,
    ) -> None:
        """Worker execute gae-ac algorithm.

        Train in the worker model, get the gradients, then send to
        the master model and optimize master model.
        """
        torch.manual_seed(seed + rank)
        self.env = gym.make(env_name)
        self.env.seed(seed + rank)
        # worker model
        features_n = self.env.observation_space.shape[0]
        actions_n = self.env.action_space.n
        self.model = model(features_n, actions_n)
        # optimizer for master model
        self.lr = lr
        self.optimizer = (
            torch.optim.Adam(master_model.parameters(), lr=self.lr)
            if not optimizer
            else optimizer
        )
        self.master_model = master_model
        # training settings
        self.episodes_n = episodes_n
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

    def train(self):
        """Training process using A3C."""
        self.model.train()
        while True:
            # synchronize thread-specific parameters
            self.model.load_state_dict(self.master_model.state_dict())
            s = self.env.reset()
            d = True

            steps = 0
            entropies = []
            q_values = []
            log_prob_of_as = []
            rewards = []

            for step in range(self.episodes_n):
                s = torch.from_numpy(s).type(torch.FloatTensor)
                q, logit = self.model(s.unsqueeze(0))
                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)
                # entropy calc in discrete action space
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)

                # sample action from action probs
                a = prob.multinomial(num_samples=1).detach()
                # save log prob of a
                log_prob_of_a = log_prob.gather(1, a)

                # take an action
                s_, r, d, _ = self.env.step(a.item())
                # reward sharping
                r = max(min(r, 1), -1)

                # save q, log prob ad reward of current action a
                q_values.append(q)
                log_prob_of_as.append(log_prob_of_a)
                rewards.append(r)

                s = s_
                if d:
                    break

            # for terminal s_{t}, R = 0
            R = torch.zeros(1, 1)
            # for non-terminal s_{t}, bootstrap from last state
            if not d:
                s = torch.from_numpy(s).type(torch.FloatTensor)
                q, _ = self.model(s.unsqueeze(0))
                R = q.detach()
            q_values.append(R)

            # online algorithm, calc loss
            policy_loss = value_loss = 0
            gae = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                R = self.gamma * R + rewards[i]
                advantage = R - q_values[i]
                # v_loss = 1/2 * advantage ** 2
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # generalized advantage estimation
                delta_t = rewards[i] + self.gamma * q_values[i + 1] - q_values[i]
                gae = gae * self.gamma * self.gae_lambda + delta_t

                policy_loss = (
                    policy_loss
                    - log_prob_of_as[i] * gae.detach()
                    - self.entropy_coef * entropies[i]
                )

            # calc grads of worker process
            self.optimizer.zero_grad()
            (policy_loss + self.value_loss_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            # push workers' grads to master model
            self.ensure_master_grads()
            # update master model
            self.optimizer.step()

    def ensure_master_grads(self):
        """Ensure worker's grads are pushed to master model."""
        for worker_params, master_params in zip(
            self.model.parameters(), self.master_model.parameters()
        ):
            if master_params.grad is not None:
                # TODO: this is not right...
                return
            master_params.grad = worker_params.grad

    def test(self):
        """Test current model."""
        self.model.eval()

        while True:
            time.sleep(60)

            s = self.env.reset()
            s = torch.from_numpy(s).type(torch.FloatTensor)
            reward_sum = 0
            times = 0
            start_time = time.time()

            for _ in range(self.episodes_n):
                times += 1
                with torch.no_grad():
                    q, logit = self.model(s.unsqueeze(0))
                prob = F.softmax(logit, dim=-1)
                a = prob.max(1, keepdim=True)[1].numpy()
                s_, r, d, _ = self.env.step(a[0, 0])
                d = d
                reward_sum += r
                s = s_

                if d:
                    break

            print(
                "Time {}, Steps {}, Total Reward {},".format(
                    start_time, times, reward_sum
                )
            )


class Master:
    def __init__(
        self,
        process_n,
        seed,
        env_name,
        lr,
        model,
        shared_optimizer,
        gamma,
        gae_lambda,
        entropy_coef,
        value_loss_coef,
    ) -> None:
        self.process_n = process_n
        self.seed = seed
        self.env_name = env_name
        self.env = gym.make(env_name)

        self.worker_model = model

        # master model
        self.model = model(self.env.observation_space.shape[0], self.env.action_space.n)

        # shared optimizer
        if shared_optimizer:
            self.optimizer = SharedAdam(self.model.parameters(), lr=lr)
            self.optimizer.share_memory()
        else:
            self.optimizer = None

        # hyper parameters setting
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_norm_grad = 1
        self.episodes_n = 20

        # container for worker process
        self.workers = []

    def train(self):
        """Train model."""
        lock = mp.Lock()
        for rank in range(self.process_n):
            worker = Worker(
                rank,
                self.seed,
                self.env_name,
                self.lr,
                self.worker_model,
                self.model,
                self.optimizer,
                self.gamma,
                self.gae_lambda,
                self.entropy_coef,
                self.value_loss_coef,
                self.max_norm_grad,
                self.episodes_n,
            )
            p = mp.Process(target=worker.train, args=())
            p.start()
            self.workers.append(p)

        for p in self.workers:
            p.join()

    def test(self):
        """Test model."""
        worker = Worker(
            10,
            self.seed,
            self.env_name,
            self.lr,
            self.worker_model,
            self.model,
            self.optimizer,
            self.gamma,
            self.gae_lambda,
            self.entropy_coef,
            self.value_loss_coef,
            self.max_norm_grad,
            self.episodes_n,
        )
        p = mp.Process(target=worker.test, args=())
        p.start()
        p.join()
