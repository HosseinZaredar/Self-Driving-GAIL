from env import CarlaEnv
from ppo import PPOAgent
from cnn_backbone import CNNBackbone

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import numpy as np
from distutils.util import strtobool
import argparse
import time
import random
from torch.utils.tensorboard import SummaryWriter


# command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppo-learning-rate', type=float, default=3e-4)
    parser.add_argument('--disc-learning-rate', type=float, default=6e-5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--total_timesteps', type=int, default=500_000)
    parser.add_argument('--use-cuda', type=lambda x: strtobool(x), nargs='?', default=False, const=True)
    parser.add_argument('--deterministic-cuda', type=lambda x: strtobool(x), nargs='?', default=True, const=True)
    parser.add_argument('--num-steps', type=int, default=64,
                        help='number of steps in each environment before training')
    parser.add_argument('--anneal-lr', type=lambda x: strtobool(x), nargs='?', default=False, const=True)
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=5, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")

    parser.add_argument("--num-disc-minibatches", type=int, default=4)
    parser.add_argument("--half-life", type=int, default=250)
    parser.add_argument("--wasserstein", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--grad-penalty", type=lambda x: bool(strtobool(x)), default=True)

    args = parser.parse_args()
    args.batch_size = args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args


class Discriminator(nn.Module):  # Discriminator Network

    # initializer
    def __init__(self, device, lr, state_dim, num_actions, wasserstein, grad_penalty):
        super(Discriminator, self).__init__()
        self.lr = lr
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = device
        self.checkpoint_file = 'disc'

        self.grad_penalty = grad_penalty
        self.wasserstein = wasserstein
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.cnn = CNNBackbone(n_channels=3)

        self.disc = nn.Sequential(
            nn.Linear(512+3+num_actions, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    # forward propagation
    def forward(self, state, command, action):
        state_features = self.cnn(state)
        disc_inp = torch.cat([state_features, command, action], dim=1)
        pred = self.disc(disc_inp)
        return pred

    # gradient penalty
    def compute_grad_pen(self, expert_states, expert_commands, expert_actions,
                         agent_states, agent_commands, agent_actions, lambda_=10):

        alpha = torch.rand(expert_states.size(0), 1)

        # Change state values
        exp_states = self.cnn(expert_states)
        agt_states = self.cnn(agent_states)

        expert_data = torch.cat([exp_states, expert_commands, expert_actions], dim=1)
        policy_data = torch.cat([agt_states, agent_commands, agent_actions], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data

        disc = self.disc(mixup_data)
        ones = torch.ones(disc.size()).to(self.device)
        grad = autograd.grad(outputs=disc, inputs=mixup_data, grad_outputs=ones,
                             create_graph=True, retain_graph=True, only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def learn(self, expert_states, expert_commands, expert_actions, agent_states, agent_commands, agent_actions):

        n_states = agent_states.shape[0]

        # sample expert data
        expert_states_sample, expert_commands_sample, expert_actions_sample = sample_expert(
            expert_states, expert_commands, expert_actions, n_states)

        # creating tensors from expert states and actions
        expert_states_sample = torch.from_numpy(expert_states_sample).float()
        expert_commands_sample = torch.from_numpy(expert_commands_sample).float()
        expert_actions_sample = torch.from_numpy(expert_actions_sample).float()

        # creating mini-batches
        disc_batch_size = n_states // args.num_disc_minibatches
        batch_starts = np.arange(0, n_states, disc_batch_size)

        mean_raw_loss = 0
        mean_loss = 0

        for b in batch_starts:
            expert_states_batch = expert_states_sample[b: b + disc_batch_size].to(self.device)
            expert_commands_batch = expert_commands_sample[b: b + disc_batch_size].to(self.device)
            expert_actions_batch = expert_actions_sample[b: b + disc_batch_size].to(self.device)
            agent_states_batch = agent_states[b: b + disc_batch_size].to(self.device)
            agent_commands_batch = agent_commands[b: b + disc_batch_size].to(self.device)
            agent_actions_batch = agent_actions[b: b + disc_batch_size].to(self.device)

            # calculating discriminator prediction
            agent_preds = torch.squeeze(
                self.forward(agent_states_batch, agent_commands_batch, agent_actions_batch), dim=-1)
            expert_preds = torch.squeeze(
                self.forward(expert_states_batch, expert_commands_batch, expert_actions_batch), dim=-1)

            if self.wasserstein:
                raw_disc_loss = -(expert_preds.mean() - agent_preds.mean())
            else:
                # real and fake labels (1s and 0s)
                real_labels = torch.ones(expert_states_batch.shape[0]).to(self.device)
                fake_labels = torch.zeros(agent_states_batch.shape[0]).to(self.device)

                # calculating the loss
                loss_real = self.bce_loss(expert_preds, real_labels)
                loss_fake = self.bce_loss(agent_preds, fake_labels)
                raw_disc_loss = loss_real + loss_fake

            if self.grad_penalty:
                grad_pen = self.compute_grad_pen(expert_states_batch, expert_commands_batch, expert_actions_batch,
                                                 agent_states_batch, agent_commands_batch, agent_actions_batch)
                disc_loss = raw_disc_loss + grad_pen
            else:
                disc_loss = raw_disc_loss

            # backpropagation and optimization
            self.optimizer.zero_grad()
            disc_loss.backward()
            self.optimizer.step()

            mean_raw_loss += raw_disc_loss.item()
            mean_loss += disc_loss.item()

        mean_raw_loss /= len(batch_starts)
        mean_loss /= len(batch_starts)

        return mean_raw_loss, mean_loss

    # generate rewards
    def generate_rewards(self, agent_states, agent_commands, agent_actions):
        with torch.no_grad():
            probs = torch.sigmoid(self.forward(agent_states, agent_commands, agent_actions))
            clamped_probs = torch.clamp(probs, min=0.01, max=0.99)
            # rewards = torch.log(clamped_probs) - torch.log(1 - clamped_probs)
            rewards = - torch.log(1 - clamped_probs)
        return rewards


def sample_expert(expert_states, expert_commands, expert_actions, num):
    indices = np.random.choice(np.arange(expert_states.shape[0]), num, replace=False)
    return expert_states[indices], expert_commands[indices], expert_actions[indices]


if __name__ == '__main__':

    args = parse_args()
    run_name = f'train-bc-gail_{args.seed}_{int(time.time())}'

    # tensorboard setup
    writer = SummaryWriter(os.path.join('runs', run_name))
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic_cuda

    # compute device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # carla env setup
    env = CarlaEnv()

    # load expert trajectories
    expert_states = []
    expert_commands = []
    expert_actions = []
    for dir in os.listdir('expert_data'):
        expert_states.append(np.load(os.path.join('expert_data', dir, 'expert_states.npy')))
        expert_commands.append(np.load(os.path.join('expert_data', dir, 'expert_commands.npy')))
        expert_actions.append(np.load(os.path.join('expert_data', dir, 'expert_actions.npy')))

    expert_states = np.concatenate(expert_states)
    expert_commands = np.concatenate(expert_commands)
    expert_actions = np.concatenate(expert_actions)

    # initialize discriminator
    disc = Discriminator(device, args.disc_learning_rate, env.observation_space, 3, args.wasserstein, args.grad_penalty)

    # initialize ppo agent
    agent = PPOAgent('bc_gail_learner', 3, args.ppo_learning_rate, env, device, args.num_steps, writer).to(device)

    # start the game
    global_step = 0
    num_updates = args.total_timesteps // args.batch_size

    # starting alpha
    alpha_0 = 0.5 ** (1 / args.half_life)

    for update in range(1, num_updates + 1):

        agent.save_models()

        # annealing the rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            agent.optimizer.param_groups[0]['lr'] = lrnow

        # rollout agent
        global_step = agent.rollout(global_step)

        # update discriminator
        disc_raw_loss, disc_loss = disc.learn(
            expert_states, expert_commands, expert_actions, agent.obs, agent.commands, agent.actions)

        # calculate rewards
        agent.rewards = disc.generate_rewards(agent.obs, agent.commands, agent.actions)

        # calculate episodic return (for debugging purposes)
        returns = agent.calc_returns(global_step - args.num_steps)
        for r in returns:
            writer.add_scalar("charts/episodic_return_disc", r[1], r[0])

        # calculate advantage
        agent.calc_advantage(args.gamma, args.gae_lambda)

        # calculating alpha
        t = (update * args.update_epochs * args.num_steps) / len(expert_states)
        alpha = alpha_0 ** t

        # agent learning
        v_loss, pg_loss, entropy_loss = agent.learn(
            args.batch_size, args.minibatch_size, args.update_epochs, args.norm_adv, args.clip_coef,
            args.clip_vloss, args.ent_coef, args.vf_coef, args.max_grad_norm, interpolate_bc=True,
            bc_alpha=alpha, bc_states=expert_states, bc_commands=expert_commands, bc_actions=expert_actions)

        # record rewards for plotting purposes
        writer.add_scalar("charts/alpha", alpha, global_step)
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/disc_raw", disc_raw_loss, global_step)
        writer.add_scalar("losses/disc_total", disc_loss, global_step)
