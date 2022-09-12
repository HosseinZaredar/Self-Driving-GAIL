from algo.cnn import CNN

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import numpy as np


class Discriminator(nn.Module):  # Discriminator Network

    # initializer
    def __init__(self, device, lr, state_dim, num_actions, wasserstein, grad_penalty, branched=False):
        super(Discriminator, self).__init__()
        self.lr = lr
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = device
        self.branched = branched

        self.grad_penalty = grad_penalty
        self.wasserstein = wasserstein
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.cnn = CNN(n_channels=9)
        if self.branched:
            self.disc = nn.Linear(512+1+num_actions, 3)
        else:
            self.disc = nn.Linear(512+3+1+num_actions, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.1)
        self.to(self.device)

        checkpoint_dir = os.path.join(Path(__file__).parent.parent, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_file = os.path.join(checkpoint_dir, f'bc_gail_disc')

    # forward propagation
    def forward(self, state, command, speed, action):
        cnn_out = self.cnn(state)
        if self.branched:
            disc_inp = torch.cat([cnn_out, speed, action], dim=1)
            preds = self.disc(disc_inp)
            pred = command[:, 0:1] * preds[:, 0:1] + \
                   command[:, 1:2] * preds[:, 1:2] + \
                   command[:, 2:3] * preds[:, 2:3]
        else:
            disc_inp = torch.cat([cnn_out, command, speed, action], dim=1)
            pred = self.disc(disc_inp)
        return pred

    # gradient penalty (only works with single-branched architecture)
    def compute_grad_pen(self, expert_states, expert_commands, expert_speeds, expert_actions,
                         agent_states, agent_commands, agent_speeds, agent_actions, lambda_=10):

        alpha = torch.rand(expert_states.size(0), 1)

        exp_cnn_out = self.cnn(expert_states)
        agt_cnn_out = self.cnn(agent_states)
        expert_data = torch.cat([exp_cnn_out, expert_commands, expert_speeds, expert_actions], dim=1)
        policy_data = torch.cat([agt_cnn_out, agent_commands, agent_speeds, agent_actions], dim=1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)
        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        disc = self.disc(mixup_data)

        ones = torch.ones(disc.size()).to(self.device)
        grad = autograd.grad(outputs=disc, inputs=mixup_data, grad_outputs=ones,
                             create_graph=True, retain_graph=True, only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def learn(self, disc_batch_size, expert_states, expert_commands, expert_speeds, expert_actions,
              agent_states, agent_commands, agent_speeds, agent_actions):

        n_states = agent_states.shape[0]

        # sample expert data
        expert_states_sample, expert_commands_sample, expert_speeds_sample, expert_actions_sample = sample_expert(
            expert_states, expert_commands, expert_speeds, expert_actions, n_states)

        # creating tensors from expert states and actions
        expert_states_sample = torch.from_numpy(expert_states_sample)
        expert_commands_sample = torch.from_numpy(expert_commands_sample)
        expert_speeds_sample = torch.from_numpy(expert_speeds_sample)
        expert_actions_sample = torch.from_numpy(expert_actions_sample)

        # creating mini-batches
        batch_starts = np.arange(0, n_states, disc_batch_size)

        mean_real_loss = 0
        mean_fake_loss = 0
        mean_raw_loss = 0
        mean_loss = 0

        for b in batch_starts:
            expert_states_batch = expert_states_sample[b: b + disc_batch_size].to(self.device)
            expert_commands_batch = expert_commands_sample[b: b + disc_batch_size].to(self.device)
            expert_speeds_batch = expert_speeds_sample[b: b + disc_batch_size].to(self.device)
            expert_actions_batch = expert_actions_sample[b: b + disc_batch_size].to(self.device)
            agent_states_batch = agent_states[b: b + disc_batch_size].to(self.device)
            agent_commands_batch = agent_commands[b: b + disc_batch_size].to(self.device)
            agent_speeds_batch = agent_speeds[b: b + disc_batch_size].to(self.device)
            agent_actions_batch = agent_actions[b: b + disc_batch_size].to(self.device)

            # calculating discriminator prediction
            agent_preds = torch.squeeze(
                self.forward(agent_states_batch, agent_commands_batch,
                             agent_speeds_batch, agent_actions_batch), dim=-1)

            expert_preds = torch.squeeze(
                self.forward(expert_states_batch, expert_commands_batch,
                             expert_speeds_batch, expert_actions_batch), dim=-1)

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
                grad_pen = self.compute_grad_pen(
                    expert_states_batch, expert_commands_batch, expert_speeds_batch, expert_actions_batch,
                    agent_states_batch, agent_speeds_batch, agent_commands_batch, agent_actions_batch)
                disc_loss = raw_disc_loss + grad_pen
            else:
                disc_loss = raw_disc_loss

            # backpropagation and optimization
            self.optimizer.zero_grad()
            disc_loss.backward()
            self.optimizer.step()

            mean_real_loss += loss_real.item() if not self.wasserstein else 0.0
            mean_fake_loss += loss_fake.item() if not self.wasserstein else 0.0
            mean_raw_loss += raw_disc_loss.item()
            mean_loss += disc_loss.item()

        mean_real_loss /= len(batch_starts)
        mean_fake_loss /= len(batch_starts)
        mean_raw_loss /= len(batch_starts)
        mean_loss /= len(batch_starts)

        return mean_real_loss, mean_fake_loss, mean_raw_loss, mean_loss

    # generate rewards
    def generate_rewards(self, agent_states, agent_commands, agent_speeds, agent_actions):
        with torch.no_grad():
            probs = torch.sigmoid(self.forward(agent_states, agent_commands, agent_speeds, agent_actions))
            clamped_probs = torch.clamp(probs, min=0.01, max=0.99)
            rewards = - torch.log(1 - clamped_probs)
        return rewards.squeeze()

    def save_models(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_models(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


def sample_expert(expert_states, expert_commands, expert_speeds, expert_actions, num):
    indices = np.random.choice(np.arange(expert_states.shape[0]), num, replace=False)
    return expert_states[indices], expert_commands[indices], expert_speeds, expert_actions[indices]