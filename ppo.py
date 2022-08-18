from cnn_backbone import CNNBackbone

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.optim as optim


# layer initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, env, num_actions, branched=False):
        super(Actor, self).__init__()
        self.branched = branched
        if branched:
            self.actor_mean = layer_init(nn.Linear(512 + 1, 3 * num_actions), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, 3 * num_actions))
        else:
            self.actor_mean = layer_init(nn.Linear(512+3+1, num_actions), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

    def forward(self, cnn_out, command, speed):
        if self.branched:
            features = torch.concat((cnn_out, speed), dim=1)
            action_means = self.actor_mean(features)
            action_mean = command[:, 0:1] * action_means[:, 0:3] + \
                          command[:, 1:2] * action_means[:, 3:6] + \
                          command[:, 2:3] * action_means[:, 6:9]
            action_logstd = command[:, 0:1] * self.actor_logstd[:, 0:3] + \
                          command[:, 1:2] * self.actor_logstd[:, 3:6] + \
                          command[:, 2:3] * self.actor_logstd[:, 6:9]
            action_logstd = action_logstd.expand_as(action_mean)
            return action_mean, action_logstd

        else:
            features = torch.concat((cnn_out, command, speed), dim=1)
            action_mean = self.actor_mean(features)
            action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd


class Critic(nn.Module):
    def __init__(self, branched=False):
        super(Critic, self).__init__()
        self.branched = branched
        if branched:
            self.critic = layer_init(nn.Linear(512+1, 3), std=1)
        else:
            self.critic = layer_init(nn.Linear(512+3+1, 1), std=1)

    def forward(self, cnn_out, command, speed):
        if self.branched:
            features = torch.concat((cnn_out, speed), dim=1)
            values = self.critic(features)
            value = command[:, 0:1] * values[:, 0:1] + \
                    command[:, 1:2] * values[:, 1:2] + \
                    command[:, 2:3] * values[:, 2:3]
            return value
        else:
            features = torch.concat((cnn_out, command, speed), dim=1)
            return self.critic(features)


def sample_expert(expert_states, expert_commands, expert_speeds, expert_actions, num):
    indices = np.random.choice(np.arange(expert_states.shape[0]), num, replace=False)
    return expert_states[indices], expert_commands[indices], expert_speeds, expert_actions[indices]


# PPO agent class
class PPOAgent(nn.Module):
    def __init__(self, agent_name, num_actions, learning_rate, env, device, num_steps, writer,
                 n_channels=3, branched=False):
        super(PPOAgent, self).__init__()

        # storing values
        self.num_steps = num_steps
        self.writer = writer
        self.device = device
        self.env = env

        # networks
        self.cnn = CNNBackbone(n_channels=n_channels)
        self.actor = Actor(env, num_actions, branched=branched)
        self.critic = Critic(branched=branched)

        # model optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

        # next observation and done in the environment
        obs, command, speed = env.reset()
        self.next_obs = torch.Tensor(obs).to(device)
        self.next_command = torch.Tensor(command.copy()).to(device)
        self.next_speed = torch.Tensor([speed]).to(device)
        self.next_done = torch.zeros(()).to(device)

        # single-episode memory
        self.obs = torch.zeros((num_steps,) + env.observation_space).to(device)
        self.commands = torch.zeros((num_steps, 3)).to(device)
        self.speeds = torch.zeros((num_steps, 1)).to(device)
        self.actions = torch.zeros((num_steps,) + env.action_space).to(device)
        self.logprobs = torch.zeros((num_steps,)).to(device)
        self.rewards = torch.zeros((num_steps,)).to(device)
        self.dones = torch.zeros((num_steps,)).to(device)
        self.values = torch.zeros((num_steps,)).to(device)
        self.advantages = torch.zeros((num_steps,)).to(self.device)
        self.returns = None

        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_file = os.path.join('checkpoints', agent_name)

    def get_action_and_value(self, x, command, speed, action=None, deterministic=False):
        cnn_out = self.cnn(x)
        action_mean, action_logstd = self.actor(cnn_out, command, speed)
        value = self.critic(cnn_out, command, speed)

        # action_logstd = torch.tensor(-3.2).expand_as(action_logstd).to(self.device)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if deterministic:
                action = action_mean.clone().detach()
            else:
                action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def get_value(self, x, command, speed):
        cnn_out = self.cnn(x)
        return self.critic(cnn_out, command, speed)

    # rollout for num_steps in the environment
    def rollout(self, start_global_step):

        global_step = start_global_step

        for step in range(0, self.num_steps):

            global_step += 1
            self.obs[step] = self.next_obs
            self.commands[step] = self.next_command
            self.speeds[step] = self.next_speed
            self.dones[step] = self.next_done

            # action logic
            with torch.no_grad():
                action, logproba, _, value = self.get_action_and_value(
                    self.obs[step].unsqueeze(0), self.commands[step].unsqueeze(0), self.speeds[step].unsqueeze(0))

            self.values[step] = value.view(-1)
            self.actions[step] = action.view(-1)
            self.logprobs[step] = logproba.view(-1)

            # execute the game and log data
            next_obs, next_command, next_speed, reward, done, info = self.env.step(action.view(-1).cpu().numpy())
            self.rewards[step] = torch.tensor(reward).to(self.device)
            self.next_obs = torch.Tensor(next_obs).to(self.device)
            self.next_command = torch.Tensor(next_command).to(self.device)
            self.next_speed = torch.Tensor([next_speed]).to(self.device)
            self.next_done = torch.tensor(int(done)).to(self.device)

            if self.next_done.item() == 1:
                obs, command, speed = self.env.reset()
                self.next_obs = torch.Tensor(obs).to(self.device)
                self.next_command = torch.Tensor(command).to(self.device)
                self.next_speed = torch.Tensor([speed]).to(self.device)

            if 'distance' in info.keys():
                self.writer.add_scalar('charts/distance', info['distance'], global_step)

        return global_step

    def calc_returns(self, start_global_step):
        returns = []
        episodic_return = 0
        incomplete_episode = True

        for s in range(self.num_steps):
            if self.dones[s].item() == 1:
                if incomplete_episode:
                    incomplete_episode = False
                else:
                    returns.append((start_global_step + s, episodic_return))
                episodic_return = 0

            episodic_return += self.rewards[s].item()

        return returns

    # calculate generalized advantage estimation (GAE)
    def calc_advantage(self, gamma, gae_lambda):
        with torch.no_grad():
            next_value = self.get_value(
                self.next_obs.unsqueeze(0), self.next_command.unsqueeze(0), self.next_speed.unsqueeze(0)).reshape(1, -1)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            self.returns = self.advantages + self.values

    # learn from the experiences in the memory
    def learn(self, batch_size, minibatch_size, update_epochs, norm_adv, clip_coef, clip_vloss, ent_coef, vf_coef,
              max_grad_norm, interpolate_bc=False, bc_alpha=0.5, bc_states=None, bc_commands=None,
              bc_speeds=None, bc_actions=None):

        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)

            if interpolate_bc:
                # sample expert data
                bc_states_sample, bc_commands_sample, bc_speeds_sample, bc_actions_sample = \
                    sample_expert(bc_states, bc_commands, bc_speeds, bc_actions, self.num_steps)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(
                    self.obs[mb_inds], self.commands[mb_inds], self.speeds[mb_inds], self.actions[mb_inds])
                logratio = newlogprob - self.logprobs[mb_inds]
                ratio = logratio.exp()

                # advantage normalization
                mb_advantages = self.advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # interpolate with behavioral cloning
                if interpolate_bc:
                    bc_states_batch = torch.from_numpy(bc_states_sample[mb_inds]).float().to(self.device)
                    bc_commands_batch = torch.from_numpy(bc_commands_sample[mb_inds]).float().to(self.device)
                    bc_speeds_batch = torch.from_numpy(bc_speeds_sample[mb_inds]).float().to(self.device)
                    bc_actions_batch = torch.from_numpy(bc_actions_sample[mb_inds]).float().to(self.device)
                    _, bc_log_probs, _, _ = self.get_action_and_value(
                        bc_states_batch, bc_commands_batch, bc_speeds_batch, bc_actions_batch)
                    bc_loss = -bc_log_probs.mean()
                    full_pg_loss = bc_alpha * bc_loss + (1 - bc_alpha) * pg_loss
                else:
                    full_pg_loss = pg_loss

                # value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - self.returns[mb_inds]) ** 2
                    v_clipped = self.values[mb_inds] + torch.clamp(
                        newvalue - self.values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - self.returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - self.returns[mb_inds]) ** 2).mean()

                # entropy loss
                entropy_loss = entropy.mean()

                # total loss
                loss = full_pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                # optimizing
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                self.optimizer.step()

        return v_loss, pg_loss, bc_loss, full_pg_loss, entropy_loss

    def save_models(self):
        print('.... saving models ....')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_models(self):
        print('.... loading models ....')
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
