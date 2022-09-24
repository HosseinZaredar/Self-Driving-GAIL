from env.env import CarlaEnv
from algo.ppo import PPOAgent

import os
import torch
import numpy as np
from distutils.util import strtobool
import argparse
import time
import random
from torch.utils.tensorboard import SummaryWriter


# command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--minibatch_size', type=int, default=32)
    parser.add_argument('--use-cuda', type=lambda x: strtobool(x), nargs='?', default=True, const=True)
    parser.add_argument('--deterministic-cuda', type=lambda x: strtobool(x), nargs='?', default=False, const=True)
    parser.add_argument("--branched", type=lambda x: strtobool(x), nargs='?', default=True, const=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    run_name = f'train-bc_{args.seed}_{int(time.time())}'

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

    # cuda setup
    torch.backends.cudnn.deterministic = args.deterministic_cuda
    torch.backends.cudnn.benchmark = True

    # compute device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # carla env setup
    env = CarlaEnv()

    # load expert trajectories
    print('loading data...')

    total_length = 0
    for dir in os.listdir('expert_data'):
        with open(os.path.join('expert_data', dir, 'len.txt')) as f:
            total_length += int(f.readline())

    expert_states = np.empty((total_length, 9, 112, 256), dtype=np.float32)
    expert_commands = np.empty((total_length, 3), dtype=np.float32)
    expert_speeds = np.empty((total_length, 1), dtype=np.float32)
    expert_actions = np.empty((total_length, 3), dtype=np.float32)

    loaded_length = 0
    for dir in os.listdir('expert_data'):
        np_states = np.load(os.path.join('expert_data', dir, 'expert_states.npy'))
        np_commands = np.load(os.path.join('expert_data', dir, 'expert_commands.npy'))
        np_speeds = np.load(os.path.join('expert_data', dir, 'expert_speeds.npy'))
        np_actions = np.load(os.path.join('expert_data', dir, 'expert_actions.npy'))

        episode_length = np_states.shape[0]
        expert_states[loaded_length: loaded_length + episode_length] = np_states
        expert_commands[loaded_length: loaded_length + episode_length] = np_commands
        expert_speeds[loaded_length: loaded_length + episode_length] = np_speeds
        expert_actions[loaded_length: loaded_length + episode_length] = np_actions

        loaded_length += episode_length

    print('data loaded!')

    # train-validation split
    ratio = 0.8
    num_states = len(expert_states)
    expert_states_train = expert_states[: int(ratio * num_states)]
    expert_commands_train = expert_commands[: int(ratio * num_states)]
    expert_speeds_train = expert_speeds[: int(ratio * num_states)]
    expert_actions_train = expert_actions[: int(ratio * num_states)]
    expert_states_val = expert_states[int(ratio * num_states):]
    expert_commands_val = expert_commands[int(ratio * num_states):]
    expert_speeds_val = expert_speeds[int(ratio * num_states):]
    expert_actions_val = expert_actions[int(ratio * num_states):]

    # initialize ppo agent
    agent = PPOAgent('bc', 3, args.learning_rate, env, device, 0, writer, branched=args.branched).float()

    for epoch in range(1, args.max_epochs + 1):

        if epoch % 10 == 0:
            print('.... saving models ....')
            agent.save_models()

        batch_starts_train = np.arange(0, len(expert_states_train), args.minibatch_size)
        mean_loss_train = 0

        # train
        for b in batch_starts_train:
            expert_states_batch = torch.from_numpy(
                expert_states_train[b: b + args.minibatch_size]).to(device)
            expert_commands_batch = torch.from_numpy(
                expert_commands_train[b: b + args.minibatch_size]).to(device)
            expert_speeds_batch = torch.from_numpy(
                expert_speeds_train[b: b + args.minibatch_size]).to(device)
            expert_actions_batch = torch.from_numpy(
                expert_actions_train[b: b + args.minibatch_size]).to(device)

            _, bc_log_probs, _, _ = agent.get_action_and_value(
                expert_states_batch, expert_commands_batch, expert_speeds_batch, expert_actions_batch)

            bc_loss = -bc_log_probs.mean()
            mean_loss_train += bc_loss.item()

            agent.optimizer.zero_grad()
            bc_loss.backward()
            agent.optimizer.step()

        batch_starts_val = np.arange(0, len(expert_states_val), args.minibatch_size)
        mean_loss_val = 0

        # validation
        with torch.no_grad():
            for b in batch_starts_val:
                expert_states_batch = torch.from_numpy(
                    expert_states_val[b: b + args.minibatch_size]).to(device)
                expert_commands_batch = torch.from_numpy(
                    expert_commands_val[b: b + args.minibatch_size]).to(device)
                expert_speeds_batch = torch.from_numpy(
                    expert_speeds_val[b: b + args.minibatch_size]).to(device)
                expert_actions_batch = torch.from_numpy(
                    expert_actions_val[b: b + args.minibatch_size]).to(device)

                _, bc_log_probs, _, _ = agent.get_action_and_value(
                    expert_states_batch, expert_commands_batch, expert_speeds_batch, expert_actions_batch)

                bc_loss = -bc_log_probs.mean()
                mean_loss_val += bc_loss.item()

        # deterministic evaluation in the environment
        if epoch % 2 == 0:

            with torch.no_grad():
                done = False
                obs, command, speed = env.reset()

                while not done:
                    obs = torch.tensor(obs.copy(), dtype=torch.float).to(device)
                    command = torch.tensor(command, dtype=torch.float).to(device)
                    speed = torch.tensor([speed], dtype=torch.float).to(device)
                    action, _, _, _ = agent.get_action_and_value(
                        obs.unsqueeze(0), command.unsqueeze(0), speed.unsqueeze(0), deterministic=True)
                    next_obs, command, speed, _, done, info = env.step(action.view(-1).cpu().numpy())
                    obs = next_obs

                writer.add_scalar('charts/distance', info['distance'], epoch)

        # record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("losses/train_loss", mean_loss_train, epoch)
        writer.add_scalar("losses/val_loss", mean_loss_val, epoch)

    agent.save_models()
    env.close()
