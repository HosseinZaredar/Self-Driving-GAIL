from env.env import CarlaEnv
from algo.ppo import PPOAgent
from algo.disc import Discriminator

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
    parser.add_argument('--ppo-learning-rate', type=float, default=3e-4)
    parser.add_argument('--disc-learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--total_timesteps', type=int, default=150_000)
    parser.add_argument('--use-cuda', type=lambda x: strtobool(x), nargs='?', default=True, const=True)
    parser.add_argument('--deterministic-cuda', type=lambda x: strtobool(x), nargs='?', default=False, const=True)
    parser.add_argument('--num-steps', type=int, default=512,
                        help='number of steps in each environment before training')
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=16, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")

    parser.add_argument("--num-disc-epochs", type=int, default=1)
    parser.add_argument("--num-disc-minibatches", type=int, default=16)
    parser.add_argument("--half-life", type=int, default=160)
    parser.add_argument("--wasserstein", type=lambda x: bool(strtobool(x)), nargs='?', default=False, const=True)
    parser.add_argument("--grad-penalty", type=lambda x: bool(strtobool(x)), nargs='?', default=False, const=True)
    parser.add_argument("--branched", type=lambda x: bool(strtobool(x)), nargs='?', default=True, const=True)

    args = parser.parse_args()
    args.batch_size = args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args


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

    expert_states = np.empty((total_length, 9, 144, 256), dtype=np.float32)
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
        expert_states[loaded_length: loaded_length+episode_length] = np_states
        expert_commands[loaded_length: loaded_length + episode_length] = np_commands
        expert_speeds[loaded_length: loaded_length + episode_length] = np_speeds
        expert_actions[loaded_length: loaded_length + episode_length] = np_actions

        loaded_length += episode_length

    print('data loaded!')

    # initialize discriminator
    disc = Discriminator(device, args.disc_learning_rate, env.observation_space, 3,
                         args.wasserstein, args.grad_penalty, branched=args.branched).float()

    # initialize ppo agent
    agent = PPOAgent('bc_gail', 3, args.ppo_learning_rate, env, device,
                     args.num_steps, writer, branched=args.branched).float()

    # start the game
    global_step = 0
    num_updates = args.total_timesteps // args.batch_size

    # starting alpha
    alpha_0 = 0.5 ** (1 / args.half_life)

    for update in range(1, num_updates + 1):

        # save the agent model
        print('.... saving models ....')
        agent.save_models()
        disc.save_models()

        # rollout agent
        global_step = agent.rollout(global_step)

        # update discriminator
        for _ in range(args.num_disc_epochs if global_step > 10_000 else 2):
            disc_real_loss, disc_fake_loss, disc_raw_loss, disc_loss = disc.learn(
                args.num_steps // args.num_disc_minibatches,
                expert_states, expert_commands, expert_speeds, expert_actions,
                agent.obs, agent.commands, agent.speeds, agent.actions)

        # calculate rewards from the discriminator
        agent.rewards += disc.generate_rewards(agent.obs, agent.commands, agent.speeds, agent.actions)

        # calculate episodic return (for logging/debugging purposes)
        returns = agent.calc_returns(global_step - args.num_steps)
        for r in returns:
            writer.add_scalar("charts/episodic_return_disc", r[1], r[0])

        # calculate advantage
        agent.calc_advantage(args.gamma, args.gae_lambda)

        # calculating alpha
        t = (update * args.update_epochs * args.num_steps) / len(expert_states)
        alpha = alpha_0 ** t

        # agent learning
        v_loss, pg_loss, bc_loss, full_pg_loss, entropy_loss = agent.learn(
            args.batch_size, args.minibatch_size, args.update_epochs, args.norm_adv, args.clip_coef,
            args.clip_vloss, args.ent_coef, args.vf_coef, args.max_grad_norm, interpolate_bc=True,
            bc_alpha=alpha, bc_states=expert_states, bc_commands=expert_commands,
            bc_speeds=expert_speeds, bc_actions=expert_actions)

        # record rewards for plotting purposes
        writer.add_scalar("charts/alpha", alpha, global_step)
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/bc_loss", bc_loss.item(), global_step)
        writer.add_scalar("losses/full_pg_loss", full_pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/disc_real", disc_real_loss, global_step)
        writer.add_scalar("losses/disc_fake", disc_fake_loss, global_step)
        writer.add_scalar("losses/disc_raw", disc_raw_loss, global_step)
        writer.add_scalar("losses/disc_total", disc_loss, global_step)
