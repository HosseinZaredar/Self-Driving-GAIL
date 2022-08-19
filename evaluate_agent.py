from env import CarlaEnv
from ppo import PPOAgent
import routes

import numpy as np
import random
import argparse
from distutils.util import strtobool
import torch


# command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--agent-name', type=str, default='bc_gail_learner')
    parser.add_argument('--record', type=lambda x: strtobool(x), default=True)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--deterministic-cuda', type=lambda x: strtobool(x), nargs='?', default=False, const=True)
    parser.add_argument('--deterministic', type=lambda x: strtobool(x), default=True)
    parser.add_argument("--branched", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--on-test-set", type=lambda x: bool(strtobool(x)), default=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic_cuda
    torch.backends.cudnn.benchmark = True

    # compute device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # carla env setup
    env = CarlaEnv(evaluate=True, on_test_set=args.on_test_set)

    # initializing and loading agent
    agent = PPOAgent(args.agent_name, 3, 0, env, device, 0, None, branched=args.branched).float()
    agent.load_models()

    if not args.on_test_set:
        num_routes = len(routes.train_spawns)
    else:
        num_routes = len(routes.test_spawns)

    for i in range(num_routes):
        done = False
        obs, command, speed = env.reset(path=i)

        while not done:
            obs = torch.tensor(obs, dtype=torch.float).to(device)
            command = torch.tensor(command, dtype=torch.float).to(device)
            speed = torch.tensor([speed], dtype=torch.float).to(device)
            action, _, _, _ = agent.get_action_and_value(
                obs.unsqueeze(0), command.unsqueeze(0), speed.unsqueeze(0), deterministic=args.deterministic)
            next_obs, command, speed, _, done, info = env.step(action.view(-1).cpu().numpy())
            obs = next_obs

            if done:
                print(f'route {i:02}: {info}')

    env.close()
