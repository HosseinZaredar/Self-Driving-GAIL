import os

from env.env import CarlaEnv
from env import routes
from algo.ppo import PPOAgent

import numpy as np
import random
import argparse
from distutils.util import strtobool
import matplotlib.pyplot as plt
import torch


# command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--agent-name', type=str, default='bc_gail')
    parser.add_argument('--use-cuda', type=lambda x: bool(strtobool(x)), nargs='?', default=False, const=True)
    parser.add_argument('--deterministic-cuda', type=lambda x: bool(strtobool(x)), nargs='?', default=False, const=True)
    parser.add_argument('--deterministic', type=lambda x: bool(strtobool(x)), nargs='?', default=True, const=True)
    parser.add_argument("--branched", type=lambda x: bool(strtobool(x)), nargs='?', default=True, const=True)
    parser.add_argument("--on-test-set", type=lambda x: bool(strtobool(x)), nargs='?', default=True, const=True)
    parser.add_argument("--generate-saliency", type=lambda x: bool(strtobool(x)), nargs='?', default=False, const=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # cuda setting
    torch.backends.cudnn.deterministic = args.deterministic_cuda
    torch.backends.cudnn.benchmark = True

    # compute device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # carla env setup
    env = CarlaEnv(evaluate=True, on_test_set=args.on_test_set)

    # initializing and loading agent
    agent = PPOAgent(args.agent_name, 3, 0, env, device, 0, None, branched=args.branched).float()
    print('.... loading models ....')
    agent.load_models()

    # dataset
    if not args.on_test_set:
        num_routes = len(routes.train_spawns)
    else:
        num_routes = len(routes.test_spawns)

    # config for generating saliency maps
    if args.generate_saliency:
        agent.requires_grad_(False)
        obs_requires_grad = True

        save_dir = os.path.join('agent_rollout', 'train' if not args.on_test_set else 'test')

        forward_relu_outputs = []


        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            forward_relu_outputs.append(ten_out)

        for pos, module in agent.cnn.conv._modules.items():
            if isinstance(module, torch.nn.LeakyReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    else:
        obs_requires_grad = False

    for i in range(0, num_routes):
        done = False
        obs, command, speed = env.reset(route=i)

        if args.generate_saliency:
            saliency_dir = os.path.join(save_dir, f'ep_{i}', 'saliency')
            if not os.path.exists(saliency_dir):
                os.makedirs(saliency_dir)
            if not os.path.exists(os.path.join(saliency_dir, 'right')):
                os.makedirs(os.path.join(saliency_dir, 'right'))
            if not os.path.exists(os.path.join(saliency_dir, 'front')):
                os.makedirs(os.path.join(saliency_dir, 'front'))
            if not os.path.exists(os.path.join(saliency_dir, 'left')):
                os.makedirs(os.path.join(saliency_dir, 'left'))

        step_number = -1
        while not done:
            step_number += 1
            obs = torch.tensor(obs, dtype=torch.float, requires_grad=obs_requires_grad).to(device)
            command = torch.tensor(command, dtype=torch.float).to(device)
            speed = torch.tensor([speed], dtype=torch.float).to(device)
            action, _, _, _ = agent.get_action_and_value(
                obs.unsqueeze(0), command.unsqueeze(0), speed.unsqueeze(0), deterministic=args.deterministic)

            # generate and save saliency maps
            if args.generate_saliency:
                action[0][1].backward()

                # left camera
                slc, _ = torch.max(torch.abs(obs.grad[0:3]), dim=0)
                slc = (slc - slc.min()) / (slc.max() - slc.min())
                plt.imsave(os.path.join(saliency_dir, 'left', f'obs_{step_number:03}.png'), slc, cmap=plt.cm.hot)

                # front camera
                slc, _ = torch.max(torch.abs(obs.grad[3:6]), dim=0)
                slc = (slc - slc.min()) / (slc.max() - slc.min())
                plt.imsave(os.path.join(saliency_dir, 'front', f'obs_{step_number:03}.png'), slc, cmap=plt.cm.hot)

                # right camera
                slc, _ = torch.max(torch.abs(obs.grad[6:9]), dim=0)
                slc = (slc - slc.min()) / (slc.max() - slc.min())
                plt.imsave(os.path.join(saliency_dir, 'right', f'obs_{step_number:03}.png'), slc, cmap=plt.cm.hot)

            # perform action
            action = action.clone().detach()
            next_obs, command, speed, _, done, info = env.step(action.view(-1).cpu().numpy())
            obs = next_obs

            if done:
                print(f'route {i:02}: {info}')

    env.close()
