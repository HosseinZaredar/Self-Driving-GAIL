import math

import random
import os
import queue
import numpy as np
import matplotlib.pyplot as plt

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption


class CarlaEnv:
    def __init__(self, world='Town02_Opt', fps=10, image_w=256, image_h=144, record=False):
        self.image_w = image_w
        self.image_h = image_h
        self.record = record
        self.episode_number = -1
        self.obs_number = 0

        self.spawn = {'x': 10.0, 'y': 191.7}
        self.dest = {'x': 75.0, 'y': 241.0}

        # dimension
        self.observation_space = (3, image_h, image_w)
        self.action_space = (3,)

        # connecting to carla
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        print('carla connected.')

        # setting the world up
        self.world = self.client.load_world(world)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / fps
        self.world.apply_settings(settings)

        self.image_queue = queue.Queue()
        self.vehicle = None
        self.agent = None
        self.camera = None

    def reset(self):

        self.episode_number += 1
        self.obs_number = 0

        # deleting vehicle and camera (if they exist)
        self.image_queue = queue.Queue()
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()

        # spawning a vehicle
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_bp = blueprint_lib.filter('model3')[0]
        vehicle_spawn_point = carla.Transform(
            carla.Location(x=self.spawn['x'] + 20 * random.random(), y=self.spawn['y'], z=0.5),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_spawn_point)

        # setting up a camera
        camera_bp = blueprint_lib.find('sensor.camera.rgb')

        bound_x = 0.5 + self.vehicle.bounding_box.extent.x
        bound_y = 0.5 + self.vehicle.bounding_box.extent.y
        bound_z = 0.5 + self.vehicle.bounding_box.extent.z

        camera_bp.set_attribute('image_size_x', f'{self.image_w}')
        camera_bp.set_attribute('image_size_y', f'{self.image_h}')
        camera_spawn_point = carla.Transform(carla.Location(x=+0.8 * bound_x, y=+0.0 * bound_y, z=1.3 * bound_z))
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_spawn_point,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid)

        # save camera images in queue
        self.camera.listen(self.image_queue.put)

        self.world.tick()
        obs = self.process_image(self.image_queue.get())
        if self.record:
            self.save_obs(obs)

        # setup agent to provide high-level commands
        self.agent = BasicAgent(self.vehicle)
        self.agent.ignore_traffic_lights(active=True)
        dest = carla.Location(x=self.dest['x'], y=self.dest['y'], z=0.0)
        self.agent.set_destination(dest)

        # command: move forward
        command = np.array([0.0, 1.0, 0.0])

        # speed
        speed = 0

        return obs, command, speed

    def step(self, action):
        throttle = float(np.clip(action[0], 0, 1))
        steer = float(np.clip(action[1], -1, 1))
        brake = float(np.clip(action[2], 0, 1))
        if brake < 0.01:
            brake = 0
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        self.world.tick()
        obs = self.process_image(self.image_queue.get())
        if self.record:
            self.save_obs(obs)
        self.obs_number += 1

        control, road_option = self.agent.run_step()
        if road_option == RoadOption.LEFT:
            command = np.array([1.0, 0.0, 0.0])
        elif road_option == RoadOption.LANEFOLLOW:
            command = np.array([0.0, 1.0, 0.0])
        elif road_option == RoadOption.RIGHT:
            command = np.array([0.0, 0.0, 1.0])

        location = self.vehicle.get_transform().location
        dist = math.sqrt((self.dest['x'] - location.x) ** 2 + (self.dest['y'] - location.y) ** 2)

        if self.obs_number == 220 or dist < 5:
            done = True
            info = {'distance': dist}
        else:
            done = False
            info = {}

        # reward
        reward = 0

        # speed
        v = self.vehicle.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        return obs, command, speed, reward, done, info

    def close(self):
        self.camera.stop()
        self.camera.destroy()
        self.vehicle.destroy()
        print('carla disconnected.')

    def save_obs(self, obs):
        dir = 'agent_rollout'
        if not os.path.exists(dir):
            os.makedirs(dir)

        sub_dir = os.path.join(dir, f'ep_{self.episode_number}')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        plt.imsave(os.path.join(sub_dir, f'obs_{self.obs_number}.png'), obs.transpose(1, 2, 0))

    @staticmethod
    def process_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = array.transpose((2, 0, 1))
        return array
