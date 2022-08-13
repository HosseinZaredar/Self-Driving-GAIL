import math

import os
import queue
import random

import numpy as np
import matplotlib.pyplot as plt

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner


class CarlaEnv:
    def __init__(self, world='Town02_Opt', fps=10, image_w=256, image_h=144, record=False,
                 random_spawn=False, evaluate=False, eval_image_w=1280, eval_image_h=720):

        self.image_w = image_w
        self.image_h = image_h
        self.record = record
        self.evaluate = evaluate
        self.eval_image_w = eval_image_w
        self.eval_image_h = eval_image_h

        self.episode_number = -1
        self.obs_number = 0

        # spawn and destination location
        self.spawn_points = []
        self.spawn = carla.Location(x=155.0, y=109.4, z=0.5)
        self.rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        self.dest = carla.Location(x=189.9, y=130.0, z=0.0)
        self.random_spawn = random_spawn

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

        # random spawn
        if self.random_spawn:
            world_map = self.world.get_map()
            planner = GlobalRoutePlanner(world_map, sampling_resolution=1.0)
            start_waypoint = world_map.get_waypoint(self.spawn)
            end_waypoint = world_map.get_waypoint(self.dest)
            route = planner.trace_route(
                start_waypoint.transform.location, end_waypoint.transform.location)
            for point in route:
                if point[1] == RoadOption.LANEFOLLOW:
                    self.spawn_points.append((point[0].transform.location, point[0].transform.rotation))

        self.image_queue = queue.Queue()
        self.eval_image_queue = queue.Queue()
        self.vehicle = None
        self.agent = None
        self.camera = None
        self.eval_camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.early_terminate = False

    def reset(self):

        self.episode_number += 1
        self.obs_number = 0
        self.early_terminate = False

        # deleting vehicle and sensors (if already exist)
        self.image_queue = queue.Queue()
        sensors = [self.camera, self.eval_camera, self.collision_sensor, self.lane_invasion_sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()

        # random spawn
        if self.random_spawn:
            spawn_point = random.choice(self.spawn_points)
            self.spawn = carla.Location(x=spawn_point[0].x, y=spawn_point[0].y, z=0.5)
            self.rotation = spawn_point[1]

        # spawning a vehicle
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_bp = blueprint_lib.filter('model3')[0]
        vehicle_spawn_point = carla.Transform(self.spawn, self.rotation)
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_spawn_point)

        # collision sensor
        self.collision_sensor = self.world.spawn_actor(
            blueprint_lib.find('sensor.other.collision'), carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.terminate())

        # lane invasion sensor
        self.lane_invasion_sensor = self.world.spawn_actor(
            blueprint_lib.find('sensor.other.lane_invasion'), carla.Transform(), attach_to=self.vehicle)
        #self.lane_invasion_sensor.listen(lambda event: self.terminate())

        # setting up main camera
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

        self.camera.listen(self.image_queue.put)

        # setting up evaluation camera
        if self.evaluate:
            camera_bp = blueprint_lib.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', f'{self.eval_image_w}')
            camera_bp.set_attribute('image_size_y', f'{self.eval_image_h}')
            camera_spawn_point = carla.Transform(
                carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z),
                carla.Rotation(pitch=8.0))
            self.eval_camera = self.world.spawn_actor(
                camera_bp,
                camera_spawn_point,
                attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.SpringArm)

            self.eval_camera.listen(self.eval_image_queue.put)

        self.world.tick()
        obs = self.process_image(self.image_queue.get())
        if self.record:
            if self.evaluate:
                image = self.process_image(self.eval_image_queue.get())
            else:
                image = obs
            self.save_image(image)

        # setup agent to provide high-level commands
        self.agent = BasicAgent(self.vehicle)
        self.agent.ignore_traffic_lights(active=True)
        self.agent.set_destination(self.dest)

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
            if self.evaluate:
                image = self.process_image(self.eval_image_queue.get())
            else:
                image = obs
            self.save_image(image)
        self.obs_number += 1

        # get high-level command form global planner
        _, road_option = self.agent.run_step()
        if road_option == RoadOption.LEFT:
            command = np.array([1.0, 0.0, 0.0])
        elif road_option == RoadOption.LANEFOLLOW:
            command = np.array([0.0, 1.0, 0.0])
        elif road_option == RoadOption.RIGHT:
            command = np.array([0.0, 0.0, 1.0])

        # calculate distance to destination
        location = self.vehicle.get_transform().location
        dist = math.sqrt((self.dest.x - location.x) ** 2 + (self.dest.y - location.y) ** 2)

        # episode termination
        if self.obs_number == 130 or dist < 5 or self.early_terminate:
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

    def terminate(self):
        self.early_terminate = True

    def save_image(self, image):
        dir = 'agent_rollout'
        if not os.path.exists(dir):
            os.makedirs(dir)

        sub_dir = os.path.join(dir, f'ep_{self.episode_number}')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        plt.imsave(os.path.join(sub_dir, f'obs_{self.obs_number:03}.png'), image.transpose(1, 2, 0))

    @staticmethod
    def process_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = array.transpose((2, 0, 1))
        return array
