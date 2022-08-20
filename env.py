import routes

import math
import os
import queue
import numpy as np
import matplotlib.pyplot as plt

import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner


class CarlaEnv:
    def __init__(self, world='Town02_Opt', fps=10, image_w=256, image_h=144, random_spawn=False,
                 evaluate=False, on_test_set=False, eval_image_w=640, eval_image_h=360):

        self.image_w = image_w
        self.image_h = image_h
        self.random_spawn = random_spawn
        self.evaluate = evaluate
        self.eval_image_w = eval_image_w
        self.eval_image_h = eval_image_h

        self.episode_number = -2
        self.obs_number = 0
        self.max_episode_steps = 135
        self.current_path = 0

        # spawn and destination location
        self.on_test_set = on_test_set
        if not on_test_set:
            self.spawns = routes.train_spawns
            self.rotations = routes.train_rotations
            self.dests = routes.train_dests
        else:
            self.spawns = routes.test_spawns
            self.rotations = routes.test_rotations
            self.dests = routes.test_dests

        # dimension
        self.observation_space = (9, image_h, image_w)
        self.action_space = (3,)

        # connecting to carla
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)

        # setting the world up
        self.world = self.client.load_world(world)
        print('carla connected.')

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / fps
        self.world.apply_settings(settings)

        # calculate the route to destination
        world_map = self.world.get_map()
        planner = GlobalRoutePlanner(world_map, sampling_resolution=1.0)

        route = []
        for i in range(len(self.spawns)):
            start_waypoint = world_map.get_waypoint(self.spawns[i])
            end_waypoint = world_map.get_waypoint(self.dests[i])
            route.append(planner.trace_route(
                start_waypoint.transform.location, end_waypoint.transform.location))

        # calculate distance to destination for every point along the road
        self.distances = [[] for _ in range(len(self.spawns))]

        for i in range(len(self.spawns)):
            for j in range(len(route[i]) - 1):
                p1 = route[i][j][0].transform.location
                p2 = route[i][j+1][0].transform.location
                distance = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
                self.distances[i].append(distance)
            self.distances[i].append(0)

            for j in reversed(range(len(route[i]) - 1)):
                self.distances[i][j] += self.distances[i][j+1]

        # recording directory
        if self.evaluate:
            main_dir = 'agent_rollout'
            if not os.path.exists(main_dir):
                os.makedirs(main_dir)
            self.dir = os.path.join(main_dir, 'train' if not self.on_test_set else 'test')

        self.image_queues = [queue.Queue(), queue.Queue(), queue.Queue()]
        self.eval_image_queue = queue.Queue()
        self.vehicle = None
        self.agent = None
        self.cameras = []
        self.eval_camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.early_terminate = False
        self.invaded = False

    def reset(self, path=None):

        self.episode_number += 1
        self.obs_number = 0
        self.early_terminate = False
        self.invaded = False

        # choose a path from the list of paths randomly
        if path is None:
            self.current_path = self.episode_number % len(self.spawns)
        else:
            self.current_path = path

        # deleting vehicle and sensors (if they already exist)
        self.image_queues = [queue.Queue(), queue.Queue(), queue.Queue()]
        self.destroy_sensors()
        self.cameras = []

        # spawning vehicle
        blueprint_lib = self.world.get_blueprint_library()
        vehicle_bp = blueprint_lib.filter('model3')[0]
        vehicle_spawn_point = carla.Transform(self.spawns[self.current_path], self.rotations[self.current_path])
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_spawn_point)

        # collision sensor
        self.collision_sensor = self.world.spawn_actor(
            blueprint_lib.find('sensor.other.collision'), carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.terminate())

        # lane invasion sensor
        self.lane_invasion_sensor = self.world.spawn_actor(
            blueprint_lib.find('sensor.other.lane_invasion'), carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(lambda event: self.invade())

        # setting up main cameras
        camera_bp = blueprint_lib.find('sensor.camera.rgb')

        bound_x = 0.5 + self.vehicle.bounding_box.extent.x
        bound_y = 0.5 + self.vehicle.bounding_box.extent.y
        bound_z = 0.5 + self.vehicle.bounding_box.extent.z

        camera_bp.set_attribute('image_size_x', f'{self.image_w}')
        camera_bp.set_attribute('image_size_y', f'{self.image_h}')

        for i, degree in enumerate([315, 0, 45]):
            camera_spawn_point = carla.Transform(
                carla.Location(x=+0.8 * bound_x, y=+0.0 * bound_y, z=1.0 * bound_z),
                carla.Rotation(yaw=degree)
            )
            self.cameras.append(self.world.spawn_actor(
                camera_bp,
                camera_spawn_point,
                attach_to=self.vehicle,
                attachment_type=carla.AttachmentType.Rigid))

            self.cameras[i].listen(self.image_queues[i].put)

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

        # episode record directory
        if self.evaluate and self.episode_number != -1:
            sub_dir = os.path.join(self.dir, f'ep_{self.episode_number}')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        self.world.tick()
        img_0 = self.process_image(self.image_queues[0].get())
        img_1 = self.process_image(self.image_queues[1].get())
        img_2 = self.process_image(self.image_queues[2].get())
        obs = np.vstack((img_0, img_1, img_2))

        if self.evaluate and self.episode_number != -1:
            image = self.process_image(self.eval_image_queue.get())
            self.save_image(image)

        # setup agent to provide high-level commands
        self.agent = BasicAgent(self.vehicle)
        self.agent.ignore_traffic_lights(active=True)
        self.agent.set_destination(self.dests[self.current_path])

        # command: move forward
        command = np.array([0.0, 1.0, 0.0])

        # vehicle speed
        speed = 0

        return obs, command, speed

    def step(self, action):

        # clipping values
        throttle = float(np.clip(action[0], 0, 1))
        steer = float(np.clip(action[1], -1, 1))
        brake = float(np.clip(action[2], 0, 1))
        if brake < 0.01:
            brake = 0

        # applying the action
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        self.world.tick()

        # get the next observation
        img_0 = self.process_image(self.image_queues[0].get())
        img_1 = self.process_image(self.image_queues[1].get())
        img_2 = self.process_image(self.image_queues[2].get())
        obs = np.vstack((img_0, img_1, img_2))

        if self.evaluate:
            image = self.process_image(self.eval_image_queue.get())
            self.save_image(image)
        self.obs_number += 1

        # get high-level command from global planner
        _, road_option, num_points_done = self.agent.run_step()
        if road_option == RoadOption.LANEFOLLOW or road_option == RoadOption.STRAIGHT:
            command = np.array([0.0, 1.0, 0.0])
        elif road_option == RoadOption.LEFT:
            command = np.array([1.0, 0.0, 0.0])
        elif road_option == RoadOption.RIGHT:
            command = np.array([0.0, 0.0, 1.0])

        # calculate distance to destination
        location = self.vehicle.get_transform().location
        aerial_dist = math.sqrt(
            (self.dests[self.current_path].x - location.x) ** 2
            + (self.dests[self.current_path].y - location.y) ** 2)
        road_dist = self.distances[self.current_path][num_points_done]

        # check episode termination
        if self.obs_number == self.max_episode_steps or self.early_terminate:
            done = True
            info = {'distance': road_dist}
        elif aerial_dist < 5 or road_dist < 5:
            done = True
            info = {'distance': 0}
        else:
            done = False
            info = {}

        # environment reward (in case of lane invasion, -20)
        if self.invaded:
            reward = -20
            self.invaded = False
        else:
            reward = 0

        # vehicle speed
        v = self.vehicle.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        return obs, command, speed, reward, done, info

    def close(self):
        self.destroy_sensors()
        print('carla disconnected.')

    def terminate(self):
        self.early_terminate = True

    def invade(self):
        self.invaded = True

    def save_image(self, image):
        plt.imsave(os.path.join(self.dir, f'ep_{self.episode_number}', f'obs_{self.obs_number:03}.png'),
                   image.transpose(1, 2, 0))

    @staticmethod
    def process_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = array.transpose((2, 0, 1))
        return array.copy()

    def destroy_sensors(self):
        sensors = [*self.cameras, self.eval_camera, self.collision_sensor, self.lane_invasion_sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
