"""
    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    P            : toggle autopilot
    TAB          : toggle camera
"""


import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

import os
import numpy as np
import argparse
import datetime
import random
import math
import weakref
import matplotlib.pyplot as plt

import pygame
from pygame.locals import K_TAB
from pygame.locals import K_q
from pygame.locals import K_p
from pygame.locals import K_w
from pygame.locals import K_a
from pygame.locals import K_s
from pygame.locals import K_d


# ------------------------------------------------------------------------------

class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.camera_manager = None
        self.vehicle_name = args.vehicle

        # command
        self.command = None
        self.set_command('forward')

        # vehicle spawn
        self.spawn = args.spawn
        self.rotation = args.rotation

        # vehicle destination
        self.dest = args.dest

        # autopilot agent
        self.agent = None

        self.restart()

        # setup autopilot agent
        self.agent = BasicAgent(self.player)
        self.agent.ignore_traffic_lights(active=True)
        self.agent.set_destination(self.dest)

        self.world.on_tick(hud.on_world_tick)

        # recording
        self.episode_number = args.episode_number
        self.save_png = args.save_png
        self.recording = False
        self.observations = []
        self.commands = []
        self.speeds = []
        self.actions = []

    def toggle_recording(self):
        if self.recording:
            directory = os.path.join('expert_data', f'eps_{self.episode_number}')
            if not os.path.exists(directory):
                os.makedirs(directory)

            np_observations = np.empty(
                (len(self.observations), * self.observations[0].transpose((2, 0, 1)).shape), dtype=np.float32)
            np_commands = np.array(self.commands, dtype=np.float32)
            np_speeds = np.expand_dims(np.array(self.speeds, dtype=np.float32), axis=1)
            np_actions = np.array(self.actions, dtype=np.float32)

            for i, obs in enumerate(self.observations):
                np_observations[i] = np.transpose(obs, (2, 0, 1))
                if self.save_png:
                    plt.imsave(os.path.join(directory, f'obs_{i}.png'), obs)

            np.save(os.path.join(directory, 'expert_states.npy'), np_observations)
            np.save(os.path.join(directory, 'expert_commands.npy'), np_commands)
            np.save(os.path.join(directory, 'expert_speeds.npy'), np_speeds)
            np.save(os.path.join(directory, 'expert_actions.npy'), np_actions)

            self.observations = []
            self.commands = []
            self.speeds = []
            self.actions = []

        self.recording = not self.recording

    def record(self):
        if self.recording:
            self.observations.append(np.copy(self.camera_manager.obs))
            self.commands.append(np.copy(self.command))
            c = self.player.get_control()
            self.actions.append(np.array([c.throttle, c.steer, c.brake]))

            v = self.player.get_velocity()
            self.speeds.append(math.sqrt(v.x**2 + v.y**2 + v.z**2))

    def restart(self):

        # keep same camera config if the camera manager exists
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # get vehicle blueprint
        blueprint = self.world.get_blueprint_library().filter(self.vehicle_name)[0]

        # spawn the vehicle
        while self.player is None:
            spawn_point = carla.Transform( self.spawn, self.rotation)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            physics_control = self.player.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            self.player.apply_physics_control(physics_control)

        # collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.world.get_blueprint_library().find('sensor.other.collision'), carla.Transform(), attach_to=self.player)
        self.collision_sensor.listen(lambda event: print('collision!'))

        # setup the camera
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index)
        self.world.tick()

    def set_command(self, command):
        if command == 'forward':
            self.command = np.array([0.0, 1.0, 0.0])
        elif command == 'left':
            self.command = np.array([1.0, 0.0, 0.0])
        elif command == 'right':
            self.command = np.array([0.0, 0.0, 1.0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        sensors = [self.camera_manager.sensor, self.collision_sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ------------------------------------------------------------------------------


class KeyboardControl(object):
    def __init__(self, world, autopilot_enabled):
        self._autopilot_enabled = autopilot_enabled
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        world.player.set_light_state(self._lights)
        self._steer_cache = 0.0

    def parse_events(self, world, clock):
        current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                if event.key == K_TAB:
                    world.camera_manager.toggle_camera()

        # high-level command
        control, road_option, _ = world.agent.run_step()
        if road_option == RoadOption.LEFT:
            world.set_command('left')
        elif road_option == RoadOption.LANEFOLLOW or road_option == RoadOption.STRAIGHT:
            world.set_command('forward')
        elif road_option == RoadOption.RIGHT:
            world.set_command('right')

        if self._autopilot_enabled:
            world.player.apply_control(control)
        else:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0

            # set automatic control-related vehicle lights
            if self._control.brake:
                current_lights |= carla.VehicleLightState.Brake
            else:  # remove the Brake flag
                current_lights &= ~carla.VehicleLightState.Brake
            if self._control.reverse:
                current_lights |= carla.VehicleLightState.Reverse
            else:  # remove the Reverse flag
                current_lights &= ~carla.VehicleLightState.Reverse
            if current_lights != self._lights:  # change the light state only if necessary
                self._lights = current_lights
                world.player.set_light_state(carla.VehicleLightState(self._lights))

            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)


# ------------------------------------------------------------------------------

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % 'Model 3',
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            '',
        ]
        self._info_text += [
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse)]

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # at this point has to be a str
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18


# ------------------------------------------------------------------------------


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.obs = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            (carla.Transform(carla.Location(x=+0.8 * bound_x, y=+0.0 * bound_y, z=1.3 * bound_z)),
             Attachment.Rigid),
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)),
             Attachment.SpringArm),
        ]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', None, 'Camera RGB', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
            for attr_name, attr_value in item[3].items():
                bp.set_attribute(attr_name, attr_value)

            item.append(bp)
        self.index = None

    def set_sensor(self, index, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None

            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # we need to pass the lambda a weak reference to self to avoid circular reference
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        self.index = index

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, force_respawn=True)

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.obs = array
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


# ------------------------------------------------------------------------------

def game_loop(args):
    if args.no_screen:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    pygame.init()
    world = None

    try:
        # connect to carla server
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
    finally:
        pass

    for eps in range(args.num_episodes * len(args.spawns)):

        pygame.font.init()
        args.episode_number = eps
        idx = eps // args.num_episodes

        args.spawn = args.spawns[idx]
        args.dest = args.dests[idx]
        args.rotation = args.rotations[idx]

        # add random shift to vehicle spawn location
        if args.num_episodes > 1 and eps % args.num_episodes != 0:
            shift = 5 * random.random()
            if args.rotation.yaw == 0:
                args.spawn.x += shift
            elif args.rotation.yaw == 90:
                args.spawn.y += shift
            elif args.rotation.yaw == 180:
                args.spawn.x -= shift
            elif args.rotation.yaw == 270:
                args.spawn.y -= shift

        try:
            # load world
            sim_world = client.load_world(args.map)

            # apply settings
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1 / args.fps
            sim_world.apply_settings(settings)

            # initialize pygame
            display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            display.fill((0, 0, 0))
            pygame.display.flip()

            hud = HUD(args.width, args.height)
            world = World(sim_world, hud, args)
            controller = KeyboardControl(world, autopilot_enabled=args.autopilot)

            # recording
            if args.record:
                world.toggle_recording()

            sim_world.tick()
            clock = pygame.time.Clock()

            while not world.agent.done():
                sim_world.tick()
                clock.tick_busy_loop(args.fps)
                controller.parse_events(world, clock)
                world.tick(clock)
                world.render(display)
                pygame.display.flip()
                world.record()

            if args.record:
                world.toggle_recording()

        finally:
            if world is not None:
                world.destroy()
            pygame.quit()


# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', metavar='H', default='127.0.0.1',
                        help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--port', metavar='P', default=2000, type=int,
                         help='TCP port to listen to (default: 2000)')
    parser.add_argument('--res', metavar='WIDTHxHEIGHT', default='256x144',
                        help='window resolution')
    parser.add_argument('--vehicle', default='model3',
                        help='vehicle name')
    parser.add_argument('--map', default='Town02_Opt',
                        help='map name')
    parser.add_argument('--fps', default=30)
    parser.add_argument('--save-png', type=bool, default=False)
    parser.add_argument('--no-screen', type=bool, default=True)

    parser.add_argument('--record', type=bool, default=True)
    parser.add_argument('--autopilot', type=bool, default=True)
    parser.add_argument('--num-episodes', type=int, default=1)

    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    args.spawns = [
        carla.Location(x=103.0, y=191.9, z=0.5),
        carla.Location(x=161.0, y=187.5, z=0.5),

        carla.Location(x=103.0, y=241.2, z=0.5),
        carla.Location(x=161.0, y=236.7, z=0.5),

        carla.Location(x=153.0, y=191.9, z=0.5),
        carla.Location(x=153.0, y=191.9, z=0.5),

        carla.Location(x=153.0, y=241.2, z=0.5),
        carla.Location(x=153.0, y=241.2, z=0.5),

        carla.Location(x=4.7, y=191.9, z=0.5),
        carla.Location(x=65.8, y=187.4, z=0.5),

        carla.Location(x=41.7, y=210.0, z=0.5),
        carla.Location(x=45.8, y=278.2, z=0.5),
    ]

    args.rotations = [
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),

        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),

        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),

        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),

        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),

        carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),
        carla.Rotation(pitch=0.0, yaw=270.0, roll=0.0),
    ]

    args.dests = [
        carla.Location(x=161.0, y=191.9, z=0.0),
        carla.Location(x=133.0, y=187.5, z=0.0),

        carla.Location(x=161.0, y=241.2, z=0.0),
        carla.Location(x=133.0, y=236.7, z=0.0),

        carla.Location(x=189.9, y=218.0, z=0.0),
        carla.Location(x=193.9, y=170.0, z=0.0),

        carla.Location(x=189.9, y=267.0, z=0.0),
        carla.Location(x=193.9, y=220.0, z=0.0),

        carla.Location(x=41.7, y=217.9, z=0.0),
        carla.Location(x=41.7, y=217.9, z=0.0),

        carla.Location(x=71.8, y=241.2, z=0.0),
        carla.Location(x=71.8, y=241.2, z=0.0),
    ]

    print(f'listening to server {args.host}:{args.port}')

    game_loop(args)
