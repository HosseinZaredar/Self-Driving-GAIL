import carla

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    print('connected.')
    world = client.load_world('Town02_Opt')
    world.unload_map_layer(carla.MapLayer.StreetLights)
    world.unload_map_layer(carla.MapLayer.Foliage)
    world.unload_map_layer(carla.MapLayer.Particles)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)

finally:
    print('disconnected.')

