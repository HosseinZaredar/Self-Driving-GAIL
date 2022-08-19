import carla

train_spawns = [
    carla.Location(x=103.0, y=191.9, z=0.5),
    carla.Location(x=161.0, y=187.5, z=0.5),

    carla.Location(x=103.0, y=241.2, z=0.5),
    carla.Location(x=161.0, y=236.7, z=0.5),

    carla.Location(x=153.0, y=191.9, z=0.5),
    carla.Location(x=153.0, y=191.9, z=0.5),

    carla.Location(x=193.9, y=218.0, z=0.5),
    carla.Location(x=189.9, y=170.0, z=0.5),

    carla.Location(x=153.0, y=241.2, z=0.5),
    carla.Location(x=153.0, y=241.2, z=0.5),

    carla.Location(x=193.9, y=267.0, z=0.5),
    carla.Location(x=189.9, y=220.0, z=0.5),

    carla.Location(x=4.7, y=191.9, z=0.5),
    carla.Location(x=65.8, y=187.4, z=0.5),

    carla.Location(x=46.2, y=217.9, z=0.5),
    carla.Location(x=46.2, y=217.9, z=0.5),

    carla.Location(x=41.7, y=210.0, z=0.5),
    carla.Location(x=45.8, y=278.2, z=0.5),

    carla.Location(x=71.8, y=236.7, z=0.5),
    carla.Location(x=71.8, y=236.7, z=0.5),
]

train_rotations = [
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),

    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),

    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),

    carla.Rotation(pitch=0.0, yaw=270.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),

    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),

    carla.Rotation(pitch=0.0, yaw=270.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),

    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),

    carla.Rotation(pitch=0.0, yaw=270.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=270.0, roll=0.0),

    carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=270.0, roll=0.0),

    carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0),
]

train_dests = [
    carla.Location(x=161.0, y=191.9, z=0.0),
    carla.Location(x=103.0, y=187.5, z=0.0),

    carla.Location(x=161.0, y=241.2, z=0.0),
    carla.Location(x=103.0, y=236.7, z=0.0),

    carla.Location(x=189.9, y=218.0, z=0.0),
    carla.Location(x=193.9, y=170.0, z=0.0),

    carla.Location(x=153.0, y=187.5, z=0.0),
    carla.Location(x=153.0, y=187.5, z=0.0),

    carla.Location(x=189.9, y=267.0, z=0.0),
    carla.Location(x=193.9, y=220.0, z=0.0),

    carla.Location(x=153.0, y=236.7, z=0.0),
    carla.Location(x=153.0, y=236.7, z=0.0),

    carla.Location(x=41.7, y=217.9, z=0.0),
    carla.Location(x=41.7, y=217.9, z=0.0),

    carla.Location(x=65.8, y=191.9, z=0.0),
    carla.Location(x=4.7, y=187.4, z=0.0),

    carla.Location(x=71.8, y=241.2, z=0.0),
    carla.Location(x=71.8, y=241.2, z=0.0),

    carla.Location(x=45.8, y=210.0, z=0.0),
    carla.Location(x=41.7, y=278.2, z=0.0),
]

# ---------------------------------------------------------------

test_spawns = [
    carla.Location(x=103.0, y=191.9, z=0.5),
    carla.Location(x=103.0, y=241.2, z=0.5),
    carla.Location(x=189.9, y=170.0, z=0.5)
]

test_rotations = [
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),
]

test_dests = [
    carla.Location(x=103.0, y=236.7, z=0.0),
    carla.Location(x=189.9, y=218.0, z=0.0),
    carla.Location(x=189.9, y=218.0, z=0.0),
]
