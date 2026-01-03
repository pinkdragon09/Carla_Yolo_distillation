#!/usr/bin/env python3
# CARLA 0.9.14 — capture front-camera images near a YELLOW traffic light
# Adds realistic objects: traffic vehicles (autopilot), pedestrians, bikes
# Output: all images go into out/yellowlight/ with filename including weather, distance, speed.

import sys
import math
import time
import queue
import random
import argparse
from pathlib import Path

import carla

MPH_TO_MPS = 0.44704

# -------------------------
# World settings helpers
# -------------------------
def set_sync(world, fixed_dt=0.05):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_dt
    settings.substepping = True
    settings.max_substep_delta_time = fixed_dt / 2
    settings.max_substeps = 10
    world.apply_settings(settings)
    return settings

def restore_settings(world, old):
    world.apply_settings(old)

def forward_vec(transform):
    v = transform.get_forward_vector()
    return carla.Vector3D(v.x, v.y, v.z)

def distance(a: carla.Location, b: carla.Location):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def flatten(lists):
    out = []
    for x in lists:
        if isinstance(x, (list, tuple)):
            out.extend(x)
        else:
            out.append(x)
    return out

# -------------------------
# Blueprints
# -------------------------
def pick_vehicle_blueprint(bp_lib):
    for name in ["vehicle.tesla.model3", "vehicle.lincoln.mkz_2020", "vehicle.audi.tt", "vehicle.bmw.grandtourer"]:
        try:
            return bp_lib.find(name)
        except Exception:
            pass
    return random.choice(bp_lib.filter("vehicle.*"))

def make_front_rgb_camera(bp_lib, width, height, fov_deg, sensor_tick="0"):
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov_deg))
    cam_bp.set_attribute("sensor_tick", sensor_tick)
    return cam_bp

def find_walker_controller_bp(bp_lib):
    """
    CARLA 0.9.14 usually uses 'controller.ai.walker'.
    Some environments might expose slightly different names.
    """
    candidates = [
        "controller.ai.walker",
        "walker.controller.ai",   # some code examples use this, but often not present in 0.9.14
        "controller.ai.walker",   # duplicate on purpose, harmless
    ]
    for c in candidates:
        try:
            return bp_lib.find(c)
        except Exception:
            continue
    return None

def choose_bike_blueprints(bp_lib):
    """
    Bikes may or may not exist depending on your CARLA content pack.
    We'll try common-ish patterns; if none found, return empty list.
    """
    bikes = []
    for pat in ["*bike*", "*bicycle*", "*crossbike*"]:
        bikes.extend(bp_lib.filter(f"vehicle.*{pat}"))
    # de-dup by id
    seen = set()
    uniq = []
    for b in bikes:
        if b.id not in seen:
            seen.add(b.id)
            uniq.append(b)
    return uniq

# -------------------------
# Traffic light stop line helpers
# -------------------------
def get_stop_waypoints_safe(tl, town_map):
    stop_wps = []
    if hasattr(tl, "get_stop_waypoints"):
        try:
            stop_wps = tl.get_stop_waypoints()
            stop_wps = flatten(stop_wps)
        except Exception:
            stop_wps = []
    if stop_wps:
        return stop_wps

    # Fallback approximation via trigger volume (less accurate)
    try:
        base_tf = tl.get_transform()
        tl_fwd = base_tf.get_forward_vector()
        approx_stop = carla.Location(
            x=base_tf.location.x - 5.0 * tl_fwd.x,
            y=base_tf.location.y - 5.0 * tl_fwd.y,
            z=base_tf.location.z
        )
        wp = town_map.get_waypoint(approx_stop, project_to_road=True, lane_type=carla.LaneType.Driving)
        return [wp] if wp else []
    except Exception:
        return []

def find_tl_and_stop_wp(world):
    town_map = world.get_map()
    tls = world.get_actors().filter("traffic.traffic_light*")

    for tl in tls:
        stop_wps = get_stop_waypoints_safe(tl, town_map)
        for swp in stop_wps:
            if swp.lane_type != carla.LaneType.Driving:
                continue
            back = swp.previous(100.0)
            if back:
                return tl, swp, back[0]

    for _ in range(2000):
        w = town_map.get_random_waypoint(carla.LaneType.Driving)
        tl = w.get_traffic_light() if hasattr(w, "get_traffic_light") else None
        if tl:
            stop_wps = get_stop_waypoints_safe(tl, town_map)
            if stop_wps:
                swp = min(stop_wps, key=lambda wp: distance(w.transform.location, wp.transform.location))
                back = swp.previous(100.0)
                if back:
                    return tl, swp, back[0]

    raise RuntimeError("Could not find a usable traffic light with a stop waypoint and 100 m approach.")

# -------------------------
# Spawning realism: traffic, pedestrians, bikes
# -------------------------
def spawn_traffic_vehicles(world, client, traffic_manager, bp_lib, num_vehicles, avoid_near_location=None, min_dist=25.0):
    spawned = []
    if num_vehicles <= 0:
        return spawned

    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)

    vehicle_bps = bp_lib.filter("vehicle.*")
    for sp in spawn_points:
        if len(spawned) >= num_vehicles:
            break
        if avoid_near_location is not None and distance(sp.location, avoid_near_location) < min_dist:
            continue
        bp = random.choice(vehicle_bps)
        # optional: make them not all the same color
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

        veh = world.try_spawn_actor(bp, sp)
        if veh:
            spawned.append(veh)

    # enable autopilot
    for v in spawned:
        try:
            v.set_autopilot(True, traffic_manager.get_port())
        except Exception:
            try:
                v.set_autopilot(True)
            except Exception:
                pass

    # a few TM tweaks
    try:
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.global_percentage_speed_difference(0.0)  # 0 = normal speed
    except Exception:
        pass

    return spawned

def spawn_bikes(world, bp_lib, num_bikes, avoid_near_location=None, min_dist=20.0):
    spawned = []
    if num_bikes <= 0:
        return spawned

    bike_bps = choose_bike_blueprints(bp_lib)
    if not bike_bps:
        # If your CARLA build doesn't include bike blueprints, we just skip bikes cleanly.
        print("INFO: No bike blueprints found in this CARLA content pack. Skipping bikes.")
        return spawned

    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)

    for sp in spawn_points:
        if len(spawned) >= num_bikes:
            break
        if avoid_near_location is not None and distance(sp.location, avoid_near_location) < min_dist:
            continue

        bp = random.choice(bike_bps)
        bike = world.try_spawn_actor(bp, sp)
        if bike:
            spawned.append(bike)

    return spawned

def spawn_walkers(world, bp_lib, num_walkers):
    walkers = []
    controllers = []
    if num_walkers <= 0:
        return walkers, controllers

    controller_bp = find_walker_controller_bp(bp_lib)
    if controller_bp is None:
        print("WARN: Walker controller blueprint not found (expected 'controller.ai.walker'). Skipping pedestrians.")
        return walkers, controllers

    walker_bps = bp_lib.filter("walker.pedestrian.*")
    if not walker_bps:
        print("WARN: No pedestrian blueprints found. Skipping pedestrians.")
        return walkers, controllers

    # Choose random nav locations
    spawn_locs = []
    for _ in range(num_walkers * 3):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_locs.append(loc)
        if len(spawn_locs) >= num_walkers:
            break

    for loc in spawn_locs:
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        walker_actor = world.try_spawn_actor(bp, carla.Transform(loc))
        if walker_actor:
            walkers.append(walker_actor)

    # Spawn controllers for walkers
    for w in walkers:
        ctrl = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=w)
        if ctrl:
            controllers.append(ctrl)

    return walkers, controllers

def start_walkers(world, walkers, controllers, seed=None):
    if seed is not None:
        random.seed(seed)

    for ctrl in controllers:
        try:
            ctrl.start()
            dest = world.get_random_location_from_navigation()
            if dest:
                ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.0 + random.random() * 1.5)  # ~1–2.5 m/s
        except Exception:
            pass

# -------------------------
# Weather presets
# -------------------------
def build_weather_presets():
    wp = carla.WeatherParameters
    preset_lookup = {
        name: getattr(wp, name)
        for name in dir(wp)
        if not name.startswith("_") and isinstance(getattr(wp, name), wp)
    }

    # A curated set including evening/sunset, rain, fog
    preferred = [
        "ClearSunset",
        "CloudySunset",
        "CloudyNoon",
        "ClearNoon",
      ]

    out = []
    for name in preferred:
        if name in preset_lookup:
            out.append((name, preset_lookup[name]))

    # fallback safety
    if not out and "ClearNoon" in preset_lookup:
        out = [("ClearNoon", preset_lookup["ClearNoon"])]

    return out

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=8000)

    ap.add_argument("--town", default="", help="Optional: TownXX to load (e.g., Town03). Leave blank to keep current.")
    ap.add_argument("--speed_mph", type=float, default=30.0)
    ap.add_argument("--distance_m", type=float, default=25.0)

    ap.add_argument("--frames", type=int, default=150, help="Frames to save")
    ap.add_argument("--hz", type=float, default=20.0)

    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=90.0)

    ap.add_argument("--outdir", default="out/yellowlight_30m_30mph", help="All images go into this ONE folder")

    # realism knobs
    ap.add_argument("--num_vehicles", type=int, default=25)
    ap.add_argument("--num_walkers", type=int, default=40)
    ap.add_argument("--num_bikes", type=int, default=6)

    # weather selection
    ap.add_argument("--weathers", nargs="*", default=[], help="Optional override list, e.g. FoggyNoon SoftRainSunset")

    args = ap.parse_args()

    speed_mps = args.speed_mph * MPH_TO_MPS
    fixed_dt = 1.0 / args.hz

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_synchronous_mode(True)

    world = client.get_world()
    if args.town:
        world = client.load_world(args.town)
        world.tick()

    bp_lib = world.get_blueprint_library()
    town_map = world.get_map()

    img_dir = Path(args.outdir)
    img_dir.mkdir(parents=True, exist_ok=True)

    old_settings = set_sync(world, fixed_dt=fixed_dt)

    # Weather presets
    weather_presets = build_weather_presets()
    wp = carla.WeatherParameters
    preset_lookup = {name: param for (name, param) in weather_presets}

    chosen_weathers = []
    if args.weathers:
        for name in args.weathers:
            if name in preset_lookup:
                chosen_weathers.append((name, preset_lookup[name]))
        if not chosen_weathers:
            chosen_weathers = weather_presets[:]
    else:
        chosen_weathers = weather_presets[:]

    actor_list = []
    q = queue.Queue()

    try:
        # Find traffic light + approach waypoint
        tl, stop_wp, back_wp = find_tl_and_stop_wp(world)

        # Set TL to YELLOW and freeze so it stays yellow
        tl.set_state(carla.TrafficLightState.Yellow)
        if hasattr(tl, "freeze"):
            tl.freeze(True)

        # Spawn ego vehicle
        veh_bp = pick_vehicle_blueprint(bp_lib)
        veh_tf = back_wp.transform
        veh_tf.location.z += 0.3

        ego = world.try_spawn_actor(veh_bp, veh_tf)
        if ego is None:
            veh_tf.location.x += 0.5
            veh_tf.location.y += 0.5
            ego = world.try_spawn_actor(veh_bp, veh_tf)
        if ego is None:
            raise RuntimeError("Failed to spawn the ego vehicle at the chosen waypoint.")
        actor_list.append(ego)

        # Attach front RGB camera
        cam_bp = make_front_rgb_camera(bp_lib, args.width, args.height, args.fov, sensor_tick="0")
        cam_rel = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-2.0))
        cam = world.spawn_actor(cam_bp, cam_rel, attach_to=ego)
        actor_list.append(cam)

        cam.listen(lambda image: q.put(image))

        # Place ego exactly distance_m behind stop line if possible
        exact = stop_wp.previous(args.distance_m)
        if exact:
            ego.set_transform(exact[0].transform)
            world.tick()

        # Give ego target velocity (straight lane direction)
        fwd = forward_vec(ego.get_transform())
        ego.set_target_velocity(carla.Vector3D(fwd.x * speed_mps, fwd.y * speed_mps, 0))

        # Spawn realism actors away from ego start area to reduce collisions at spawn
        ego_loc = ego.get_transform().location

        traffic = spawn_traffic_vehicles(world, client, traffic_manager, bp_lib,
                                         num_vehicles=args.num_vehicles,
                                         avoid_near_location=ego_loc, min_dist=35.0)
        actor_list.extend(traffic)

        bikes = spawn_bikes(world, bp_lib, num_bikes=args.num_bikes,
                            avoid_near_location=ego_loc, min_dist=35.0)
        actor_list.extend(bikes)

        walkers, controllers = spawn_walkers(world, bp_lib, num_walkers=args.num_walkers)
        actor_list.extend(controllers)
        actor_list.extend(walkers)

        # Start walker AI after one tick (controllers need to exist in-world)
        world.tick()
        start_walkers(world, walkers, controllers, seed=42)

        # Warm-up ticks
        for _ in range(5):
            world.tick()

        # Capture loop
        seq = 0
        for wname, wparam in chosen_weathers:
            world.set_weather(wparam)
            world.tick()
            world.tick()

            for _ in range(args.frames):
                world.tick()
                try:
                    img = q.get(timeout=2.0)
                except queue.Empty:
                    print("WARN: no image received this tick")
                    continue

                cur_d = distance(ego.get_transform().location, stop_wp.transform.location)
                # filename includes weather, distance, speed
                fname = (
                    f"{seq:06d}"
                    f"_weather-{wname}"
                    f"_dist-{cur_d:05.2f}m"
                    f"_speed-{args.speed_mph:05.1f}mph.png"
                )
                img.save_to_disk(str(img_dir / fname))
                seq += 1

        print(f"Done. Saved {seq} frames to: {img_dir.resolve()}")

    finally:
        # Cleanup
        for a in actor_list[::-1]:
            try:
                a.destroy()
            except Exception:
                pass
        try:
            restore_settings(world, old_settings)
        except Exception:
            pass
        try:
            tl.freeze(False)
        except Exception:
            pass
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
