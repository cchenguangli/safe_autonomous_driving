#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, math
import carla, torch

import numpy as np

from stable_baselines3 import SAC
# from stable_baselines3 import PPO
from sac_env import CarlaEnv

# ---- pygame &  ----
try:
    import pygame
    from pygame.locals import K_ESCAPE
except Exception:
    pygame = None

from carla import ColorConverter as cc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="sac00/best_model.zip")
    # p.add_argument("--model-path", type=str, default="best_model.zip")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--slow", action="store_true", help="sleep a bit each step for easier viewing")
    p.add_argument("--trail", action="store_true", help="draw driven trail (slower if very long)")
    p.add_argument("--draw-world", action="store_true", help="draw GREEN waypoint trail in CARLA world")
    return p.parse_args()



def draw_green_waypoints_in_world(world, env,
                                  n_points=40,     
                                  step_dist=2.0,    
                                  life_time=0.35,
                                  size=0.08,
                                  z_offset=0.15):
    """Strictly follow env.route and plot points forward starting from the current last_waypoint_idx."""
    route = getattr(env, "route", None)
    if not route:
        return

    
    i0 = int(max(0, min(getattr(env, "last_waypoint_idx", 0), len(route) - 1)))

    
    res = float(getattr(env, "sampling_resolution", 1.0)) or 1.0
    stride = max(1, int(round(step_dist / res)))  

    i_end = min(len(route), i0 + n_points * stride)
    for i in range(i0, i_end, stride):
        wp, _ = route[i]
        loc = wp.transform.location
        world.debug.draw_point(
            carla.Location(loc.x, loc.y, loc.z + z_offset),
            size=size, life_time=life_time, color=carla.Color(0, 255, 0)
        )



def route_signature(env):
    r = getattr(env, "route", None)
    if not r:
        return (0, 0.0, 0.0)
    first = r[0][0].transform.location
    return (len(r), round(first.x, 2), round(first.y, 2))



class SimpleCameraManager:
    def __init__(self, world, parent_actor, width=1280, height=720, gamma=2.2):
        self.world = world
        self.parent = parent_actor
        self.width = width
        self.height = height
        self.gamma = gamma
        self.sensor = None
        self.surface = None
        self._spawn_sensor()

    def _spawn_sensor(self):
        if self.sensor is not None:
            try:
                self.sensor.stop()
                self.sensor.destroy()
            except Exception:
                pass
            self.sensor = None
            self.surface = None

        bp_library = self.world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        if bp.has_attribute('gamma'):
            bp.set_attribute('gamma', str(self.gamma))

        transform = carla.Transform(
            carla.Location(x=-6.0, z=3.5),
            carla.Rotation(pitch=12.0)
        )
        attachment = carla.AttachmentType.SpringArmGhost

        self.sensor = self.world.spawn_actor(
            bp,
            transform,
            attach_to=self.parent,
            attachment_type=attachment
        )

        def _on_image(image):
            image.convert(cc.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3][:, :, ::-1]   # BGRA -> RGB
            if pygame:
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        self.sensor.listen(_on_image)

    def destroy(self):
        if self.sensor is not None:
            try:
                self.sensor.stop()
                self.sensor.destroy()
            except Exception:
                pass
            self.sensor = None
        self.surface = None


# ---------- draw HUD ----------
def draw_hud(display, font, clock, env, scan, tf, step, ep, total_episodes):
    """
    Information Panel (Left)：
    - Speed
    - ego_gapF / ego_ttcF / left_ok / right_ok / ego_lane / ped_gap
    - TL distance、Stopline distance
    - NPC number
    - lat_error / d_e, yaw_diff
    - position and（steer/throttle/brake）
    """
#    panel_width = 460
#    panel_height = 260
       
    screen_w, screen_h = display.get_size()
    panel_width = int(screen_w * 0.25)   
    panel_height = int(screen_h * 0.9)  
    panel = pygame.Surface((panel_width, panel_height))
    panel.set_alpha(140)
    panel.fill((0, 0, 0))
    display.blit(panel, (10, 10))

    speed_ms = env.get_speed()
    speed_kmh = speed_ms * 3.6
    lat_error = env.get_lat_error()
    yaw_diff = env.get_yaw_diff()

   
    d_e = float(scan.get("d_e", lat_error))
    ego_lane = int(scan.get("ego_lane", 0))
    gapF = float(scan.get("ego_gapF", 0.0))
    ttcF = float(scan.get("ego_ttcF", 0.0))
    left_ok = int(scan.get("left_ok", 0))
    right_ok = int(scan.get("right_ok", 0))
    ped_gap = float(scan.get("ped_gap", env.GAP_CLIP))

  
    aff_tl, sdist_tl, tl_state, _ = env._red_light_info(lookahead_points=getattr(env, "TL_LOOKAHEAD", 50))
    aff_s, sdist_s, _, _ = env._stop_info(lookahead_points=getattr(env, "STOP_LOOKAHEAD", 50))

    if not aff_tl or sdist_tl is None:
        tl_str = "--"
    else:
        tl_str = f"{sdist_tl:5.2f} m"

    if not aff_s or sdist_s is None:
        stop_str = "--"
    else:
        stop_str = f"{sdist_s:5.2f} m"

    
    npc_count = len(getattr(env, "npc_vehicles", []))

    
    steer = float(getattr(env, "last_steer", 0.0))
    long_cmd = float(getattr(env, "last_long_cmd", 0.0))
    throttle = max(0.0, long_cmd)
    brake = max(0.0, -long_cmd)

#    lines = [
#        f"Episode {ep}/{total_episodes}  Step {step}",
#        f"FPS: {clock.get_fps():5.1f}",
#        f"Speed: {speed_kmh:6.2f} km/h   NPCs: {npc_count}",
#        f"Pos: x={tf.location.x:7.2f}  y={tf.location.y:7.2f}  z={tf.location.z:5.2f}",
#        f"Yaw: {tf.rotation.yaw:6.1f} deg | yaw_diff={yaw_diff: .3f} rad",
#        f"err(lat)={lat_error: .3f} m  d_e={d_e: .3f} m  lane={ego_lane}",
#        f"gapF={gapF:6.2f} m  ttcF={ttcF:4.2f} s",
#        f"left_ok={left_ok}  right_ok={right_ok}  ped_gap={ped_gap:5.2f} m",
#        f"dist_TL={tl_str}   dist_stop={stop_str}",
#        f"ctrl: steer={steer: .3f}  throttle={throttle: .3f}  brake={brake: .3f}",
#    ]

#    y = 18
#    for line in lines:
#        surf = font.render(line, True, (255, 255, 255))
#        display.blit(surf, (24, y))
#        y += 22
    lines = [
        # ----- episode & FPS -----
        f"Episode {ep}/{total_episodes}  Step {step}",
        "",
        f"FPS: {clock.get_fps():5.1f}",
        "",

        
        f"Speed: {speed_kmh:6.2f} km/h",
        f"NPCs: {npc_count}",
        "",

       
        f"Pos: x={tf.location.x:7.2f}",
        f"     y={tf.location.y:7.2f}  z={tf.location.z:5.2f}",
        "",

       
        f"Yaw: {tf.rotation.yaw:6.1f} deg",
        f"yaw_diff={yaw_diff: .3f} rad",
        "",

        
        f"err(lat)={lat_error: .3f} m",
        f"d_e={d_e: .3f} m   lane={ego_lane}",
        "",

        
        f"gapF={gapF:6.2f} m",
        f"ttcF={ttcF:4.2f} s",
        "",

        
        f"left_ok={left_ok}  right_ok={right_ok}",
        f"ped_gap={ped_gap:5.2f} m",
        "",

        
        f"dist_TL={tl_str}",
        f"dist_stop={stop_str}",
        "",

        
        f"ctrl: steer={steer: .3f}",
        f"      throttle={throttle: .3f}",
        f"      brake={brake: .3f}",
    ]

   
    top_margin = 20
    bottom_margin = 20
    n_lines = len(lines)
    usable_h = panel_height - top_margin - bottom_margin

    if n_lines > 1:
        step = usable_h / (n_lines - 1)
    else:
        step = 0

    for i, line in enumerate(lines):
        surf = font.render(line, True, (255, 255, 255))
        x = 10 + 14
        y = 10 + top_margin + int(i * step)
        display.blit(surf, (x, y))


  
    top_margin = 20           
    bottom_margin = 20
    n_lines = len(lines)
    usable_h = panel_height - top_margin - bottom_margin


    if n_lines > 1:
        step = usable_h / (n_lines - 1)
    else:
        step = 0

    for i, line in enumerate(lines):
        surf = font.render(line, True, (255, 255, 255))
        x = 10 + 14
        y = 10 + top_margin + int(i * step)
        display.blit(surf, (x, y))




if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Loading model on {device} ...")
    model = SAC.load(args.model_path, device=device)
    # model = PPO.load(args.model_path, device=device)
    print("[Info] Model loaded.\n")

    if pygame is None:
        raise RuntimeError("need pygame：pip install pygame")

    
    env = CarlaEnv(render=True, auto_reset=False)

    
    width, height = 1280, 720
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA RL Evaluation with HUD")
    font = pygame.font.Font(pygame.font.get_default_font(), 18)
    clock = pygame.time.Clock()

    
    cam = SimpleCameraManager(env.world, env.vehicle, width, height, gamma=2.2)

   

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()

            
            if cam is not None:
                cam.destroy()
            cam = SimpleCameraManager(env.world, env.vehicle, width, height, gamma=2.2)

            
            env.enable_rendering(True)
            world = env.world
            spectator = world.get_spectator()


           

            print(f"=== Episode {ep}/{args.episodes} ===")
            step = 0
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt()
                    elif event.type == pygame.KEYUP and event.key == K_ESCAPE:
                        raise KeyboardInterrupt()

               
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated) or bool(truncated)


             
                veh = env.vehicle
                if veh is not None and spectator is not None:
                    try:
                        tf = veh.get_transform()
                        yaw_rad = math.radians(tf.rotation.yaw)
                        ofs = carla.Location(
                            x=-10 * math.cos(yaw_rad),
                            y=-10 * math.sin(yaw_rad),
                            z=7,
                        )
                        spec_tf = carla.Transform(
                            tf.location + ofs,
                            carla.Rotation(pitch=-20, yaw=tf.rotation.yaw, roll=0),
                        )
                        spectator.set_transform(spec_tf)

                       
                        if args.draw_world:
                            draw_green_waypoints_in_world(world, env, n_points=40, step_dist=2.0, life_time=0.35)

                       
                        display.fill((0, 0, 0))
                        if cam is not None and cam.surface is not None:
                            display.blit(cam.surface, (0, 0))

                        scan = env._scan_surroundings()
                        draw_hud(display, font, clock, env, scan, tf, step, ep, args.episodes)

                        pygame.display.flip()

                       
                        print(
                            f"step={step:4d} | action=(steer={action[0]: .3f}, thr={action[1]: .3f}) | "
                            f"pos=({tf.location.x: .2f},{tf.location.y: .2f})   ",
                            end="\r",
                            flush=True,
                        )
                    except RuntimeError:                       
                        pass

                step += 1
                if args.slow:
                    time.sleep(0.04)

                clock.tick_busy_loop(60)

            print(f"\n[Episode {ep}] finished after {step} steps\n")

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
    finally:
        try:
            if cam is not None:
                cam.destroy()
        except Exception:
            pass

        try:
            env.close()
        except Exception:
            pass

        try:
            pygame.quit()
        except Exception:
            pass

        print("[Info] Environment closed.")

