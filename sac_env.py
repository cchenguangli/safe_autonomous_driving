import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
import random

import sys, os

carla_path = os.path.expanduser('~/carla/PythonAPI/carla')
if carla_path not in sys.path:
    sys.path.insert(0, carla_path)
from agents.navigation.global_route_planner import GlobalRoutePlanner


SYNCHRONOUS_MODE = True
FIXED_DELTA_SECONDS = 0.04


class CarlaEnv(gym.Env):
    """
    CARLA RL Env: Point-to-Point Safe Navigation in Urban Environments
Speed units have been standardized to m/s
    
    Action: [steer ∈ [-1,1], long_cmd ∈ [-1,1]]  # Positive = Throttle, Negative = Brake (Mutually Exclusive)
    """
    metadata = {"render_modes": ["human"]}

#    def __init__(self, host='localhost', port=2000, town='Town05',
    def __init__(self, host='localhost', port=2000, town='Town03',
                 render=False, auto_reset=True):
        super().__init__()

        
        self.last_steer = 0.0

        self.render = render
        self.auto_reset = auto_reset

        # CARLA Client & World
        self.client = carla.Client(host, port)
        self.client.set_timeout(45.0)
        self.world  = self.client.load_world(town)
        self.map    = self.world.get_map()

        settings    = self.world.get_settings()
        settings.no_rendering_mode  = (not self.render)
        settings.synchronous_mode    = SYNCHRONOUS_MODE
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        settings.actor_active_distance = 1000.0
        
#        settings.tile_stream_distance = 1000.0
        self.world.apply_settings(settings)
        self.world.tick()
        


        # Clear the scene
        self._clear_actors()
        
        # Traffic Manager
        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(SYNCHRONOUS_MODE)
        self.tm.global_percentage_speed_difference(0.0)  
        self.tm_port = self.tm.get_port()

        # NPC Cache and Quantity
        self.npc_vehicles = []
        self.NUM_NPC = 60

        # Vehicle & Sensors
        self.bp_lib  = self.world.get_blueprint_library()
        self.vehicle = None
        self.collision_sensor = None

        # Road Network Planner
        self.sampling_resolution = 1.0
        self.route_planner = GlobalRoutePlanner(self.map, self.sampling_resolution)
        self.spawn_points   = self.map.get_spawn_points()

        self.max_lat_error = 4.5                # m
        self.max_speed     = 30.0 / 3.6       


        
        # ===== sensing & safety thresholds =====
 #       self.SENSE_RADIUS = 35.0   
        self.GAP_CLIP     = 40.0   
        self.TTC_CLIP     = 6.0    
        self._EPS         = 1e-3
        
        
                
        # Lane Change Safety Threshold (Front/Rear) - Effective Distance for Bumper-to-Bumper Driving
        self.GAP_SAFE_F   = 5.0        
        self.TTC_SAFE_F   = 2.5
        


        # Action/Observation space
        self.action_space = spaces.Box(
            low  = np.array([-1.0, -1.0]),
            high = np.array([ 1.0, 1.0]),
            dtype=np.float32)

#        self.observation_space = spaces.Box(
#            low=np.array([-1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
#            high=np.array([ 1.0, 10.0,  1.0,  1.0,  1.0, 1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0], dtype=np.float32),
#            dtype=np.float32)
            
            
            
        low  = np.array(
            [-3.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            dtype=np.float32
        )
        high = np.array(
            [ 3.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Episode control
        self.max_episode_steps = 3500
        self.step_count        = 0

       
        self.collision_occurred = False
        self.collision_kind = None          # 'pedestrian' | 'vehicle' | 'other'
        self.collision_penalty = 0.0        # Penalty corresponding to this collision
        self.route = None

 
        self.last_waypoint_idx  = 0
        self.offroute_count = 0
        self.stopline_norm_dist = 40.0  # m
        self.RL_STOP_START = 15.0
        self.RL_SENSE_DIST = 40.0      
        self._tl_infos = []  # [(tl_actor, stop_wp, route_idx)]
        self._tl_violation_latched = False  # One-time penalty lock
        self.TL_LOOKAHEAD = 50
        
      
        self.stuck_counter = 0
        self.IDLE_STEP_PENALTY = -0.3   # Penalty for each step when stopping without external cause
        self.STUCK_RESET_STEPS = 50    # How many consecutive steps constitute a stop and reset
        self.STUCK_RESET_PENALTY = -30  # One-time penalty 
        self.IDLE_V_THRESH = 1.0      
       
       
        # ---- route corridor 
        self.COR_AHEAD_M      = 40.0   # # Corridor Front Length (m)
        self.COR_BACK_M  = 20.0      # # Corridor Back Length (m)
        
    
        self.SENSE_RADIUS = 40        
        self.SEG_HYST = 0.1
        self._seg_mem = {}
        
        
        
        self.last_long_cmd = 0.0
        
        # === Crossing pedestrians config (per-episode triggers)
        self.walkers = []                         # track spawned walkers for cleanup
        self.CROSS_SPAWN_MIN = 15.0               # meters ahead (min)
        self.CROSS_SPAWN_MAX = 45.0               # meters ahead (max)
        self.CROSS_PER_EP_RANGE = (3,)          # 4 crossings per episode (random)
        self.CROSS_SPEED_RANGE = (0.5, 1.8)       # walker crossing speed (m/s)
        self.CROSS_LATERAL_RANGE = (3.5, 4.5)     # lateral offset from lane center (m)
        self._cross_triggers = []                 # planned step indices for crossings
        
        
        # ==== STOP sign config & state ====
        self._stop_infos = []              # [(stop_wp, stop_idx)]
        self.STOP_LOOKAHEAD = 50
        self.STOP_HOLD_STEPS = 30          # stop 30 steps
        self._stop_hold = 0
        self._stop_active = False
        self._stopped_idxs = set()         
        self._stop_violation_latched = False
        self._curr_stop_idx = None
        
        self._npc_collision_sensors = []



    def _clear_actors(self):
        vehicles = self.world.get_actors().filter('vehicle.*')
        walkers  = self.world.get_actors().filter('walker.pedestrian.*')
        actors   = list(vehicles) + list(walkers)
        for actor in actors:
            try:
                actor.destroy()
            except:
                pass
        self.world.tick()



    def _spawn_npcs(self, num=None):
        """Generate NPC vehicles: Each vehicle is assigned a random target speed of 6–15 m/s and adheres to traffic rules."""
        if num is None:
            num = self.NUM_NPC
            
        bps_4 = [bp for bp in self.bp_lib.filter('vehicle.*')
                 if bp.has_attribute('number_of_wheels')
                 and int(bp.get_attribute('number_of_wheels')) == 4]
        bps_2 = [bp for bp in self.bp_lib.filter('vehicle.*')
                 if bp.has_attribute('number_of_wheels')
                 and int(bp.get_attribute('number_of_wheels')) == 2]
        target_4 = 54
        target_2 = 6

        

        spawn_points = list(self.map.get_spawn_points())
        random.shuffle(spawn_points)

        ego_loc = self.vehicle.get_transform().location
        spawned_4 = 0
        spawned_2 = 0
        new_npcs = []
        
        for tf in spawn_points:
            # Avoid washing your car near your own vehicle.
            if tf.location.distance(ego_loc) < 8.0:
                continue

            if spawned_4 < target_4:
                bp = random.choice(bps_4)
                wheels_type = 4
            elif spawned_2 < target_2:
                bp = random.choice(bps_2)
                wheels_type = 2
            else:
                break


            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
            bp.set_attribute('role_name', 'autopilot')

            v = self.world.try_spawn_actor(bp, tf)
            if v is None:
                continue

            v.set_autopilot(True, self.tm_port)
            
            try:
                col_bp = self.bp_lib.find('sensor.other.collision')
                col_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=v)
               
                col_sensor.listen(lambda e, self=self: self._on_npc_collision(e))
                self._npc_collision_sensors.append(col_sensor)
            except Exception:
                pass            
            
     
            
            self.npc_vehicles.append(v)  
            new_npcs.append(v) 
            
            if wheels_type == 4:
                spawned_4 += 1
            else:
                spawned_2 += 1

            if spawned_4 >= target_4 and spawned_2 >= target_2:
                break


        # Set a random target speed of 6 to 15 m/s for each vehicle.
        for v in new_npcs:
            
            
            v_des = random.uniform(6.0, 15.0)

            self.tm.set_desired_speed(v, v_des)
           

            self.tm.vehicle_percentage_speed_difference(v, 0.0)
            self.tm.ignore_lights_percentage(v, 0)
            self.tm.auto_lane_change(v, True)

        self.world.tick()
               
                
                
    def _destroy_npcs(self):
        """Destroy NPC vehicles generated in the previous round"""
        if getattr(self, "npc_vehicles", None):
            for v in self.npc_vehicles:
                try:
                    v.destroy()
                except:
                    pass            
            self.npc_vehicles = []
        if getattr(self, "_npc_collision_sensors", None):
            for s in self._npc_collision_sensors:
                try:
                    s.stop()
                    s.destroy()
                except:
                    pass
            self._npc_collision_sensors = []
        self.world.tick()
            
    def _destroy_walkers(self):
        """Destroy walkers spawned by this env (called on reset/close)."""
        if getattr(self, "walkers", None):
            for w in self.walkers:
                try:
                    w.destroy()
                except:
                    pass
            self.walkers = []
            self.world.tick()


    
    def _spawn_crossing_walker_ahead(self, d=None, side=None, speed=None, lateral=None, precheck=True):
        
        if self.vehicle is None or self.route is None:
            return False

        
        current_idx = max(0, int(self.last_waypoint_idx))
        
        
        if d is None:
            idx_offset = random.randint(15, 45) 
        else:
            
            idx_offset = int(d)

        target_idx = current_idx + idx_offset

       
        if target_idx >= len(self.route) - 1:
            return False

       
        target_wp = self.route[target_idx][0] 
        wp_tf = target_wp.transform
        
        
        fwd = wp_tf.get_forward_vector()
       
        right = carla.Vector3D(fwd.y, -fwd.x, 0.0) 

        
        if side is None:
            side = 'right' if random.random() < 0.5 else 'left'
        if speed is None:
            speed = random.uniform(*self.CROSS_SPEED_RANGE)
        if lateral is None:
            lateral = random.uniform(*self.CROSS_LATERAL_RANGE)

        
        sgn = +1.0 if side == 'right' else -1.0
        
        start = carla.Location(
            x=wp_tf.location.x + right.x * (sgn * float(lateral)),
            y=wp_tf.location.y + right.y * (sgn * float(lateral)),
            z=wp_tf.location.z + 0.5
        )

        
        for _ in range(10):
            nav = self.world.get_random_location_from_navigation()
            if nav is not None and nav.distance(start) < 3.0:
                start = carla.Location(nav.x, nav.y, nav.z + 0.5)
                break
                
        if precheck:
            
            pts, S = self._corridor_polyline(L_f=self.COR_AHEAD_M, L_b=self.COR_BACK_M)
            if pts is None: return False
            
           
            s_c, d_c, _, _ = self._project_to_poly_stable(float(start.x), float(start.y), pts, S, mem_key=None)
            
            
            ego_loc = self.vehicle.get_location()
            s_e, _, _, _ = self._project_to_poly_stable(ego_loc.x, ego_loc.y, pts, S, mem_key=None)
            
            if s_c is None or s_e is None: return False

            s_rel = s_c - s_e
           
            if s_rel < 5.0 or s_rel > self.COR_AHEAD_M + 5.0:
                return False

        
        bp_list = self.bp_lib.filter('walker.pedestrian.*')
        if not bp_list: return False
        wbp = random.choice(bp_list)
        try:
            wbp.set_attribute('is_invincible', 'false')
        except: pass

        
        walker_yaw = wp_tf.rotation.yaw + (-90 if side == 'right' else 90)
        
        walker = self.world.try_spawn_actor(wbp, carla.Transform(start, carla.Rotation(yaw=walker_yaw)))
        if walker is None:
            return False

        
        ctrl = carla.WalkerControl()
        ctrl.speed = float(speed)
        
        
        dir_vec = carla.Vector3D(-right.x, -right.y, 0.0) if side == 'right' else carla.Vector3D(right.x, right.y, 0.0)
        
        ctrl.direction = dir_vec
        walker.apply_control(ctrl)

        
        self.walkers.append(walker)
        self.world.tick()
        return True
        

    def _plan_cross_triggers(self, min_step=30, max_step=2500):
        """
        Plan the step indices within this episode to trigger crossings.
        By default, avoid the first ~80 steps.
        """
        
 #       max_step = 2300
        k = random.choice(self.CROSS_PER_EP_RANGE)   # 2 or 3
        # Ensure we can sample k unique steps
        population = list(range(min_step, max_step))
        if len(population) < k:
            k = max(1, min(2, len(population)))
        self._cross_triggers = sorted(random.sample(population, k))
#        print(f"[CROSS_TRIGGERS] In this episode, pedestrians will spawn at these steps:{self._cross_triggers}")


        
    def _on_collision(self, event):
        self.collision_occurred = True
        kind = 'other'
        other = getattr(event, "other_actor", None)

        if other is not None:
            tid = getattr(other, "type_id", "")  
            if tid.startswith("walker."):
                kind = "pedestrian"
            elif tid.startswith("vehicle."):
                kind = "vehicle"
            else:
                kind = "other"

        self.collision_kind = kind
        if kind == "pedestrian":
            self.collision_penalty = -100.0
        elif kind == "vehicle":
            self.collision_penalty = -100.0
        else:
            self.collision_penalty = -50.0
            
            
    def _on_npc_collision(self, event):
        """When an NPC vehicle collides with a pedestrian, destroy the pedestrian and remove it from the cache."""
        actor = getattr(event, "actor", None)
        other = getattr(event, "other_actor", None)

    
        if not actor or not str(getattr(actor, "type_id", "")).startswith("vehicle."):
            return

    
        if not other or not str(getattr(other, "type_id", "")).startswith("walker."):
            return

        try:
            if other.is_alive:
                if other in self.walkers:
                    self.walkers.remove(other)
                self._seg_mem.pop(("ped", int(other.id)), None)
                other.destroy()
        
        except Exception:
            pass



    def get_collision_data(self):
        return self.collision_occurred

    def get_speed(self):
        # m/s
        vel = self.vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    
    def get_lat_error(self):
        
        if self.vehicle is None:
            return 0.0

        tf = self.vehicle.get_transform()
        # Use the nearest waypoint on the route
        wp, _ = self.find_current_waypoint_with_index(tf.location)
        if wp is None:
            return self.max_lat_error

        center = wp.transform.location
        fwd    = wp.transform.get_forward_vector()
        fx, fy = float(fwd.x), float(fwd.y)

        # Unitized forward vector
        norm = math.hypot(fx, fy)
        if norm < 1e-8:
            return 0.0
        fx /= norm
        fy /= norm

        dx = float(tf.location.x - center.x)
        dy = float(tf.location.y - center.y)

        # Left negative, right positive
        lat_error = dx * (-fy) + dy * (fx)
        return lat_error



    def get_yaw_diff(self):
        
        if self.vehicle is None:
            return 0.0

        tf = self.vehicle.get_transform()
        
        wp, _ = self.find_current_waypoint_with_index(tf.location)
        if wp is None:
            return 0.0
        
        lane_yaw_deg = wp.transform.rotation.yaw
        
        lane_yaw = math.radians(lane_yaw_deg)
        veh_yaw  = math.radians(tf.rotation.yaw)

        yaw_diff = (veh_yaw - lane_yaw + math.pi) % (2 * math.pi) - math.pi
        return yaw_diff



    def is_off_lane(self):
        """
        Determining whether a vehicle has truly departed the lane: The projection point is not within the lane and the lateral deviation exceeds the threshold.
        """
        lat_error = self.get_lat_error()
        valid_wp = self.map.get_waypoint(
            self.vehicle.get_transform().location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving)
        return (valid_wp is None) and (abs(lat_error) > self.max_lat_error)
    
    
    
   

    def generate_route(self, start_loc=None, min_sep=0.0, max_tries=50):
        
        if not hasattr(self, "route_planner") or self.route_planner is None:
            raise RuntimeError("route_planner not initialized")
        if not hasattr(self, "spawn_points") or not self.spawn_points:
            raise RuntimeError("spawn_points is empty")

       
        if start_loc is None:
            if self.vehicle is not None:
                start_loc = self.vehicle.get_transform().location
            else:
                sp_start = random.choice(self.spawn_points)
                start_loc = sp_start.location

        for _ in range(max_tries):
            sp_goal = random.choice(self.spawn_points)
            goal_loc = sp_goal.location

            
            if start_loc.distance(goal_loc) < float(min_sep):
                continue

            try:
                route = self.route_planner.trace_route(start_loc, goal_loc)
                if route and len(route) >= 2:
                    return route
            except Exception:
                pass

        print("[Route] Failed to find a valid route after retries.")
        return None



    def _replan_route_from_here(self):
        """
        Generate a new route from the vehicle's current location to a random destination.
        Rebuild route and signal light indexes, and reset local counters.
        """
      
        tf = self.vehicle.get_transform()
        start_wp = self.map.get_waypoint(
            tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if start_wp is None:
            print("[Route] Replan error: cannot project current location to Driving lane.")
            return False
        new_route = self.generate_route(start_loc=start_wp.transform.location)
        if not new_route:
            print("[Route] Replan failed: generate_route() returned None.")
            return False

        self.route = new_route
        self._seg_mem.clear()
        self._build_tl_index_for_route(onroute_tol=3.0)
        self._build_stop_index_for_route(onroute_tol=3.0)
        self._stopped_idxs.clear()
        self._stop_hold = 0
        self._stop_active = False
        self._stop_violation_latched = False
        self._curr_stop_idx = None

        _, idx = self.find_current_waypoint_with_index(self.vehicle.get_transform().location)
        self.last_waypoint_idx = int(idx)
        self.offroute_count = 0
        self._tl_violation_latched = False
        self._prev_yaw_diff = self.get_yaw_diff()
        
        self.stuck_counter = 0


        return True


    def find_current_waypoint_with_index(self, loc=None):
        if not self.route:
            return None, 0
        if loc is None:
            loc = self.vehicle.get_transform().location
        best_i, best_d, best_wp = 0, float('inf'), None
        for i, (wp, _) in enumerate(self.route):
            d = wp.transform.location.distance(loc)
            if d < best_d:
                best_d = d
                best_i = i
                best_wp = wp
        return best_wp, best_i


   
        
    def find_next_waypoint(self, current_waypoint, distance_lookahead=1, tol=0.1):
        
        if not self.route:
            return None

        loc = current_waypoint.transform.location
        wp_near, idx = self.find_current_waypoint_with_index(loc)
        if wp_near is not None:
            if wp_near.transform.location.distance(loc) <= float(tol):
                j = idx + int(distance_lookahead)
                return self.route[j][0] if 0 <= j < len(self.route) else None

        for i, (wp, _) in enumerate(self.route):
            if wp.transform.location.distance(loc) <= float(tol):
                j = i + int(distance_lookahead)
                return self.route[j][0] if 0 <= j < len(self.route) else None

        return None




    def get_future_heading(self, num_lookahead=4, distance_lookahead=3):
        """
        Future Course Strength ∈ [-1,1]:
          The larger the absolute value → the sharper the curve ahead;
          Left:Negative, Right:Positive.
        """
        tf = self.vehicle.get_transform()
        curr_wp, _ = self.find_current_waypoint_with_index(tf.location)
        if curr_wp is None:
           return 0.0

        def ang_diff(a, b):
            # Return the angular difference (in radians) between b and a, normalized to the range [-π, π].
            d = (b - a + math.pi) % (2 * math.pi) - math.pi
            return d

        yaw_sum_abs = 0.0
        yaw_sum_signed = 0.0

        wp = curr_wp
        steps = 0
        for _ in range(num_lookahead):
            fwp = self.find_next_waypoint(wp, distance_lookahead)
            if not fwp:
                break
            yaw1 = math.radians(wp.transform.rotation.yaw)
            yaw2 = math.radians(fwp.transform.rotation.yaw)
            d = ang_diff(yaw1, yaw2)
            yaw_sum_abs += abs(d)       # Strehgth:Unsigned Cumulative
            yaw_sum_signed += d         # Direction:Signed Accumulation
            wp = fwp
            steps += 1

        if steps == 0:
            return 0.0

        if yaw_sum_signed > math.radians(1.0):
            turn_dir = 1.0
        elif yaw_sum_signed < -math.radians(1.0):
            turn_dir = -1.0
        else:
            turn_dir = 0.0

        
        strength = min(1.0, yaw_sum_abs / math.pi)
        return max(-1.0, min(1.0, strength * turn_dir))

 
    def _corridor_polyline(self, L_f=None, L_b=None):
        """Retrieve waypoints from last_waypoint_idx, moving L_f meters forward and L_b meters backward; do not interpolate."""
        if not self.route:
            return None, None
        if L_f is None: L_f = self.COR_AHEAD_M
        if L_b is None: L_b = self.COR_BACK_M

        i0 = max(0, int(self.last_waypoint_idx))
        loc0 = self.route[i0][0].transform.location
        x0, y0 = float(loc0.x), float(loc0.y)

        # Retrospective collection
        pts_back = [(x0, y0)]
        total_b = 0.0
        j = i0
        while j - 1 >= 0 and total_b < L_b:
            loc_prev = self.route[j-1][0].transform.location
            x, y = float(loc_prev.x), float(loc_prev.y)
            d = math.hypot(x - pts_back[-1][0], y - pts_back[-1][1])
            if d > 1e-6:
                pts_back.append((x, y))
                total_b += d
            j -= 1
        pts_back.reverse()  # The latter part should be placed at the beginning.

        # Collect Forward
        pts = pts_back[:]   # The entire corridor centered around the vicinity of one's own vehicle
        total_f = 0.0
        i = i0 + 1
        while i < len(self.route) and total_f < L_f:
            loc = self.route[i][0].transform.location
            x, y = float(loc.x), float(loc.y)
            d = math.hypot(x - pts[-1][0], y - pts[-1][1])
            if d > 1e-6:
                pts.append((x, y))
                total_f += d
            i += 1

        if len(pts) < 2:
            return None, None
 
        # Cumulative Arc Length S
        S = [0.0]
        acc = 0.0
        for k in range(1, len(pts)):
            dk = math.hypot(pts[k][0] - pts[k-1][0], pts[k][1] - pts[k-1][1])
            acc += dk
            S.append(acc)
        return pts, S

    def _proj_on_segment(self, x, y, pts, S, i):
        """Perform a single projection on the i-th segment and return the details."""
        if i < 0 or i >= len(pts)-1:
            return None
        x0, y0 = pts[i]
        x1, y1 = pts[i+1]
        vx, vy = (x1 - x0), (y1 - y0)
        seg2 = vx*vx + vy*vy
        if seg2 < 1e-12:
            return None

        t_raw = ((x - x0)*vx + (y - y0)*vy) / seg2
        # Within a paragraph
        t = 0.0 if t_raw < 0.0 else (1.0 if t_raw > 1.0 else t_raw)

        px, py = x0 + t*vx, y0 + t*vy
        seg = math.sqrt(seg2)
        tx, ty = (vx/seg, vy/seg)
        nx, ny = (-ty, tx)

        dx, dy = (x - px), (y - py)
        d = dx*nx + dy*ny
        s = S[i] + t*seg
        dist2 = dx*dx + dy*dy
        inside = 0 if (0.0 <= t_raw <= 1.0) else 1  
        return {"s": s, "d": d, "tx": tx, "ty": ty, "idx": i,
                "t_raw": t_raw, "dist2": dist2, "inside": inside}


    def _project_to_poly_euclid_greedy(self, x, y, pts, S):
       
        best = None
        for i in range(len(pts) - 1):
            r = self._proj_on_segment(x, y, pts, S, i)
            if r is None:
                continue
            if best is None:
                best = r
                continue
            
            if (r["inside"], r["dist2"]) < (best["inside"], best["dist2"]):
                best = r
            elif (r["inside"] == best["inside"]) and (abs(r["dist2"] - best["dist2"]) < 1e-10):
                
                if abs(r["d"]) < abs(best["d"]):
                    best = r
        return best 
        
        
    def _project_to_poly_stable(self, x, y, pts, S, mem_key=None):
        
        cand = self._project_to_poly_euclid_greedy(x, y, pts, S)
        if cand is None:
            return None, None, 0.0, 0.0

        
        if not mem_key or mem_key not in self._seg_mem:
            if mem_key:
                self._seg_mem[mem_key] = {"idx": cand["idx"], "dist2": cand["dist2"]}
            return cand["s"], cand["d"], cand["tx"], cand["ty"]

        
        old = self._seg_mem[mem_key]
        reproj_old = self._proj_on_segment(x, y, pts, S, old["idx"])
        if reproj_old is None:
            
            self._seg_mem[mem_key] = {"idx": cand["idx"], "dist2": cand["dist2"]}
            return cand["s"], cand["d"], cand["tx"], cand["ty"]

        
        dist_new = math.sqrt(cand["dist2"])
        dist_old = math.sqrt(reproj_old["dist2"])
        H = float(self.SEG_HYST)

        
        prefer_inside = (cand["inside"] < reproj_old["inside"])

        if (dist_new + H < dist_old) or (prefer_inside and dist_new + 0.5*H < dist_old):
            chosen = cand
        else:
            chosen = reproj_old

        
        self._seg_mem[mem_key] = {"idx": chosen["idx"], "dist2": chosen["dist2"]}
        return chosen["s"], chosen["d"], chosen["tx"], chosen["ty"]

 
    def _scan_band(self, band_min, band_max, pts, S, margin=0.5):
        
        """
        Use the Frenet coordinates of the public corridor polyline [pts, S] to measure the distance between the vehicle and NPCs:
          - Perform _project_to_poly on both ego and NPC to obtain (s, d, tx, ty)
          - Front/Rear: s_rel = s_o - s_e (>0 Front, <0 Rear)
          - Corridor judgment: use d_o and [band_min, band_max]
          - TTC: use speed of two vehicles
        """

        GAP_CLIP, TTC_CLIP = self.GAP_CLIP, self.TTC_CLIP

        # Ego Vehicle
        ego_tf = self.vehicle.get_transform()
        ex, ey = float(ego_tf.location.x), float(ego_tf.location.y)
        s_e, d_e, tx_e, ty_e = self._project_to_poly_stable(ex, ey, pts, S, mem_key=("ego",))
        # If the corridor projection fails, it is directly deemed obstacle-free.
        if s_e is None:
            return GAP_CLIP, TTC_CLIP, GAP_CLIP, TTC_CLIP, None, None, None

        ev = self.vehicle.get_velocity()
        
        # Vehicle half-length
        def _half_len(actor):
            try:
                return float(actor.bounding_box.extent.x)
            except Exception:
                return 2.0
        E = _half_len(self.vehicle)

        bestF = None   # (s_rel, s_eff, rel_v)
        bestR = None

        # Scan surrounding vehicles
        ego_loc = ego_tf.location
        for v in self.world.get_actors().filter('vehicle.*'):
            if v.id == self.vehicle.id:
                continue
            loc = v.get_location()
            if loc.distance(ego_loc) > float(self.SENSE_RADIUS):
                continue

            
            s_o, d_o, tx_o, ty_o = self._project_to_poly_stable(
                float(loc.x), float(loc.y), pts, S, mem_key=("veh", int(v.id))
            )
            if s_o is None:
                continue
                
                
            path_len = S[-1]
            
            
            if s_o > (path_len - 0.5):
                # Calculate the true physical straight-line distance.
                phys_dist = loc.distance(ego_loc)                
                s_rel = phys_dist
            else:
                s_rel = s_o - s_e                
                
                
                

#            s_rel = s_o - s_e  #  >0 in front, <0 behind (along the corridor)
            
            if s_rel < -self.COR_BACK_M or s_rel > self.COR_AHEAD_M:
                continue
            # Band filtering: Using NPC's d_o (negative on the left, positive on the right)
            if d_o < band_min or d_o > band_max:
                continue
#######################################################################################################
 #           center_dist = loc.distance(ego_loc)  # Euclidean distance between the centers of two vehicles
            
#########################################################################################################
            
            
 
            vel = v.get_velocity()
            v_other_spd = math.sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z)
            v_ego_spd = math.sqrt(ev.x*ev.x + ev.y*ev.y + ev.z*ev.z)

            
            
            

            T = _half_len(v)
            #####################################################################################################
#            print(f"[DBG] dist={center_dist:.2f}m  s_rel={s_rel:.2f}m  |d|={abs(d_o):.2f}m  E={E:.2f}m  T={T:.2f}m")
#            print(f"[DBG] s_rel={s_rel:.2f}m  |d|={abs(d_o):.2f}m  E={E:.2f}m  T={T:.2f}m")
#            print(f"[DBG2] s_e={s_e:.2f}  s_o={s_o:.2f}")
            
#            dx = float(loc.x - ego_tf.location.x)
#            dy = float(loc.y - ego_tf.location.y)
#            s_rel_geo = dx*tx_e + dy*ty_e   
#            print(f"s_rel_geo={s_rel_geo:.2f}")
            ###########################################################################################################
          

            if s_rel >= 0.0:           # Front Vehicle
                s_eff = max(0.0, s_rel - (E + T) - margin)     # Bumper to bumper
                rel = v_ego_spd - v_other_spd            # >0 We are closing in.
                if (bestF is None) or (s_rel < bestF[0]):
                    #bestF = (s_rel, s_eff, rel)
                    bestF = (s_rel, s_eff, rel, int(v.id), float(s_o), float(s_e))
            else:                      # Behind Vehicle
                s_abs = -s_rel
                s_eff = max(0.0, s_abs - (E + T) - margin)
                rel = v_other_spd - v_ego_spd           # >0 The vehicle behind is closing in.
                if (bestR is None) or (s_abs < abs(bestR[0])):
                    bestR = (s_rel, s_eff, rel)

  
        if bestF is None:
            gapF, ttcF, relF = GAP_CLIP, TTC_CLIP, None
            front_id = None
            s_rel_raw = None
        else:
            s_eff, rel = bestF[1], bestF[2]
            gapF = min(GAP_CLIP, s_eff)
            ttcF = TTC_CLIP if rel <= self._EPS else min(TTC_CLIP, s_eff / rel)
            relF = rel  
            front_id = bestF[3]
            s_rel_raw = bestF[0]  # s_o - s_e,No bumper-to-bumper coverage

        if bestR is None:
            gapR, ttcR = GAP_CLIP, TTC_CLIP
        else:
            s_eff, rel = bestR[1], bestR[2]
            gapR = min(GAP_CLIP, s_eff)
            ttcR = TTC_CLIP if rel <= self._EPS else min(TTC_CLIP, s_eff / rel)

        #return float(gapF), float(ttcF), float(gapR), float(ttcR), (None if relF is None else float(relF))
        return float(gapF), float(ttcF), float(gapR), float(ttcR), (None if relF is None else float(relF)), front_id, (None if s_rel_raw is None else float(s_rel_raw))
     

    def _scan_surroundings(self):
        
        pts, S = self._corridor_polyline(L_f=self.COR_AHEAD_M, L_b=self.COR_BACK_M)
        if pts is None:
            return {
                "ego_gapF": float(self.GAP_CLIP),
                "ego_ttcF": float(self.TTC_CLIP),
                "left_ok": 0.0,
                "right_ok": 0.0,
                "l_gapF": 0.0,
                "l_ttcF": 0.0,
                "r_gapF": 0.0,
                "r_ttcF": 0.0,
                "ego_lane": 0,          # -1/0/+1
                "ego_closing": 0.0,
                "d_e": 0.0,
                "front_id": None,
                "front_srel_raw": 0.0,
                "ped_active": 0.0,         
                "ped_gap": float(self.GAP_CLIP), 
            }

        
        lane_w = 3.5
        ego_wp = self.map.get_waypoint(self.vehicle.get_transform().location,
                                       project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp:
            try:
                lane_w = max(2.5, float(ego_wp.lane_width) or 3.5)
            except Exception:
                pass
        w_c = 0.5 * lane_w

        # ego vehicle (s_e, d_e)
        ego_tf = self.vehicle.get_transform()
        ex, ey = float(ego_tf.location.x), float(ego_tf.location.y)
        s_e, d_e, _, _ = self._project_to_poly_stable(ex, ey, pts, S, mem_key=("ego",))
        if s_e is None:
            return {
                "ego_gapF": float(self.GAP_CLIP),
                "ego_ttcF": float(self.TTC_CLIP),
                "left_ok": 0.0,
                "right_ok": 0.0,
                "l_gapF": 0.0,
                "l_ttcF": 0.0,
                "r_gapF": 0.0,
                "r_ttcF": 0.0,
                "ego_lane": 0,
                "ego_closing": 0.0,
                "d_e": 0.0,
                "front_id": None,
                "front_srel_raw": 0.0,
                "ped_active": 0.0,         
                "ped_gap": float(self.GAP_CLIP), 
            }

        
        if d_e < -0.5 * lane_w:
            ego_lane = -1
        elif d_e > 0.5 * lane_w:
            ego_lane = +1
        else:
            ego_lane = 0

       
        d0 = float(ego_lane) * lane_w

        # three band
        band_curr = (d0 - (w_c+0.4),           d0 + (w_c+0.4))
        band_left = (d0 - 1.5*lane_w,    d0 - 0.5*lane_w -0.4)
        band_right= (d0 + 0.5*lane_w +0.4,    d0 + 1.5*lane_w)

        
        if ego_lane == 0:
            left_drv  = self._adjacent_lane_is_driving('left',  lookahead_m=min(self.COR_AHEAD_M, 10.0))
            right_drv = self._adjacent_lane_is_driving('right', lookahead_m=min(self.COR_AHEAD_M, 10.0))
        elif ego_lane == -1:
            left_drv  = False  
            right_drv = self._adjacent_lane_is_driving('right', lookahead_m=min(self.COR_AHEAD_M, 10.0))
        else:  # ego_lane == +1
            left_drv  = self._adjacent_lane_is_driving('left',  lookahead_m=min(self.COR_AHEAD_M, 10.0))
            right_drv = False  

       
        #ego_gapF, ego_ttcF, _, _, ego_relF = self._scan_band(band_curr[0], band_curr[1], pts, S)
        ego_gapF, ego_ttcF, _, _, ego_relF, ego_front_id, ego_srel_raw = self._scan_band(band_curr[0], band_curr[1], pts, S)
        # Center-to-center -> Bumper-to-bumper clearance
        def _half_len(actor):
            try:
                return float(actor.bounding_box.extent.x)
            except Exception:
                return 2.0

        E = _half_len(self.vehicle)
        T = 0.0
        if ego_front_id is not None:
            tgt = self.world.get_actor(ego_front_id)
            if tgt is not None:
                T = _half_len(tgt)

        s_eff_front = None if ego_srel_raw is None else (ego_srel_raw - (E + T))

        if left_drv:
            l_gapF, l_ttcF, l_gapR, l_ttcR, _, _, _ = self._scan_band(band_left[0], band_left[1], pts, S)
        else:
            l_gapF = l_ttcF = l_gapR = l_ttcR = 0.0
 
        if right_drv:
            r_gapF, r_ttcF, r_gapR, r_ttcR, _, _, _ = self._scan_band(band_right[0], band_right[1], pts, S)
        else:
            r_gapF = r_ttcF = r_gapR = r_ttcR = 0.0

        
        left_ok  = 1.0 if (left_drv  and l_gapF >= 20.0 and l_gapR >= 2.0 and l_ttcF >= 2.5 and l_ttcR >= 2.5) else 0.0
        right_ok = 1.0 if (right_drv and r_gapF >= 20.0 and r_gapR >= 2.0 and r_ttcF >= 2.5 and r_ttcR >= 2.5) else 0.0
        ego_closing = 1.0 if (ego_relF is not None and ego_relF > self._EPS) else 0.0
        
        
        
        # ===== pedestrians (front only, corridor-based) =====
        best_ped = None  # (s_rel, ped_gap)
       
        T_PED = 0.35
        margin = 0.5
       

        ego_loc = ego_tf.location










##################################################################################################################################################
        
        
     
        for w in self.world.get_actors().filter('walker.pedestrian.*'):
            loc = w.get_location()
            
            if loc.distance(ego_loc) > float(self.SENSE_RADIUS):
                continue

        
            s_o, d_o, tx, ty = self._project_to_poly_stable(
                float(loc.x), float(loc.y), pts, S, mem_key=("ped", int(w.id))
            )
            if s_o is None:
                continue
            
            s_rel = s_o - s_e

            
            if not (-10.0 < s_rel <= self.COR_AHEAD_M):
                continue

            
            vel = w.get_velocity()
            vxy = (float(vel.x), float(vel.y))
            
            # Normal vector (nx, ny) points right.
            nx, ny = -ty, tx
            
            # Lateral velocity (>0 moving to the right, <0 moving to the left).
            v_lat = vxy[0] * nx + vxy[1] * ny
            # v_lon: Longitudinal velocity (used to determine whether it is moving parallel to the road).
            v_lon = vxy[0] * tx + vxy[1] * ty
 #           print(f"Walker ID {w.id}: d_o={d_o:.2f}, v_lat={v_lat:.2f}, v_lon={v_lon:.2f}")
            # ==============================================================
            # Cleanup
            # ==============================================================
            should_destroy = False

            # A. Parallel-interference removal: If the longitudinal velocity component is much greater than the lateral component, it indicates that the agent is moving along the road (forward or backward):
            
            if abs(v_lon) > 0.5 and abs(v_lat) < 0.5:
                should_destroy = True

            # B. Crossing-completed removal:
            # Case 1: Crossing from left to right
            if v_lat > 0.1 and d_o > 2.5:
                should_destroy = True
            
            # Case 2: Crossing from right to left
            elif v_lat < -0.1 and d_o < -2.5:
                should_destroy = True
            
            # C. Removal due to abnormal distance
            if abs(d_o) > 6.0:
                should_destroy = True

          
            if should_destroy:

                found_in_list = False
                for i, stored_w in enumerate(self.walkers):
                    if stored_w.id == w.id:
                        self.walkers.pop(i) 
                        found_in_list = True
                        break
                
                
                if found_in_list:
                    w.destroy()
#                    print(f"[Walker Destroy] ID={w.id} killed (d={d_o:.2f}, v={v_lat:.2f})") 
             
                self._seg_mem.pop(("ped", int(w.id)), None)                                
                continue

           
            
            if abs(d_o) > 6.0:
                continue

            
            if s_rel <= 0.0:
                continue

            speed_p = math.hypot(vxy[0], vxy[1])
            if speed_p < 0.05:
                continue

            
            affect = False
            if v_lat > 0:            # move to the right
                affect = (d_o <= w_c) # 
            elif v_lat < 0:          # move to the left
                affect = (d_o >= -w_c) # 
            
            if not affect:
                continue

            # calculate Gap
            s_eff = max(0.0, s_rel - (E + T_PED) - margin)
            ped_gap = min(self.GAP_CLIP, s_eff)

            if (best_ped is None) or (s_rel < best_ped[0]):
                best_ped = (s_rel, ped_gap)
        
###################################################################################################################################################
#            print(f"[PED_DBG]d_o={d_o:.2f}  s_rel={s_o - s_e:.2f}  walker_id={w.id}")

#####################################################################################################################################################
        
        if best_ped is None:
            ped_active = 0.0
            ped_gap = float(self.GAP_CLIP)
        else:
            ped_active = 1.0
            ped_gap = float(best_ped[1])
################################################################################################################################################
#        print(f"[PED_SUMMARY] ped_active={ped_active:.0f}  ped_gap={ped_gap:.2f}[DEBUG]")

################################################################################################################################################        
 

        return {
            "ego_gapF": float(ego_gapF), "ego_ttcF": float(ego_ttcF),
            "left_ok": float(left_ok), "right_ok": float(right_ok),
            "l_gapF": float(l_gapF) if left_drv else 0.0,
            "l_ttcF": float(l_ttcF) if left_drv else 0.0,   # 
            "r_gapF": float(r_gapF) if right_drv else 0.0,
            "r_ttcF": float(r_ttcF) if right_drv else 0.0,  # 
            "ego_lane": int(ego_lane),
            "ego_closing": float(ego_closing), 
            "d_e": float(d_e),                                # 
            "front_id": (None if ego_front_id is None else int(ego_front_id)),
            "front_srel_raw": (0.0 if ego_srel_raw is None else float(ego_srel_raw)),
            "front_s_eff": (0.0 if s_eff_front is None else float(s_eff_front)),
            "ped_active": float(ped_active),
            "ped_gap": float(ped_gap),
        }


 
    def _build_tl_index_for_route(self, onroute_tol=3.0):
        self._tl_infos = []
        tls = self.world.get_actors().filter('traffic.traffic_light')
        visited = set()

        for tl in tls:
            if tl.id in visited:
                continue

            wps = tl.get_stop_waypoints()
            if not wps:
                continue

            best_wp, best_i, best_d = None, None, float('inf')
            for swp in wps:
                for i, (rwp, _) in enumerate(self.route):
                    d = rwp.transform.location.distance(swp.transform.location)
                    if d < best_d:
                        best_d, best_i, best_wp = d, i, swp
            if best_i is None or best_d > onroute_tol:
                continue

            grp = tl.get_group_traffic_lights() or [tl]
            for t in grp:
                visited.add(t.id)


            self._tl_infos.append((tl, best_wp, best_i))

#        self.world.tick()
######################################################################################################################################
#        print("[TL Check] Route-related traffic lights:")
        if self._tl_infos:
            for i, (tl, _, _) in enumerate(self._tl_infos):
                grp = tl.get_group_traffic_lights() or [tl]
                for j, t in enumerate(grp):
                    st = t.get_state()
#                    print(f"[TL Check] RouteTL{i}-Light{j}: "
#                          f"red={t.get_red_time():.1f}, yellow={t.get_yellow_time():.1f}, green={t.get_green_time():.1f}, "
#                          f"elapsed={t.get_elapsed_time():.2f}, state={st.name}")
        else:
#            print("[TL Check] No traffic lights found on route.")
            pass

###################################################################################################################################


      
        
    def _red_light_info(self, lookahead_points=None):
        """
        Return (affecting, sdist_route, state, stop_wp)
        - affecting: a red/yellow light ahead that is relevant to this route
        - sdist_route: Distance from vehicle to parking line along route (meters)
        - state: carla.TrafficLightState
        - stop_wp: Stopline waypoint
        """
        if lookahead_points is None:
            lookahead_points = self.TL_LOOKAHEAD
        if not self.route or not self._tl_infos:
            return False, None, None, None

        
        ego_loc = self.vehicle.get_transform().location
        curr_i, best_d = 0, float('inf')
        for i, (rwp, _) in enumerate(self.route):
            d = rwp.transform.location.distance(ego_loc)
            if d < best_d:
                best_d, curr_i = d, i

        # Find the nearest relevant light within the future lookahead_points
        end_i = min(len(self.route) - 1, curr_i + lookahead_points)
        best_tl = None
        best_idx = None
        best_wp = None
        for (tl, stop_wp, tl_idx) in self._tl_infos:
            if curr_i <= tl_idx <= end_i:
                if best_idx is None or tl_idx < best_idx:
                    best_tl, best_idx, best_wp = tl, tl_idx, stop_wp
##################################################################################
        
 #       self._affecting_tl = best_tl
        

        
        
        
        
        if best_tl is not None and (self.step_count % 50 == 0):
            try:
                grp = best_tl.get_group_traffic_lights() or [best_tl]
#                ctrl = grp[0]
                st   = best_tl.get_state()

#                print(
#                      f"[LIVE TL] tl_id={best_tl.id} state={getattr(st,'name',st)} "
#                      f"elapsed={best_tl.get_elapsed_time():.2f} "
#                      f"R/Y/G(best)={best_tl.get_red_time():.1f}/{best_tl.get_yellow_time():.1f}/{best_tl.get_green_time():.1f} "
#                      f"num_lights_in_group={len(grp)}"
#                )
            except Exception:
                pass

####################################################
        if best_tl is None:
 #           return False, None, None, None
            
 #           self._affecting_tl = None
            
            return False, None, None, None


        state = best_tl.get_state()
        if state not in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
            
            return False, None, state, best_wp

   
        def route_arc_len(i_from, i_to):
            if i_to <= i_from:
                return 0.0
            arc_len = 0.0
            for k in range(i_from, i_to):
                a = self.route[k][0].transform.location
                b = self.route[k+1][0].transform.location
                arc_len += a.distance(b)
            return arc_len

        sdist_pos = route_arc_len(curr_i, best_idx)
        if sdist_pos > self.RL_SENSE_DIST:
    
            
            return False, None, best_tl.get_state(), best_wp

       
        veh2stop = best_wp.transform.location.distance(ego_loc)

        f = best_wp.transform.get_forward_vector()
        fx, fy = float(f.x), float(f.y)
        norm = math.hypot(fx, fy) or 1.0
        fx /= norm; fy /= norm

       
        relx = float(ego_loc.x - best_wp.transform.location.x)
        rely = float(ego_loc.y - best_wp.transform.location.y)
        ahead_signed = relx * fx + rely * fy  # m

        # Crossing condition
        crossed = (best_idx < curr_i) or (ahead_signed > 0.2 and self.get_speed() > 0.1)


        sdist = -veh2stop if crossed else sdist_pos
        return True, sdist, state, best_wp

    def _build_stop_index_for_route(self, onroute_tol=3.0):
        """Build STOP stop-line indices for the current route:[(stop_wp, stop_idx)]"""
        self._stop_infos = []
        cand_wps = []

        # Approach A: First attempt to use actors of the 'stop' type in the scene.
        try:
            for a in self.world.get_actors().filter('traffic.stop*'):
                wp = self.map.get_waypoint(
                    a.get_transform().location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
                if wp: cand_wps.append(wp)
        except Exception:
            pass

        # Approach B: Use map landmarks to gather STOP points
        try:
            for lm in getattr(self.map, "get_all_landmarks", lambda: [])():
                t = str(getattr(lm, "type", "")).lower()
                if "stop" in t:  
                    wp = None
                    try:
                        wp = self.map.get_waypoint_xodr(lm.road_id, lm.lane_id, lm.s)
                    except Exception:
                        pass
                    if wp and wp.lane_type == carla.LaneType.Driving:
                        cand_wps.append(wp)
        except Exception:
            pass

       
        used = set()
        for swp in cand_wps:
            best_idx, best_d = None, 1e9
            sloc = swp.transform.location
            for i, (rwp, _) in enumerate(self.route):
                d = rwp.transform.location.distance(sloc)
                if d < best_d:
                    best_d, best_idx = d, i
            if best_idx is not None and best_d <= onroute_tol and best_idx not in used:
                used.add(best_idx)
                self._stop_infos.append((swp, best_idx))


    def _stop_info(self, lookahead_points=None):
        """
        return (affecting, sdist_stop, stop_wp, stop_idx)
        """
        if lookahead_points is None:
            lookahead_points = self.STOP_LOOKAHEAD
        if not self.route or not self._stop_infos:
            return False, None, None, None

        
        ego_loc = self.vehicle.get_transform().location
        curr_i, best_d = 0, float('inf')
        for i, (rwp, _) in enumerate(self.route):
            d = rwp.transform.location.distance(ego_loc)
            if d < best_d:
                best_d, curr_i = d, i

        end_i = min(len(self.route) - 1, curr_i + lookahead_points)

        # Select the nearest unprocessed STOP ahead
        best_idx, best_wp = None, None
        for wp, idx in self._stop_infos:
            if curr_i <= idx <= end_i and idx not in self._stopped_idxs:
                if best_idx is None or idx < best_idx:
                    best_idx, best_wp = idx, wp
        if best_idx is None:
            return False, None, None, None

        # Distance along the route
        def route_arc_len(i_from, i_to):
            if i_to <= i_from: return 0.0
            s = 0.0
            for k in range(i_from, i_to):
                a = self.route[k][0].transform.location
                b = self.route[k+1][0].transform.location
                s += a.distance(b)
            return s

        sdist = route_arc_len(curr_i, best_idx)
        
        if curr_i > best_idx:
            sdist = -self.route[best_idx][0].transform.location.distance(ego_loc)
            
        f = best_wp.transform.get_forward_vector()
        rel = ego_loc - best_wp.transform.location
        ahead_signed = rel.x*f.x + rel.y*f.y + rel.z*f.z
        if (ahead_signed > 0.2) and (self.get_speed() > 0.1):
            sdist = -best_wp.transform.location.distance(ego_loc)
        

        return True, sdist, best_wp, best_idx


  

        
        
    def desired_speed_from_curve(self, future_h):
        future_h_abs = abs(float(future_h))
        max_v = self.max_speed

       
        try:
            limit_kph = self.vehicle.get_speed_limit()
            if limit_kph is not None and limit_kph > 0:
                max_v = float(limit_kph) / 3.6
        except Exception:
            pass

       
        self.max_speed = max_v

   
        if future_h_abs > 0.25:
            v_des = 0.8 * max_v
        else:
            v_des = max_v

        return v_des
            
   
        

    def _adjacent_lane_is_driving(self, side: str, lookahead_m: float = 10.0) -> bool:
       
        ego_wp = self.map.get_waypoint(
            self.vehicle.get_transform().location,
            project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if not ego_wp:
            return False

        def adj(wp):
            return wp.get_left_lane() if side == 'left' else wp.get_right_lane()

        
        a = adj(ego_wp)
        if (a is None) or (a.lane_type != carla.LaneType.Driving):
            return False

       
        if not self.route:
            return False
            
        if a.lane_id * ego_wp.lane_id <= 0:   
            return False

        dist = 0.0
        i = max(0, int(self.last_waypoint_idx))
        prev = self.route[i][0].transform.location

        while (i + 1) < len(self.route) and dist < lookahead_m:
            i += 1
            wp_i = self.route[i][0]
            loc  = wp_i.transform.location
            dist += prev.distance(loc)
            prev = loc

            a = adj(wp_i)
            if (a is None) or (a.lane_type != carla.LaneType.Driving):
                return False

        return True


    

    def _remaining_arc_length(self, start_idx=None):
      
        if not self.route:
            return 0.0
        if start_idx is None:
            start_idx = self.last_waypoint_idx
        rem = 0.0
        for k in range(start_idx, len(self.route) - 1):
            a = self.route[k][0].transform.location
            b = self.route[k+1][0].transform.location
            rem += a.distance(b)
        return rem


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy_npcs()
        self._destroy_walkers()
        self.step_count = 0
        self.collision_occurred = False
        self.collision_kind = None
        self.collision_penalty = 0.0
        self.last_steer = 0.0
        
        self.offroute_count = 0  
        self.stuck_counter = 0
        
  

        sp_start = random.choice(self.spawn_points)
        start_wp = self.map.get_waypoint(
            sp_start.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        
        veh_bp = self.bp_lib.find('vehicle.tesla.model3')
        veh_bp.set_attribute('role_name', 'hero')
        spawn_wp = start_wp
        spawn_tf = carla.Transform(spawn_wp.transform.location, spawn_wp.transform.rotation)
        spawn_tf.location.z += 0.3
##################################################################################################################################################
#        spectator = self.world.get_spectator()
#        spectator.set_transform(carla.Transform(spawn_tf.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        
       
#        for _ in range(20):
#            self.world.tick()
#################################################################################################################################################
        new_vehicle = None
        for _ in range(20):
            new_vehicle = self.world.try_spawn_actor(veh_bp, spawn_tf)
            if new_vehicle is not None:
                break
            next_wps = spawn_wp.next(2.0)
            if next_wps:
                spawn_wp = random.choice(next_wps)
            else:
                sp_start = random.choice(self.spawn_points)
                spawn_wp = self.map.get_waypoint(
                    sp_start.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
            spawn_tf = carla.Transform(spawn_wp.transform.location, spawn_wp.transform.rotation)
            spawn_tf.location.z += 0.3

        if new_vehicle is None:
            raise RuntimeError("Failed to spawn vehicle after retries.")

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except:
                pass
        self.vehicle = new_vehicle
        self.world.tick()


        # Reconstructing the Collision Sensor
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except Exception:
                pass
            self.collision_sensor = None
        bp = self.bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._on_collision)

        
        self.route = self.generate_route(start_loc=start_wp.transform.location)
        
        
        
        if self.route is None or len(self.route) == 0:
            raise RuntimeError("Route generation failed on reset.")
##############################################################################################
        
        self._seg_mem.clear()

        self._build_tl_index_for_route(onroute_tol=3.0)
        
        



       
        align_idx = min(3, len(self.route) - 1)  
        align_wp = self.route[align_idx][0]
        align_tf = carla.Transform(align_wp.transform.location, align_wp.transform.rotation)
        align_tf.location.z += 0.3
        self.vehicle.set_transform(align_tf)
        self.world.tick()
 
        
        
        # Generate NPC Vehicles
        self._spawn_npcs(self.NUM_NPC)
        

        _, i0 = self.find_current_waypoint_with_index(self.vehicle.get_transform().location)
        self.last_waypoint_idx = int(i0)

     
        s = self.world.get_settings()
        s.no_rendering_mode = (not self.render)
        s.synchronous_mode = SYNCHRONOUS_MODE
        s.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(s)
        self.world.tick()

        self._prev_steer = 0.0
     
        self._prev_yaw_diff = self.get_yaw_diff()
        
        
        
        self._tl_violation_latched = False
     
        
        self.last_long_cmd = 0.0
        # Plan 2~3 crossing triggers for this episode (visible in rendering)
#        self._plan_cross_triggers(min_step=80, max_step=min(self.max_episode_steps - 1, 1000))
        self._plan_cross_triggers(min_step=30, max_step=2500)
        self._build_stop_index_for_route(onroute_tol=3.0)
        self._stopped_idxs.clear()
        self._stop_hold = 0
        self._stop_active = False
        self._stop_violation_latched = False
        self._curr_stop_idx = None


        return self._get_full_obs(), {}



    def compute_reward(self, action=None, scan=None):
        
        if scan is None:
            scan = self._scan_surroundings()
        lat_error = max(-self.max_lat_error, min(self.max_lat_error, self.get_lat_error()))
        speed_now = self.get_speed()
        yaw_diff = self.get_yaw_diff()
        future_h = self.get_future_heading()
        v_des = self.desired_speed_from_curve(future_h)

        # lane keeping
        r_lane = 1.0 * math.exp(-3.5 * abs(lat_error)) + 0.2 * (1/(1+3*abs(yaw_diff)))
#        print(f"re={0.8 * math.exp(-6 * abs(lat_error))}     ")
        base   = 0.7 * (1.0 - min(1.0, abs(speed_now - v_des) / max(1.0, self.max_speed)))
        over   = -0.7 * max(0.0, speed_now - self.max_speed) / max(1.0, self.max_speed)
        r_speed_track = base + over

        dsteer  = abs(self.last_steer - getattr(self, "_prev_steer", self.last_steer))
        prev_yaw_diff = getattr(self, "_prev_yaw_diff", yaw_diff)
        yaw_rate = abs(yaw_diff - prev_yaw_diff)
        r_smooth = (-0.65 * dsteer - 0.50 * yaw_rate)

        self._prev_steer    = self.last_steer
        self._prev_yaw_diff = yaw_diff
        
        
        

        # traffic light
        r_tl = 0.0
        r_pe = 0.0
        _, sdist_rl, tl_state_rl, _ = self._red_light_info(lookahead_points=self.TL_LOOKAHEAD)
        is_ry = (tl_state_rl in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow))
        tl_close = is_ry and (sdist_rl is not None) and (sdist_rl >= 0.0) and (sdist_rl <= self.RL_STOP_START)
        if tl_close:
            D_START = self.RL_STOP_START
            V_STOP  = 0.5
            if sdist_rl >= 0.0:
                near = 1.0 - min(sdist_rl, D_START) / D_START
                r_tl += - 1.8 * (near) * (speed_now / max(1.0, self.max_speed))
                if sdist_rl < 2.0 and speed_now < V_STOP:
                    r_tl += 1.5 * (1.0 - speed_now / V_STOP)
                    
        # ===== pedestrian stop shaping =====
        ped_active_flag = (float(scan.get("ped_active", 0.0)) >= 0.5)
        ped_gap = float(scan.get("ped_gap", self.GAP_CLIP))
        if ped_active_flag:
            D_START = self.RL_STOP_START
            near = 1.0 - min(ped_gap, D_START) / D_START
            r_pe += -1.8 * (near) * (speed_now / max(1.0, self.max_speed))
            if ped_gap < 2.0 and speed_now < 0.5:
                r_pe += 1.5 * (1.0 - speed_now / 0.5)
                
        # ===== stop sign shaping =====
        r_stop = 0.0
        aff_s, sdist_s, _, stop_idx = self._stop_info(lookahead_points=self.STOP_LOOKAHEAD)
        if tl_state_rl is not None: #if traffic light,dont consider stopline
            aff_s, sdist_s = False, None
        stop_close = (aff_s and (sdist_s is not None) and (0.0 <= sdist_s <= self.RL_STOP_START))
        if stop_close:
            D_START = self.RL_STOP_START
            near = 1.0 - min(sdist_s, D_START) / D_START
            r_stop += -1.8 * (near) * (speed_now / max(1.0, self.max_speed))
            if sdist_s <= 2.0 and speed_now < 0.5:   
                r_stop += 1.5 * (1.0 - speed_now / 0.5)

       



            
#        free_to_go = (not tl_close) and (scan["ego_gapF"] > 15.0) and (ped_gap > 15.0) and (not stop_close)
#        if free_to_go:
#   
#            k_vgap = 0.5  
#            r_vgap = k_vgap * min((speed_now - v_des),0) / max(1.0, v_des)
            
#        else:
#            r_vgap = 0.0

        reward = (r_lane + r_speed_track + r_smooth + r_tl + r_pe + r_stop)#+ r_vgap)
        return float(reward)

   
    

    def _apply_smooth_control(self, steer_cmd, long_cmd):
        
        steer = float(max(-1.0, min(1.0, steer_cmd)))
        long  = float(max(-1.0, min(1.0, long_cmd)))
        return steer, long
        


    def step(self, action):
        steer_cmd, long_cmd = float(action[0]), float(action[1])
        
        steer, u = self._apply_smooth_control(steer_cmd, long_cmd)
        
        
        if u >= 0.0:
            throttle = u          # [0,1]
            brake    = 0.0
        else:
            throttle = 0.0
            brake    = -u         # [-1,0] -> [0,1]

        

       
        self.last_steer = steer
        
        prev_idx = self.last_waypoint_idx
        self.last_long_cmd = u 
      
#        lat_error = self.get_lat_error()
        


        
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=False))

        if self.render:
            time.sleep(FIXED_DELTA_SECONDS)
        self.world.tick()
        
        # === Trigger a planned crossing if this step is scheduled ===
        if self._cross_triggers and (self.step_count in self._cross_triggers):
            # randomize one crossing with your specified ranges
            d = random.uniform(self.CROSS_SPAWN_MIN, self.CROSS_SPAWN_MAX)
            side = 'right' if random.random() < 0.5 else 'left'
            speed = random.uniform(*self.CROSS_SPEED_RANGE)
            lateral = random.uniform(*self.CROSS_LATERAL_RANGE)
            _ = self._spawn_crossing_walker_ahead(d=d, side=side, speed=speed, lateral=lateral)
            # consume this trigger regardless of spawn success to avoid blocking
            self._cross_triggers.pop(0)


        speed_post     = self.get_speed()
        lat_error_post = self.get_lat_error()       
        
        
        scan = self._scan_surroundings()
 

#############################################################################################################################        
#        print(
#            f"[DBG] v={self.get_speed():.2f}m/s "
#            f"lane={scan['ego_lane']} d_e={scan.get('d_e',0.0):.2f} "
#            f"ego_gapF={scan['ego_gapF']:.2f} ego_ttcF={scan['ego_ttcF']:.2f} "
#            f"L_gapF={scan.get('l_gapF',0.0):.2f} L_ttcF={scan.get('l_ttcF',0.0):.2f} "
#            f"R_gapF={scan.get('r_gapF',0.0):.2f} R_ttcF={scan.get('r_ttcF',0.0):.2f} "
#            f"left_ok={scan['left_ok']:.0f} right_ok={scan['right_ok']:.0f} "
#            f"closing={scan.get('ego_closing',0.0):.0f}"
#        )
       
        obs    = self._get_full_obs(scan)
        reward = self.compute_reward(action, scan)
        

        
        
        _, idx = self.find_current_waypoint_with_index(self.vehicle.get_transform().location)
        if idx > self.last_waypoint_idx:
            self.last_waypoint_idx = int(idx)

        delta_idx = self.last_waypoint_idx - prev_idx
        if delta_idx > 0:
            reward += 0.4 * delta_idx      
        
        
        
        

       
        self.step_count += 1
       
        

        
        terminated = False
        truncated  = False
       
        
#############################################################################################################################        
        
      
 #       if aff and (sdist_chk is not None) and (st_chk in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)):
            
#            print(f"[TL DIST] stopline_sdist={sdist_chk:.2f} m ({'ahead' if sdist_chk >= 0 else 'past'})")
        
###############################################################################################################################        

        _, sdist_chk, st_chk, _ = self._red_light_info(lookahead_points=self.TL_LOOKAHEAD)  

        is_red_or_yellow = (st_chk in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow))
       
        tl_close = (
            is_red_or_yellow
            and (sdist_chk is not None)
            and (sdist_chk >= 0.0)
            and (sdist_chk <= self.RL_STOP_START)
        )

       
       
        blocked_by_traffic = (
            (scan["ego_gapF"] < self.GAP_SAFE_F) and
            (scan["ego_ttcF"] < self.TTC_SAFE_F)           
        )
        
        ped_close = (
            float(scan.get("ped_active", 0.0)) >= 0.5 and
            float(scan.get("ped_gap", self.GAP_CLIP)) <= self.RL_STOP_START
        )

        
        
        # ===== STOP sign state machine =====
        aff_s, sdist_s, _, stop_idx = self._stop_info(lookahead_points=self.STOP_LOOKAHEAD)
#        print(f"[STOP_INFO] aff_s={aff_s}, sdist_s={sdist_s}, "
#              f"stop_idx={stop_idx}, stopped_idxs={self._stopped_idxs}")
        if st_chk is not None:               # if traffic light,dont consider stopline
            aff_s, sdist_s = False, None
            
            
        stop_close = (aff_s and (sdist_s is not None) and (0.0 <= sdist_s <= self.RL_STOP_START))

        if stop_close:
            self._stop_active = True
            self._curr_stop_idx = stop_idx

        if self._stop_active and (stop_idx == self._curr_stop_idx):
            # Near the line and essentially stationary → Start counting
            if (sdist_s is not None) and (-0.3 <= sdist_s <= self.RL_STOP_START):
                if speed_post < 0.2:
                    self._stop_hold += 1
                else:
                    self._stop_hold = 0

            # Hold for 30 frames → Release
            if self._stop_hold >= self.STOP_HOLD_STEPS:
                self._stopped_idxs.add(self._curr_stop_idx)
                self._stop_active = False
                self._stop_hold = 0
                self._curr_stop_idx = None

          
            if (sdist_s is not None) and (sdist_s < -0.3) and (self._stop_hold < self.STOP_HOLD_STEPS):
                if not self._stop_violation_latched:
                    reward -= 100.0
                    print(f"[Stopline VIOLATION] sdist={sdist_s:.2f} m, v={speed_post:.2f} m/s")
                    self._stop_violation_latched = True
                    terminated = True
        else:
           
            self._stop_active = False
            self._stop_hold = 0
        
        
        
        
        
        no_stop_line = (not tl_close) and (not ped_close) and (not stop_close)       
        
        
        
        idle_now = (no_stop_line and (not blocked_by_traffic) and (speed_post < self.IDLE_V_THRESH))
        if idle_now:
            reward += self.IDLE_STEP_PENALTY

        
        if tl_close or ped_close or blocked_by_traffic or stop_close:
            self.stuck_counter = 0
        else:
            if speed_post < self.IDLE_V_THRESH:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

########################################################################################################            
            
        
#        print(f"[STOP_STATE] stop_active={self._stop_active}, stop_hold={self._stop_hold}, "
#              f"stop_close={stop_close}, stuck_counter={self.stuck_counter}")
            
            
        

        if not terminated and self.stuck_counter >= self.STUCK_RESET_STEPS:
            print(f"Vehicle stuck: v<{self.IDLE_V_THRESH} m/s for {self.STUCK_RESET_STEPS} frames without red/yellow/block. Resetting.")
            reward += self.STUCK_RESET_PENALTY
            terminated = True

        
        if (sdist_chk is None) or (st_chk not in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)) or (sdist_chk >= 0.0):
            self._tl_violation_latched = False

        
        if (sdist_chk is not None) and (st_chk in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)) and (sdist_chk < 0.0):

            stopped_tolerated = (sdist_chk > -0.5) and (speed_post < 0.1)  
            if not stopped_tolerated and not self._tl_violation_latched:
                self._tl_violation_latched = True
                reward -= 100.0  
                print(f"[TL VIOLATION] sdist={sdist_chk:.2f} m, v={speed_post:.2f} m/s, tl={getattr(st_chk,'name',st_chk)}")
                terminated = True 
                


#########################################################################################################################################


                
                
        
        REACH_THRESH_M = 2.0
        
        
                
        rem_len = self._remaining_arc_length(self.last_waypoint_idx)
        reached_by_arc = (rem_len <= REACH_THRESH_M)
        reached_by_idx = (self.last_waypoint_idx >= len(self.route) - 1)

        if reached_by_arc or reached_by_idx:
            reward += 100.0
            reason = f"rem={rem_len:.2f}m" if reached_by_arc else "index-end"
            print(f"Reached destination ({reason}). Replanning a new sub-route and continuing...")
            ok = self._replan_route_from_here()
            if ok:
                 
                obs = self._get_full_obs()
            else:
                print("[Route] Replan failed. Ending episode.")
                terminated = True
                
             
        if not terminated and self.get_collision_data():
            pen  = getattr(self, "collision_penalty", -50.0)
            kind = getattr(self, "collision_kind", "other")
            print(f"Collision with {kind}: penalty {pen}. Resetting environment.")
            reward += pen  
            terminated = True           
#            print("Collision detected! Resetting environment.")
#            reward -= 5.0
#            terminated = True
            
        if not terminated and self.is_off_lane():
            print("Went off the drivable lane. Resetting.")
            reward -= 10.0
            terminated = True

        if not terminated:
            if abs(lat_error_post) > 8:
                self.offroute_count += 1
            else:
                self.offroute_count = 0
            if self.offroute_count >= 10:
                print("Deviating from the established route.")
                reward -= 10.0
                terminated = True             
        if not terminated and self.step_count >= self.max_episode_steps:
                truncated = True
        info = {}
        if terminated or truncated:
            final_obs = obs
            final_info = {}  
            info["final_observation"] = final_obs
            info["final_info"] = final_info
            if self.auto_reset:
               
                new_obs, _ = self.reset()
        return obs, reward, terminated, truncated, info
           


    def _get_full_obs(self, scan=None):
        lat_error = self.get_lat_error()                     # m
        speed     = self.get_speed()   # m/s
#        print(f"speed={speed}     ")
        yaw_diff  = self.get_yaw_diff()
        future_h  = self.get_future_heading()
        steer     = self.last_steer
 #       throttle  = self.last_throttle
        long_cmd = self.last_long_cmd
        
        
        _, sdist, tl_state, _ = self._red_light_info(lookahead_points=self.TL_LOOKAHEAD)
        tl_present = (tl_state is not None)

      

        is_red_or_yellow = (tl_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow))
#        tl_active = float(is_red_or_yellow)
        if sdist is None:
            sdist_eff = self.RL_SENSE_DIST
        else:
            sdist_eff = sdist
        sdist_clip = max(-self.stopline_norm_dist, min(self.stopline_norm_dist, sdist_eff))
        sdist_norm = sdist_clip / self.stopline_norm_dist
        
        aff_s, sdist_s, _, _ = self._stop_info(lookahead_points=self.STOP_LOOKAHEAD)
        if tl_present:                                  # if traffic light, dont consider stopline
            stop_sdist_eff = self.RL_SENSE_DIST         # 40.0
        elif (not aff_s) or (sdist_s is None):
            stop_sdist_eff = self.RL_SENSE_DIST          
        else:
            
            if sdist_s >= 0.0:
                stop_sdist_eff = min(sdist_s, self.RL_SENSE_DIST)
            else:
                stop_sdist_eff = sdist_s
        stop_sdist_clip = max(-self.stopline_norm_dist, min(self.stopline_norm_dist, stop_sdist_eff))
        stop_sdist_norm = stop_sdist_clip / self.stopline_norm_dist
            
        if scan is None:
            scan = self._scan_surroundings()
        gapF = float(scan["ego_gapF"])     
        gapF_norm = min(1.0, gapF / self.GAP_CLIP)
        ttcF_norm = min(1.0, float(scan["ego_ttcF"]) / self.TTC_CLIP)
        left_ok   = float(scan["left_ok"])
        right_ok  = float(scan["right_ok"])
        ego_lane  = int(scan.get("ego_lane", 0))        
#        closing_flag = float(scan.get("ego_closing", 0.0)) 
        ped_gap_norm = float(scan.get("ped_gap", self.GAP_CLIP)) / self.GAP_CLIP
#        print(f"[OBS_RAW] traffic light={sdist_eff:.2f} m, stopline={stop_sdist_eff:.2f} m, pedes={float(scan.get('ped_gap', self.GAP_CLIP)):.2f} m")


        obs = [
            lat_error / 3.5,
            speed / self.max_speed,
            max(-1.0, min(1.0, yaw_diff / math.pi)),
            max(-1.0, min(1.0, future_h)),
            steer,
            long_cmd,
            sdist_norm,
            gapF_norm,
            ttcF_norm,
            left_ok,
            right_ok,
            ped_gap_norm, 
            stop_sdist_norm,
        ]
        return np.array(obs, dtype=np.float32)

 
    def enable_rendering(self, on=True):
        s = self.world.get_settings()
        s.no_rendering_mode = (not on)
        self.world.apply_settings(s)
        self.world.tick()

    def close(self):
        
        try:
            self._destroy_npcs()
        except:
            pass
        if self.collision_sensor:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except:
                pass
            self.collision_sensor = None

        if self.vehicle:
            try:
                self.vehicle.destroy()
            except:
                pass
            self.vehicle = None
            
        try:
            self._destroy_walkers()
        except:
            pass

