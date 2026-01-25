# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""
Prototype of base classes
Replacement of the old RaceCar, Simulator classes in C++
Author: Hongrui Zheng
"""
from enum import Enum
import warnings

import numpy as np
from numba import njit

from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid
from f110_gym.envs.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from f110_gym.envs.collision_models import get_vertices, collision_multiple

@njit(cache=True)
def resolve_collision(state1, state2, mass, restitution=0.1):
    """
    Resolve collision between two agents using impulse-based physics
    
    Args:
        state1, state2 (np.ndarray(7,)): vehicle states [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        mass (float): mass of vehicles (assumed equal)
        restitution (float): coefficient of restitution (0=perfectly inelastic, 1=perfectly elastic)
    
    Returns:
        new_vel1, new_vel2 (float): updated velocities for both vehicles
    """
    # Extract positions and velocities
    pos1 = state1[0:2]
    pos2 = state2[0:2]
    vel1 = state1[3]
    vel2 = state2[3]
    yaw1 = state1[4]
    yaw2 = state2[4]
    
    # Velocity vectors in world frame
    vel1_vec = np.array([vel1 * np.cos(yaw1), vel1 * np.sin(yaw1)])
    vel2_vec = np.array([vel2 * np.cos(yaw2), vel2 * np.sin(yaw2)])
    
    # Collision normal (from agent1 to agent2)
    collision_normal = pos2 - pos1
    dist = np.sqrt(collision_normal[0]**2 + collision_normal[1]**2)
    
    if dist < 1e-6:  # Avoid division by zero
        return vel1, vel2
    
    collision_normal = collision_normal / dist
    
    # Relative velocity
    rel_vel = vel1_vec - vel2_vec
    
    # Velocity along collision normal
    vel_along_normal = rel_vel[0] * collision_normal[0] + rel_vel[1] * collision_normal[1]
    
    # Don't resolve if velocities are separating
    if vel_along_normal > 0:
        return vel1, vel2
    
    # Calculate impulse magnitude (inelastic collision)
    impulse_magnitude = -(1.0 + restitution) * vel_along_normal / 2.0
    
    # Apply impulse
    impulse = impulse_magnitude * collision_normal
    
    # Update velocity vectors
    new_vel1_vec = vel1_vec + impulse
    new_vel2_vec = vel2_vec - impulse
    
    # Convert back to scalar velocities (magnitude in direction of heading)
    # Project onto current heading direction
    heading1 = np.array([np.cos(yaw1), np.sin(yaw1)])
    heading2 = np.array([np.cos(yaw2), np.sin(yaw2)])
    
    new_vel1 = new_vel1_vec[0] * heading1[0] + new_vel1_vec[1] * heading1[1]
    new_vel2 = new_vel2_vec[0] * heading2[0] + new_vel2_vec[1] * heading2[1]
    
    return new_vel1, new_vel2

class Integrator(Enum):
    RK4 = 1
    Euler = 2


class RaceCar(object):
    """
    Base level race car class, handles the physics and laser scan of a single vehicle

    Data Members:
        params (dict): vehicle parameters dictionary
        is_ego (bool): ego identifier
        time_step (float): physics timestep
        num_beams (int): number of beams in laser
        fov (float): field of view of laser
        state (np.ndarray (7, )): state vector [x, y, theta, vel, steer_angle, ang_vel, slip_angle]
        odom (np.ndarray(13, )): odometry vector [x, y, z, qx, qy, qz, qw, linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        accel (float): current acceleration input
        steer_angle_vel (float): current steering velocity input
        in_collision (bool): collision indicator

    """

    # static objects that don't need to be stored in class instances
    scan_simulator = None
    cosines = None
    scan_angles = None
    side_distances = None

    def __init__(self, params, seed, is_ego=False, time_step=0.01, num_beams=1080, fov=4.7, integrator=Integrator.Euler, lidar_dist=0.0):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max': 9.51, 'v_min', 'v_max', 'length', 'width'}
            is_ego (bool, default=False): ego identifier
            time_step (float, default=0.01): physics sim time step
            num_beams (int, default=1080): number of beams in the laser scan
            fov (float, default=4.7): field of view of the laser
            lidar_dist (float, default=0): vertical distance between LiDAR and backshaft

        Returns:
            None
        """

        # initialization
        self.params = params
        self.seed = seed
        self.is_ego = is_ego
        self.time_step = time_step
        self.num_beams = num_beams
        self.fov = fov
        self.integrator = integrator
        self.lidar_dist = lidar_dist
        if self.integrator is Integrator.RK4:
            warnings.warn(f"Chosen integrator is RK4. This is different from previous versions of the gym.")

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        self.state = np.zeros((7, ))

        # world-frame velocities (calculated from position changes)
        self.world_vel = np.zeros((2, ))  # [vx, vy] in world frame
        self.prev_pos = np.zeros((2, ))  # previous position for velocity calculation

        # pose of opponents in the world
        self.opp_poses = None

        # control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0

        # steering delay buffer
        self.steer_buffer = np.empty((0, ))
        self.steer_buffer_size = 2

        # collision identifier
        self.in_collision = False

        # collision threshold for iTTC to environment
        self.ttc_thresh = 0.005

        # initialize scan sim
        if RaceCar.scan_simulator is None:
            self.scan_rng = np.random.default_rng(seed=self.seed)
            RaceCar.scan_simulator = ScanSimulator2D(num_beams, fov)

            scan_ang_incr = RaceCar.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam, and precomputed cosines of each angle
            RaceCar.cosines = np.zeros((num_beams, ))
            RaceCar.scan_angles = np.zeros((num_beams, ))
            RaceCar.side_distances = np.zeros((num_beams, ))

            dist_sides = params['width']/2.
            dist_fr = (params['lf']+params['lr'])/2.

            for i in range(num_beams):
                angle = -fov/2. + i*scan_ang_incr
                RaceCar.scan_angles[i] = angle
                RaceCar.cosines[i] = np.cos(angle)

                if angle > 0:
                    if angle < np.pi/2:
                        # between 0 and pi/2
                        to_side = dist_sides / np.sin(angle)
                        to_fr = dist_fr / np.cos(angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between pi/2 and pi
                        to_side = dist_sides / np.cos(angle - np.pi/2.)
                        to_fr = dist_fr / np.sin(angle - np.pi/2.)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                else:
                    if angle > -np.pi/2:
                        # between 0 and -pi/2
                        to_side = dist_sides / np.sin(-angle)
                        to_fr = dist_fr / np.cos(-angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between -pi/2 and -pi
                        to_side = dist_sides / np.cos(-angle - np.pi/2)
                        to_fr = dist_fr / np.sin(-angle - np.pi/2)
                        RaceCar.side_distances[i] = min(to_side, to_fr)

    def update_params(self, params):
        """
        Updates the physical parameters of the vehicle
        Note that does not need to be called at initialization of class anymore

        Args:
            params (dict): new parameters for the vehicle

        Returns:
            None
        """
        self.params = params
    
    def set_map(self, map_path, map_ext):
        """
        Sets the map for scan simulator
        
        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file
        """
        RaceCar.scan_simulator.set_map(map_path, map_ext)

    def reset(self, pose):
        """
        Resets the vehicle to a pose
        
        Args:
            pose (np.ndarray (3, )): pose to reset the vehicle to

        Returns:
            None
        """
        # clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0
        # clear collision indicator
        self.in_collision = False
        self.collision_type = 0
        # clear state
        self.state = np.zeros((7, ))
        self.state[0:2] = pose[0:2]
        self.state[4] = pose[2]
        # reset world velocities
        self.world_vel = np.zeros((2, ))
        self.prev_pos = pose[0:2].copy()
        self.steer_buffer = np.empty((0, ))
        # reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.seed)

    def ray_cast_agents(self, scan):
        """
        Ray cast onto other agents in the env, modify original scan

        Args:
            scan (np.ndarray, (n, )): original scan range array

        Returns:
            new_scan (np.ndarray, (n, )): modified scan
        """

        # starting from original scan
        new_scan = scan
        
        # loop over all opponent vehicle poses
        for opp_pose in self.opp_poses:
            # get vertices of current oppoenent
            opp_vertices = get_vertices(opp_pose, self.params['length'], self.params['width'])

            new_scan = ray_cast(np.append(self.state[0:2], self.state[4]), new_scan, self.scan_angles, opp_vertices)

        return new_scan

    def check_ttc(self, current_scan):
        """
        Check iTTC against the environment, sets vehicle states accordingly if collision occurs.
        Note that this does NOT check collision with other agents.

        state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        Args:
            current_scan

        Returns:
            None
        """
        
        in_collision = check_ttc_jit(current_scan, self.state[3], self.scan_angles, self.cosines, self.side_distances, self.ttc_thresh)

        # Mark collision but don't stop vehicle here - let collision response handle it
        # This allows proper physics response for different collision types
        self.in_collision = in_collision

        return in_collision

    def update_pose(self, raw_steer, vel):
        """
        Steps the vehicle's physical simulation

        Args:
            steer (float): desired steering angle
            vel (float): desired longitudinal velocity

        Returns:
            current_scan
        """

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        # steering delay
        steer = 0.
        if self.steer_buffer.shape[0] < self.steer_buffer_size:
            steer = 0.
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)
        else:
            steer = self.steer_buffer[-1]
            self.steer_buffer = self.steer_buffer[:-1]
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)


        # steering angle velocity input to steering velocity acceleration input
        accl, sv = pid(vel, steer, self.state[3], self.state[2], self.params['sv_max'], self.params['a_max'], self.params['v_max'], self.params['v_min'])
        
        if self.integrator is Integrator.RK4:
            # RK4 integration
            k1 = vehicle_dynamics_st(
                self.state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])

            k2_state = self.state + self.time_step*(k1/2)

            k2 = vehicle_dynamics_st(
                k2_state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])

            k3_state = self.state + self.time_step*(k2/2)

            k3 = vehicle_dynamics_st(
                k3_state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])

            k4_state = self.state + self.time_step*k3

            k4 = vehicle_dynamics_st(
                k4_state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])

            # dynamics integration
            self.state = self.state + self.time_step*(1/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        elif self.integrator is Integrator.Euler:
            f = vehicle_dynamics_st(
                self.state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])
            self.state = self.state + self.time_step * f
        
        else:
            raise SyntaxError(f"Invalid Integrator Specified. Provided {self.integrator.name}. Please choose RK4 or Euler")

        # bound yaw angle
        if self.state[4] > 2*np.pi:
            self.state[4] = self.state[4] - 2*np.pi
        elif self.state[4] < 0:
            self.state[4] = self.state[4] + 2*np.pi

        # calculate world-frame velocities from position changes
        current_pos = self.state[0:2]
        self.world_vel = (current_pos - self.prev_pos) / self.time_step
        self.prev_pos = current_pos.copy()

        # update scan
        scan_x = self.state[0] + self.lidar_dist*np.cos(self.state[4])
        scan_y = self.state[1] + self.lidar_dist*np.sin(self.state[4])
        scan_pose = np.array([scan_x, scan_y, self.state[4]])
        current_scan = RaceCar.scan_simulator.scan(scan_pose, self.scan_rng)
        # current_scan = RaceCar.scan_simulator.scan(np.append(self.state[0:2],  self.state[4]), self.scan_rng)

        return current_scan

    def update_opp_poses(self, opp_poses):
        """
        Updates the vehicle's information on other vehicles

        Args:
            opp_poses (np.ndarray(num_other_agents, 3)): updated poses of other agents

        Returns:
            None
        """
        self.opp_poses = opp_poses


    def update_scan(self, agent_scans, agent_index):
        """
        Steps the vehicle's laser scan simulation
        Separated from update_pose because needs to update scan based on NEW poses of agents in the environment

        Args:
            agent scans list (modified in-place),
            agent index (int)

        Returns:
            None
        """

        current_scan = agent_scans[agent_index]

        # check ttc
        self.check_ttc(current_scan)

        # ray cast other agents to modify scan
        new_scan = self.ray_cast_agents(current_scan)

        agent_scans[agent_index] = new_scan

class Simulator(object):
    """
    Simulator class, handles the interaction and update of all vehicles in the environment

    Data Members:
        num_agents (int): number of agents in the environment
        time_step (float): physics time step
        agent_poses (np.ndarray(num_agents, 3)): all poses of all agents
        agents (list[RaceCar]): container for RaceCar objects
        collisions (np.ndarray(num_agents, )): array of collision indicator for each agent
        collision_idx (np.ndarray(num_agents, )): which agent is each agent in collision with

    """

    def __init__(self, params, num_agents, seed, time_step=0.01, ego_idx=0, integrator=Integrator.RK4, lidar_dist=0.0):
        """
        Init function

        Args:
            params (dict): vehicle parameter dictionary, includes {'mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'm', 'I', 's_min', 's_max', 'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max', 'length', 'width'}
            num_agents (int): number of agents in the environment
            seed (int): seed of the rng in scan simulation
            time_step (float, default=0.01): physics time step
            ego_idx (int, default=0): ego vehicle's index in list of agents
            lidar_dist (float, default=0): vertical distance between LiDAR and backshaft

        Returns:
            None
        """
        self.num_agents = num_agents
        self.seed = seed
        self.time_step = time_step
        self.ego_idx = ego_idx
        self.params = params
        self.agent_poses = np.empty((self.num_agents, 3))
        self.agents = []
        self.collisions = np.zeros((self.num_agents, ))
        self.collision_idx = -1 * np.ones((self.num_agents, ))

        # initializing agents
        for i in range(self.num_agents):
            if i == ego_idx:
                ego_car = RaceCar(params, self.seed, is_ego=True, time_step=self.time_step, integrator=integrator, lidar_dist=lidar_dist)
                self.agents.append(ego_car)
            else:
                agent = RaceCar(params, self.seed, is_ego=False, time_step=self.time_step, integrator=integrator, lidar_dist=lidar_dist)
                self.agents.append(agent)

    def set_map(self, map_path, map_ext):
        """
        Sets the map of the environment and sets the map for scan simulator of each agent

        Args:
            map_path (str): path to the map yaml file
            map_ext (str): extension for the map image file

        Returns:
            None
        """
        for agent in self.agents:
            agent.set_map(map_path, map_ext)


    def update_params(self, params, agent_idx=-1):
        """
        Updates the params of agents, if an index of an agent is given, update only that agent's params

        Args:
            params (dict): dictionary of params, see details in docstring of __init__
            agent_idx (int, default=-1): index for agent that needs param update, if negative, update all agents

        Returns:
            None
        """
        if agent_idx < 0:
            # update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif agent_idx >= 0 and agent_idx < self.num_agents:
            # only update one agent's params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError('Index given is out of bounds for list of agents.')

    def check_collision(self):
        """
        Checks for collision between agents using GJK and agents' body vertices

        Args:
            None

        Returns:
            None
        """
        # get vertices of all agents
        all_vertices = np.empty((self.num_agents, 4, 2))
        for i in range(self.num_agents):
            all_vertices[i, :, :] = get_vertices(np.append(self.agents[i].state[0:2],self.agents[i].state[4]), self.params['length'], self.params['width'])
        self.collisions, self.collision_idx = collision_multiple(all_vertices)


    def step(self, control_inputs, agent_idxs=None):
        """
        Steps the simulation environment

        Args:
            control_inputs (np.ndarray (num_agents, 2)): control inputs of all agents, first column is desired steering angle, second column is desired velocity
        
        Returns:
            observations (dict): dictionary for observations: poses of agents, current laser scan of each agent, collision indicators, etc.
        """
        
        agent_scans = []
        
        if agent_idxs is None:
            agent_idxs = range(self.num_agents)

        for i in agent_idxs:
            agent = self.agents[i]
            # Reset collision type each step - will be set again if collision detected
            agent.collision_type = 0
            current_scan = agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])
            agent_scans.append(current_scan)
            
            self.agent_poses[i, :] = np.append(agent.state[0:2], agent.state[4])
            
        # Check for agent-to-agent collisions (sets collision = 2 if agent collision)
        self.check_collision()
        
        # Apply collision response for agent-to-agent collisions
        # Track which agents have been processed to avoid double-processing
        processed = set()
        
        for agent_idx in agent_idxs:
            if self.collisions[agent_idx] == 2. and agent_idx not in processed:  # Agent collision detected
                other_idx = int(self.collision_idx[agent_idx])
                
                if other_idx >= 0:  # Valid collision partner
                    # Get states directly (avoid copy)
                    state1 = self.agents[agent_idx].state
                    state2 = self.agents[other_idx].state
                    
                    # Calculate separation vector
                    dx = state2[0] - state1[0]
                    dy = state2[1] - state1[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 1e-6:  # Avoid division by zero
                        # Normalize collision normal
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Approximate vehicle radius (diagonal of bounding box / 2)
                        vehicle_radius = np.sqrt(self.params['length']**2 + self.params['width']**2) / 2.0
                        overlap = 2.0 * vehicle_radius - dist
                        
                        if overlap > 0:  # Agents are overlapping
                            # Separate agents by moving them apart
                            sep_dist = overlap / 2.0
                            state1[0] -= nx * sep_dist
                            state1[1] -= ny * sep_dist
                            state2[0] += nx * sep_dist
                            state2[1] += ny * sep_dist
                            
                            # Update agent poses after separation
                            self.agent_poses[agent_idx, 0:2] = state1[0:2]
                            self.agent_poses[other_idx, 0:2] = state2[0:2]
                        
                        # Physics-based collision response using mass and friction
                        vel1 = state1[3]
                        vel2 = state2[3]
                        yaw1 = state1[4]
                        yaw2 = state2[4]
                        
                        # Velocity vectors
                        v1x = vel1 * np.cos(yaw1)
                        v1y = vel1 * np.sin(yaw1)
                        v2x = vel2 * np.cos(yaw2)
                        v2y = vel2 * np.sin(yaw2)
                        
                        # Relative velocity along collision normal
                        dvx = v1x - v2x
                        dvy = v1y - v2y
                        vel_along_normal = dvx * nx + dvy * ny
                        
                        # Only apply impulse if approaching (prevents rubber banding)
                        if vel_along_normal > 0:
                            # Get masses for both agents (may be different)
                            mass1 = self.agents[agent_idx].params['m']
                            mass2 = self.agents[other_idx].params['m']
                            
                            # Coefficient of restitution - very low to prevent bouncing
                            friction_factor = min(1.0, self.params['mu'] / 1.0)  # Normalized to default mu
                            restitution = 0.02 * friction_factor  # Very inelastic to prevent rubber banding
                            
                            # Impulse magnitude for potentially different masses
                            # J = (1 + e) * v_rel / (1/m1 + 1/m2)
                            impulse_magnitude = (1.0 + restitution) * vel_along_normal / (1.0/mass1 + 1.0/mass2)
                            
                            # Apply friction-based damping to perpendicular velocity
                            # Tangent direction (perpendicular to collision normal)
                            tx = -ny
                            ty = nx
                            vel_tangent = dvx * tx + dvy * ty
                            
                            # Friction reduces tangential velocity difference
                            friction_impulse = vel_tangent * self.params['mu'] * 0.15
                            
                            # Update velocities with impulse and friction
                            # Agent 1: subtract impulse (direction is from 1 to 2)
                            v1x -= (impulse_magnitude / mass1) * nx + friction_impulse * tx
                            v1y -= (impulse_magnitude / mass1) * ny + friction_impulse * ty
                            # Agent 2: add impulse
                            v2x += (impulse_magnitude / mass2) * nx + friction_impulse * tx
                            v2y += (impulse_magnitude / mass2) * ny + friction_impulse * ty
                            
                            # Add damping to reduce oscillations (prevents rubber banding)
                            damping = 0.85
                            v1x *= damping
                            v1y *= damping
                            v2x *= damping
                            v2y *= damping
                            
                            # Project back to heading direction
                            cos_yaw1 = np.cos(yaw1)
                            sin_yaw1 = np.sin(yaw1)
                            cos_yaw2 = np.cos(yaw2)
                            sin_yaw2 = np.sin(yaw2)
                            
                            new_vel1 = v1x * cos_yaw1 + v1y * sin_yaw1
                            new_vel2 = v2x * cos_yaw2 + v2y * sin_yaw2
                            
                            # Apply velocity limits from params
                            state1[3] = np.clip(new_vel1, self.params['v_min'], self.params['v_max'])
                            state2[3] = np.clip(new_vel2, self.params['v_min'], self.params['v_max'])
                    
                    # Mark both as processed
                    processed.add(agent_idx)
                    processed.add(other_idx)
                    
                    # Mark collision type
                    self.agents[agent_idx].in_collision = True
                    self.agents[other_idx].in_collision = True
                    self.agents[agent_idx].collision_type = 2  # Agent collision
                    self.agents[other_idx].collision_type = 2  # Agent collision
        
        for i, agent_idx in enumerate(agent_idxs):
            agent = self.agents[agent_idx]
            opp_poses = np.concatenate((self.agent_poses[0:agent_idx, :], self.agent_poses[agent_idx+1:, :]), axis=0)
            agent.update_opp_poses(opp_poses)

            # update each agent's current scan based on other agents
            agent.update_scan(agent_scans, i)

            # update agent collision with environment (wall collisions = 1)
            # Only set wall collision if not already in agent-to-agent collision
            if agent.in_collision and self.collisions[agent_idx] == 0:
                self.collisions[agent_idx] = 1.
                agent.collision_type = 1
                
                # Penetration correction: move agent back and project velocity along wall
                state = agent.state
                
                # Move back to previous valid position
                state[0:2] = agent.prev_pos
                self.agent_poses[agent_idx, 0:2] = agent.prev_pos
                
                # Get world-frame velocity
                vx = agent.world_vel[0]
                vy = agent.world_vel[1]
                vel_mag = np.sqrt(vx*vx + vy*vy)
                
                if vel_mag > 1e-6:
                    # Collision normal points opposite to velocity (away from wall)
                    nx = -vx / vel_mag
                    ny = -vy / vel_mag
                    
                    # Velocity component perpendicular to wall (remove this)
                    vel_normal = vx * nx + vy * ny
                    
                    # Only remove component going into wall
                    if vel_normal < 0:
                        # Remove normal component, keep tangential (slide along wall)
                        vx -= vel_normal * nx
                        vy -= vel_normal * ny
                        
                        # Project back to vehicle heading direction
                        yaw = state[4]
                        new_vel = vx * np.cos(yaw) + vy * np.sin(yaw)
                        state[3] = np.clip(new_vel, self.params['v_min'], self.params['v_max'])
                
        # fill in observations
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # collisions: 0 = no collision, 1 = wall collision, 2 = agent collision
        # collision_idx: -1 = no agent collision, >= 0 = index of agent in collision with
        observations = {'ego_idx': self.ego_idx,
            'scans': [],
            'poses_x': [],
            'poses_y': [],
            'poses_theta': [],
            'linear_vels_x': [],
            'linear_vels_y': [],
            'ang_vels_z': [],
            'linear_accel_x': [],
            'linear_accel_y': [],
            'collisions': self.collisions[agent_idxs],
            'collision_idx': self.collision_idx[agent_idxs],}
        
        for i, agent_idx in enumerate(agent_idxs):
            agent = self.agents[agent_idx]
            agent_scan = agent_scans[i]
            observations['scans'].append(agent_scan)
            observations['poses_x'].append(agent.state[0])
            observations['poses_y'].append(agent.state[1])
            observations['poses_theta'].append(agent.state[4])
            observations['linear_vels_x'].append(agent.state[3])
            observations['linear_vels_y'].append(0.)
            observations['ang_vels_z'].append(agent.state[5])
            observations['linear_accel_x'].append(agent.accel)
            observations['linear_accel_y'].append(0.)

        return observations

    def reset(self, poses, agent_idxs=None):
        """
        Resets the simulation environment by given poses

        Arges:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            None
        """
        
        if poses.shape[0] != self.num_agents:
            raise ValueError('Number of poses for reset does not match number of agents.')
        
        if agent_idxs is None:
            agent_idxs = range(self.num_agents)

        # loop over poses to reset
        for agent_idx in agent_idxs:
            self.agents[agent_idx].reset(poses[agent_idx, :])
