import torch
import torch.nn as nn
import numpy as np
import os
import glob
import random

# --- Configuration ---
LR = 0.0003
GAMMA = 0.99
K_EPOCHS = 20
EPS_CLIP = 0.2
N_ENVS = 16
UPDATE_TIMESTEP = 2048      # steps per env before update → 2048 * 16 = 32768 samples
MAX_TIMESTEPS = 100_000_000
ACTION_STD_INIT = 0.6
SNAPSHOT_INTERVAL = 100_000

# --- Curriculum Stages ---
START_STAGE = 0
# 0 = Ball Chasing
# 1 = Striker (Goal Scoring)
# 2 = League Play (Self-Play + Hall of Fame)

# --- Physics Constants ---
SCREEN_W, SCREEN_H = 1200, 1000
OFFSET_X, OFFSET_Y = 100, 100
ARENA_W, ARENA_H = SCREEN_W - 200, SCREEN_H - 200
GOAL_SIZE = 200
GOAL_DEPTH = 80
CORNER_SIZE = 80
PHYSICS_SUBSTEPS = 8

DRAG = 0.99
TIRE_GRIP_NORMAL = 0.95
TIRE_GRIP_DRIFT = 0.05
ACCEL = 0.08
TURN_SPEED = 3.5
MAX_SPEED = 6.0
BOOST_ACCEL = 0.15
BOOST_MAX_SPEED = 9.0
MAX_BOOST = 100.0
BALL_ELASTICITY = 0.8
CAR_WALL_ELASTICITY = 0.5
BALL_DRAG = 0.999

# --- Helper Functions ---
def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def rotate_vector(v, angle_degrees):
    rad = np.radians(angle_degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        v[0] * c - v[1] * s,
        v[0] * s + v[1] * c
    ])

CORNER_NORMALS = [
    normalize(np.array([0.707, 0.707])), normalize(np.array([-0.707, 0.707])),
    normalize(np.array([-0.707, -0.707])), normalize(np.array([0.707, -0.707]))
]

CORNER_POINTS = [
    np.array([OFFSET_X + CORNER_SIZE, OFFSET_Y]),
    np.array([OFFSET_X + ARENA_W - CORNER_SIZE, OFFSET_Y]),
    np.array([OFFSET_X + ARENA_W - CORNER_SIZE, OFFSET_Y + ARENA_H]),
    np.array([OFFSET_X + CORNER_SIZE, OFFSET_Y + ARENA_H])
]

def get_rel_obs(my_car, opp_car, ball):
    obs = [
        my_car.pos[0] / SCREEN_W, my_car.pos[1] / SCREEN_H,
        my_car.vel[0] / 10, my_car.vel[1] / 10,
        np.cos(np.radians(my_car.angle)), np.sin(np.radians(my_car.angle)),
        my_car.boost / 100.0,
        ball.pos[0] / SCREEN_W, ball.pos[1] / SCREEN_H,
        ball.vel[0] / 10, ball.vel[1] / 10,
        opp_car.pos[0] / SCREEN_W, opp_car.pos[1] / SCREEN_H,
        opp_car.vel[0] / 10, opp_car.vel[1] / 10
    ]
    return np.array(obs, dtype=np.float32)

# --- Physics Objects ---
class PhysicsObject:
    def __init__(self, x, y, mass, radius):
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        self.mass = float(mass)
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0
        self.radius = float(radius)

class Ball(PhysicsObject):
    def __init__(self, x, y): super().__init__(x, y, mass=1.0, radius=22)

class Car(PhysicsObject):
    def __init__(self, x, y, angle=0.0):
        super().__init__(x, y, mass=10.0, radius=20)
        self.angle = float(angle)
        self.width, self.height = 40.0, 24.0
        self.boost = 100.0
        self.throttle, self.turn = 0.0, 0.0
        self.boost_active, self.drifting = False, False

    def update_controls(self):
        current_accel = ACCEL
        current_max = MAX_SPEED
        if self.boost_active and self.boost > 0:
            current_accel = BOOST_ACCEL; current_max = BOOST_MAX_SPEED; self.boost -= 0.15
        elif self.boost < MAX_BOOST:
            self.boost += 0.03

        vel_mag = np.linalg.norm(self.vel)
        if vel_mag > 0.5 or self.drifting:
            orientation = rotate_vector(np.array([1.0, 0.0]), self.angle)
            direction = 1.0 if np.dot(self.vel, orientation) > 0 else -1.0
            if self.drifting: direction = 1.0
            self.angle += self.turn * TURN_SPEED * direction * 0.15

        forward = rotate_vector(np.array([1.0, 0.0]), self.angle)
        self.vel += forward * self.throttle * current_accel
        right = rotate_vector(forward, 90)
        lateral_vel = np.dot(self.vel, right)
        grip = TIRE_GRIP_DRIFT if self.drifting else TIRE_GRIP_NORMAL
        self.vel -= right * lateral_vel * grip * 0.2
        self.vel *= DRAG
        vel_mag = np.linalg.norm(self.vel)
        if vel_mag > current_max:
            self.vel = (self.vel / vel_mag) * current_max

# --- Physics Solver ---
def bounce(obj, normal, elasticity):
    vel_along_normal = np.dot(obj.vel, normal)
    if vel_along_normal < 0:
        j = -(1 + elasticity) * vel_along_normal
        obj.vel += j * normal

def resolve_arena_collisions(obj):
    r = obj.radius
    elasticity = BALL_ELASTICITY if isinstance(obj, Ball) else CAR_WALL_ELASTICITY

    goal_top = SCREEN_H / 2 - GOAL_SIZE / 2
    goal_bot = SCREEN_H / 2 + GOAL_SIZE / 2

    tl_post = np.array([OFFSET_X, goal_top])
    bl_post = np.array([OFFSET_X, goal_bot])
    tr_post = np.array([OFFSET_X + ARENA_W, goal_top])
    br_post = np.array([OFFSET_X + ARENA_W, goal_bot])

    for post in [tl_post, bl_post, tr_post, br_post]:
        diff = obj.pos - post
        dist_sq = np.dot(diff, diff)
        if dist_sq < r * r:
            dist = np.sqrt(dist_sq)
            normal = diff / dist if dist > 0 else np.array([1.0, 0.0])
            obj.pos = post + normal * r
            bounce(obj, normal, elasticity)
            return

    if obj.pos[0] - r < OFFSET_X:
        if goal_top < obj.pos[1] < goal_bot:
            back_net = OFFSET_X - GOAL_DEPTH
            if obj.pos[0] - r < back_net:
                obj.pos[0] = back_net + r
                bounce(obj, np.array([1.0, 0.0]), elasticity)
            if obj.pos[0] < OFFSET_X:
                if obj.pos[1] - r < goal_top:
                    obj.pos[1] = goal_top + r
                    bounce(obj, np.array([0.0, 1.0]), elasticity)
                if obj.pos[1] + r > goal_bot:
                    obj.pos[1] = goal_bot - r
                    bounce(obj, np.array([0.0, -1.0]), elasticity)
        else:
            obj.pos[0] = OFFSET_X + r
            bounce(obj, np.array([1.0, 0.0]), elasticity)

    elif obj.pos[0] + r > OFFSET_X + ARENA_W:
        if goal_top < obj.pos[1] < goal_bot:
            back_net = OFFSET_X + ARENA_W + GOAL_DEPTH
            if obj.pos[0] + r > back_net:
                obj.pos[0] = back_net - r
                bounce(obj, np.array([-1.0, 0.0]), elasticity)
            if obj.pos[0] > OFFSET_X + ARENA_W:
                if obj.pos[1] - r < goal_top:
                    obj.pos[1] = goal_top + r
                    bounce(obj, np.array([0.0, 1.0]), elasticity)
                if obj.pos[1] + r > goal_bot:
                    obj.pos[1] = goal_bot - r
                    bounce(obj, np.array([0.0, -1.0]), elasticity)
        else:
            obj.pos[0] = OFFSET_X + ARENA_W - r
            bounce(obj, np.array([-1.0, 0.0]), elasticity)

    if obj.pos[1] - r < OFFSET_Y:
        obj.pos[1] = OFFSET_Y + r
        bounce(obj, np.array([0.0, 1.0]), elasticity)
    elif obj.pos[1] + r > OFFSET_Y + ARENA_H:
        obj.pos[1] = OFFSET_Y + ARENA_H - r
        bounce(obj, np.array([0.0, -1.0]), elasticity)

    for i in range(4):
        anchor = CORNER_POINTS[i]
        normal = CORNER_NORMALS[i]
        diff = obj.pos - anchor
        dist = np.dot(diff, normal)
        if dist < r:
            overlap = r - dist
            obj.pos += normal * overlap
            bounce(obj, normal, elasticity)

def resolve_car_ball(car, ball):
    diff = ball.pos - car.pos
    local_ball = rotate_vector(diff, -car.angle)

    hw, hh = car.width / 2, car.height / 2
    cx = max(-hw, min(local_ball[0], hw))
    cy = max(-hh, min(local_ball[1], hh))

    closest_local = np.array([cx, cy])
    dist_vec = local_ball - closest_local
    dist_sq = np.dot(dist_vec, dist_vec)

    if dist_sq < ball.radius * ball.radius:
        if dist_sq == 0:
            dist_vec = np.array([ball.radius, 0.0])
            dist_sq = ball.radius ** 2

        dist = np.sqrt(dist_sq)
        normal_local = dist_vec / dist
        normal_world = rotate_vector(normal_local, car.angle)
        overlap = ball.radius - dist

        total_inv_mass = car.inv_mass + ball.inv_mass
        move_per_inv_mass = (normal_world * overlap) / total_inv_mass
        car.pos -= move_per_inv_mass * car.inv_mass
        ball.pos += move_per_inv_mass * ball.inv_mass

        rel_vel = ball.vel - car.vel
        vel_along_normal = np.dot(rel_vel, normal_world)
        if vel_along_normal > 0:
            return False

        j = -(1 + BALL_ELASTICITY) * vel_along_normal
        j /= total_inv_mass
        impulse = j * normal_world
        ball.vel += impulse * ball.inv_mass
        car.vel -= impulse * car.inv_mass
        return True
    return False

def resolve_car_car(c1, c2):
    dist_vec = c1.pos - c2.pos
    dist = np.linalg.norm(dist_vec)
    min_dist = 40.0

    if dist < min_dist:
        overlap = min_dist - dist
        if dist == 0:
            dist_vec = np.array([1.0, 0.0])
            dist = 1.0
        normal = dist_vec / dist

        c1.pos += normal * (overlap * 0.5)
        c2.pos -= normal * (overlap * 0.5)

        rel_vel = c1.vel - c2.vel
        vel_along = np.dot(rel_vel, normal)
        if vel_along > 0:
            return

        j = -(1 + 0.5) * vel_along
        j /= (c1.inv_mass + c2.inv_mass)
        impulse = j * normal
        c1.vel += impulse * c1.inv_mass
        c2.vel -= impulse * c2.inv_mass

# --- Environment ---
class RocketLeagueEnv:
    def __init__(self):
        self.car1 = Car(250, SCREEN_H / 2)
        self.car2 = Car(SCREEN_W - 250, SCREEN_H / 2)
        self.ball = Ball(SCREEN_W / 2, SCREEN_H / 2)
        self.stage = START_STAGE
        self.episode_steps = 0
        self.wall_contact_steps = 0

    def reset(self, stage):
        self.stage = stage
        self.episode_steps = 0
        self.wall_contact_steps = 0

        self.ball.vel[:] = 0; self.car1.vel[:] = 0; self.car2.vel[:] = 0
        self.car1.boost = 100; self.car2.boost = 100

        if self.stage == 0:
            self.ball.pos = np.array([
                np.random.uniform(OFFSET_X + 100, OFFSET_X + ARENA_W - 100),
                np.random.uniform(OFFSET_Y + 100, OFFSET_Y + ARENA_H - 100)
            ])
            angle = np.random.uniform(0, 6.28)
            dist = np.random.uniform(300, 500)
            self.car1.pos = self.ball.pos + np.array([np.cos(angle), np.sin(angle)]) * dist
            diff = self.ball.pos - self.car1.pos
            self.car1.angle = np.degrees(np.arctan2(diff[1], diff[0])) + np.random.uniform(-10, 10)
            self.car2.pos = np.array([-1000.0, -1000.0])

        elif self.stage == 1:
            self.ball.pos = np.array([
                np.random.uniform(OFFSET_X + 50, OFFSET_X + ARENA_W - 50),
                np.random.uniform(OFFSET_Y + 50, OFFSET_Y + ARENA_H - 50)
            ])
            self.car1.pos = np.array([
                np.random.uniform(OFFSET_X + 50, OFFSET_X + ARENA_W - 50),
                np.random.uniform(OFFSET_Y + 50, OFFSET_Y + ARENA_H - 50)
            ])
            self.car1.angle = np.random.uniform(0, 360)
            self.car2.pos = np.array([-1000.0, -1000.0])

        elif self.stage == 2:
            self.ball.pos = np.array([SCREEN_W / 2, SCREEN_H / 2])
            self.car1.pos = np.array([OFFSET_X + 100, SCREEN_H / 2]) + np.random.uniform(-10, 10, 2)
            self.car1.angle = 0.0
            self.car2.pos = np.array([OFFSET_X + ARENA_W - 100, SCREEN_H / 2]) + np.random.uniform(-10, 10, 2)
            self.car2.angle = 180.0

        return self._get_obs_pair()

    def step(self, action1, action2):
        self.episode_steps += 1
        self._apply_action(self.car1, action1)
        self._apply_action(self.car2, action2)

        dt = 1.0 / PHYSICS_SUBSTEPS
        hit_ball = False
        enemy_hit_ball = False
        touching_wall = False

        for _ in range(PHYSICS_SUBSTEPS):
            if self.car1.pos[0] > -500: self.car1.update_controls()
            if self.car2.pos[0] > -500: self.car2.update_controls()

            self.car1.pos += self.car1.vel * dt
            self.car2.pos += self.car2.vel * dt
            self.ball.pos += self.ball.vel * dt
            self.ball.vel *= BALL_DRAG

            resolve_arena_collisions(self.car1)
            resolve_arena_collisions(self.car2)
            resolve_arena_collisions(self.ball)

            if resolve_car_ball(self.car1, self.ball): hit_ball = True
            if resolve_car_ball(self.car2, self.ball): enemy_hit_ball = True
            resolve_car_car(self.car1, self.car2)

            if (self.car1.pos[0] - self.car1.radius <= OFFSET_X or
                    self.car1.pos[0] + self.car1.radius >= OFFSET_X + ARENA_W or
                    self.car1.pos[1] - self.car1.radius <= OFFSET_Y or
                    self.car1.pos[1] + self.car1.radius >= OFFSET_Y + ARENA_H):
                touching_wall = True

        if touching_wall:
            self.wall_contact_steps += 1
        else:
            self.wall_contact_steps = 0

        done = False; r1, r2 = 0.0, 0.0
        info = {'hit_ball': hit_ball, 'goal': False}

        # Goal detection matching enginenumpy.py (ball edge + 10px buffer)
        if self.ball.pos[0] - self.ball.radius > OFFSET_X + ARENA_W + 10:
            if abs(self.ball.pos[1] - SCREEN_H / 2) < GOAL_SIZE / 2:
                r1 += 30.0; r2 -= 30.0; done = True; info['goal'] = True
        elif self.ball.pos[0] + self.ball.radius < OFFSET_X - 10:
            if abs(self.ball.pos[1] - SCREEN_H / 2) < GOAL_SIZE / 2:
                r1 -= 30.0; r2 += 30.0; done = True

        # Distance reward
        d1 = np.linalg.norm(self.car1.pos - self.ball.pos)
        r1 -= 0.001
        if d1 < 500: r1 += (500 - d1) * 0.0001

        # Velocity toward ball
        vec_to_ball = (self.ball.pos - self.car1.pos) / (d1 + 1e-5)
        vel_to_ball = np.dot(self.car1.vel, vec_to_ball)
        if vel_to_ball > 2.0: r1 += vel_to_ball * 0.001

        # Boost conservation
        if self.car1.boost > 30.0:
            r1 += 0.005

        # Wall punishment
        if self.wall_contact_steps > 30:
            r1 -= 0.05

        # Ball moving toward own goal
        own_goal_dir = normalize(np.array([OFFSET_X, SCREEN_H / 2]) - self.ball.pos)
        ball_vel_to_own_goal = np.dot(self.ball.vel, own_goal_dir)
        if ball_vel_to_own_goal > 0:
            r1 -= ball_vel_to_own_goal * 0.05

        # Enemy touch penalty
        if enemy_hit_ball:
            r1 -= 0.5

        # Hit rewards
        if hit_ball:
            r1 += 1.0
            goal_center = np.array([OFFSET_X + ARENA_W, SCREEN_H / 2])
            ball_to_goal = normalize(goal_center - self.ball.pos)
            shoot_vel = np.dot(self.ball.vel, ball_to_goal)
            if shoot_vel > 0:
                r1 += shoot_vel * 1.0
            dist_to_own_goal = np.linalg.norm(self.ball.pos - np.array([OFFSET_X, SCREEN_H / 2]))
            if dist_to_own_goal < 300:
                r1 += 5.0

        if self.episode_steps > 1500:
            done = True

        return self._get_obs_pair(), (r1, r2), done, info

    def _get_obs_pair(self):
        obs1 = get_rel_obs(self.car1, self.car2, self.ball)
        c2_fake = Car(SCREEN_W - self.car2.pos[0], SCREEN_H - self.car2.pos[1], (self.car2.angle + 180) % 360)
        c2_fake.vel, c2_fake.boost = -self.car2.vel, self.car2.boost
        c1_fake = Car(SCREEN_W - self.car1.pos[0], SCREEN_H - self.car1.pos[1])
        c1_fake.vel = -self.car1.vel
        b_fake = Ball(SCREEN_W - self.ball.pos[0], SCREEN_H - self.ball.pos[1])
        b_fake.vel = -self.ball.vel
        obs2 = get_rel_obs(c2_fake, c1_fake, b_fake)
        return obs1, obs2

    def _apply_action(self, car, act):
        car.throttle = np.clip(act[0], -1, 1)
        car.turn = np.clip(act[1], -1, 1)
        car.boost_active = act[2] > 0
        car.drifting = act[3] > 0

# --- PPO Classes ---
class Memory:
    def __init__(self):
        self.actions, self.states, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []

    def clear(self):
        del self.actions[:]; del self.states[:]; del self.logprobs[:]
        del self.rewards[:]; del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.register_buffer('action_var', torch.full((action_dim,), action_std_init * action_std_init))
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def act(self, state):
        mu = self.actor(state)
        cov = torch.diag(self.action_var).unsqueeze(0)
        dist = torch.distributions.MultivariateNormal(mu, cov)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state, action):
        mu = self.actor(state)
        std = self.action_var.expand_as(mu)
        cov = torch.diag_embed(std)
        dist = torch.distributions.MultivariateNormal(mu, cov)
        return dist.log_prob(action), self.critic(state).squeeze(), dist.entropy()

class PPO:
    def __init__(self, state_dim, action_dim, action_std_init, device):
        self.device = device
        self.lr, self.gamma, self.eps_clip, self.K_epochs = LR, GAMMA, EPS_CLIP, K_EPOCHS
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    def select_action_batch(self, states_np, memories):
        """Batch inference across N parallel envs. states_np: (N, state_dim)"""
        states = torch.FloatTensor(states_np).to(self.device)
        with torch.no_grad():
            mu = self.policy_old.actor(states)
            cov = torch.diag(self.policy_old.action_var).unsqueeze(0).expand(len(states), -1, -1)
            dist = torch.distributions.MultivariateNormal(mu, cov)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
        for i, mem in enumerate(memories):
            mem.states.append(states[i].unsqueeze(0))
            mem.actions.append(actions[i].unsqueeze(0))
            mem.logprobs.append(logprobs[i].unsqueeze(0))
        return actions.cpu().numpy()

    def update(self, memories):
        """Merge N parallel env memories and run PPO update with AMP."""
        all_rewards, all_states, all_actions, all_logprobs = [], [], [], []

        for mem in memories:
            discounted_reward = 0
            mem_rewards = []
            for reward, is_terminal in zip(reversed(mem.rewards), reversed(mem.is_terminals)):
                if is_terminal: discounted_reward = 0
                discounted_reward = reward + self.gamma * discounted_reward
                mem_rewards.insert(0, discounted_reward)
            all_rewards.extend(mem_rewards)
            all_states.extend(mem.states)
            all_actions.extend(mem.actions)
            all_logprobs.extend(mem.logprobs)

        rewards = torch.tensor(all_rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.squeeze(torch.stack(all_states), 1).detach()
        old_actions = torch.squeeze(torch.stack(all_actions), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(all_logprobs), 1).detach()

        use_amp = self.device.type == 'cuda'
        for _ in range(self.K_epochs):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                ratios = torch.exp(logprobs - old_logprobs)
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            self.scaler.scale(loss.mean()).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.policy_old.load_state_dict(self.policy.state_dict())
        for mem in memories:
            mem.clear()

# --- MAIN LOOP ---
if __name__ == "__main__":
    if not os.path.exists("saved_agents"):
        os.makedirs("saved_agents")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Training on: {device}")
    print(f">>> {N_ENVS} parallel envs | {UPDATE_TIMESTEP * N_ENVS} samples/update | {K_EPOCHS} epochs")

    envs = [RocketLeagueEnv() for _ in range(N_ENVS)]
    memories = [Memory() for _ in range(N_ENVS)]
    main_agent = PPO(15, 4, ACTION_STD_INIT, device)
    opponent_policy = ActorCritic(15, 4, ACTION_STD_INIT).to(device)

    if os.path.exists("curriculum_brain.pth"):
        print(">>> Loading Saved Brain...")
        try:
            main_agent.policy.load_state_dict(torch.load("curriculum_brain.pth", map_location=device, weights_only=True))
            main_agent.policy_old.load_state_dict(main_agent.policy.state_dict())
            opponent_policy.load_state_dict(main_agent.policy.state_dict())
        except Exception as e:
            print(f">>> Error loading brain: {e}. Starting fresh.")

    step = 0
    curr_stage = START_STAGE
    stage0_hits, stage1_goals = 0, 0

    obs_pairs = [env.reset(curr_stage) for env in envs]
    obs1s = [p[0] for p in obs_pairs]
    obs2s = [p[1] for p in obs_pairs]
    opponent_modes = ["SELF"] * N_ENVS

    def pick_opponent_mode():
        if curr_stage < 2:
            return "DUMB"
        rand = np.random.random()
        if rand < 0.2: return "DUMB"
        if rand < 0.4: return "HISTORY"
        return "SELF"

    dumb_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    try:
        while step < MAX_TIMESTEPS:
            step += N_ENVS

            # Opponent selection at episode start for each env
            for i, env in enumerate(envs):
                if env.episode_steps == 0:
                    opponent_modes[i] = pick_opponent_mode()
                    if opponent_modes[i] == "HISTORY":
                        files = glob.glob("saved_agents/*.pth")
                        if files:
                            try:
                                opponent_policy.load_state_dict(
                                    torch.load(random.choice(files), map_location=device, weights_only=True)
                                )
                            except Exception:
                                opponent_modes[i] = "SELF"
                        else:
                            opponent_modes[i] = "SELF"
                    if opponent_modes[i] == "SELF":
                        opponent_policy.load_state_dict(main_agent.policy.state_dict())

            # Batch car1 inference
            obs1_batch = np.stack(obs1s)                                      # (N, 15)
            a1_batch = main_agent.select_action_batch(obs1_batch, memories)   # (N, 4)

            # Batch car2 inference — run all through opponent policy, override dumb envs after
            obs2_tensor = torch.FloatTensor(np.stack(obs2s)).to(device)
            with torch.no_grad():
                a2_smart, _ = opponent_policy.act(obs2_tensor)
            a2_smart_np = a2_smart.cpu().numpy()                              # (N, 4)
            is_dumb = np.array([m == "DUMB" for m in opponent_modes])
            a2_batch = np.where(is_dumb[:, None], dumb_action[None, :], a2_smart_np)

            # Step all envs
            for i, env in enumerate(envs):
                (next_obs1, next_obs2), (r1, r2), done, info = env.step(a1_batch[i], a2_batch[i])
                memories[i].rewards.append(r1)
                memories[i].is_terminals.append(done)
                obs1s[i], obs2s[i] = next_obs1, next_obs2

                if curr_stage == 0 and info['hit_ball']:
                    stage0_hits += 1
                    if stage0_hits % 5 == 0:
                        print(f"Hit Ball: {stage0_hits}/1000")
                    if stage0_hits >= 1000:
                        curr_stage = 1; print(">>> PROMOTED TO STAGE 1!"); done = True

                elif curr_stage == 1 and done and info['goal']:
                    stage1_goals += 1
                    print(f"Goal! {stage1_goals}/1000")
                    if stage1_goals >= 1000:
                        curr_stage = 2; print(">>> PROMOTED TO STAGE 2 (LEAGUE PLAY)!"); done = True

                if done:
                    obs1s[i], obs2s[i] = env.reset(curr_stage)

            # Logging
            if step % (N_ENVS * 1000) == 0:
                print(f"Step {step} | Stage {curr_stage}")

            # PPO update when enough samples collected across all envs
            total_samples = sum(len(m.rewards) for m in memories)
            if total_samples >= UPDATE_TIMESTEP * N_ENVS:
                main_agent.update(memories)
                torch.save(main_agent.policy.state_dict(), "curriculum_brain.pth")
                print(f">>> Updated Brain at step {step}.")

            # Snapshot
            if step % SNAPSHOT_INTERVAL == 0:
                snap_name = f"saved_agents/agent_{step}.pth"
                torch.save(main_agent.policy.state_dict(), snap_name)
                print(f">>> SAVED SNAPSHOT: {snap_name}")

    except KeyboardInterrupt:
        print("\nSaving before exit...")
        torch.save(main_agent.policy.state_dict(), "curriculum_brain.pth")
