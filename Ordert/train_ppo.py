import torch
import torch.nn as nn
import numpy as np
import pygame
import os
from collections import deque

# --- Configuration ---
SCREEN_W, SCREEN_H = 1200, 1000
FPS = 60
UPDATE_TIMESTEP = 2000
LR = 0.0003
GAMMA = 0.99
K_EPOCHS = 10
EPS_CLIP = 0.2
ACTION_STD_INIT = 0.6

HUMAN_MODEL_LR = 0.001
HUMAN_MODEL_BATCH = 256
HUMAN_MEMORY_SIZE = 5000

# --- Physics Constants ---
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
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def rotate_vector(v: np.ndarray, angle_degrees: float) -> np.ndarray:
    rad = np.radians(angle_degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])

CORNER_NORMALS = [
    normalize(np.array([0.707,  0.707])),
    normalize(np.array([-0.707, 0.707])),
    normalize(np.array([-0.707, -0.707])),
    normalize(np.array([0.707,  -0.707])),
]
CORNER_POINTS = [
    np.array([OFFSET_X + CORNER_SIZE,          OFFSET_Y]),
    np.array([OFFSET_X + ARENA_W - CORNER_SIZE, OFFSET_Y]),
    np.array([OFFSET_X + ARENA_W - CORNER_SIZE, OFFSET_Y + ARENA_H]),
    np.array([OFFSET_X + CORNER_SIZE,           OFFSET_Y + ARENA_H]),
]

def get_rel_obs(my_car, opp_car, ball) -> np.ndarray:
    obs = [
        my_car.pos[0] / SCREEN_W, my_car.pos[1] / SCREEN_H,
        my_car.vel[0] / 10,       my_car.vel[1] / 10,
        np.cos(np.radians(my_car.angle)), np.sin(np.radians(my_car.angle)),
        my_car.boost / 100.0,
        ball.pos[0] / SCREEN_W,   ball.pos[1] / SCREEN_H,
        ball.vel[0] / 10,         ball.vel[1] / 10,
        opp_car.pos[0] / SCREEN_W, opp_car.pos[1] / SCREEN_H,
        opp_car.vel[0] / 10,      opp_car.vel[1] / 10,
    ]
    return np.array(obs, dtype=np.float32)

# --- Physics Classes ---
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
        current_accel, current_max = ACCEL, MAX_SPEED
        if self.boost_active and self.boost > 0:
            current_accel, current_max = BOOST_ACCEL, BOOST_MAX_SPEED
            self.boost -= 0.15
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
        if np.linalg.norm(self.vel) > current_max:
            self.vel = (self.vel / np.linalg.norm(self.vel)) * current_max

# --- Physics Solver ---
def bounce(obj, normal, elasticity):
    vel_along_normal = np.dot(obj.vel, normal)
    if vel_along_normal < 0:
        obj.vel += -(1 + elasticity) * vel_along_normal * normal

def resolve_arena_collisions(obj):
    r = obj.radius
    elasticity = BALL_ELASTICITY if isinstance(obj, Ball) else CAR_WALL_ELASTICITY

    goal_top = SCREEN_H / 2 - GOAL_SIZE / 2
    goal_bot = SCREEN_H / 2 + GOAL_SIZE / 2

    for post in [
        np.array([OFFSET_X,          goal_top]),
        np.array([OFFSET_X,          goal_bot]),
        np.array([OFFSET_X + ARENA_W, goal_top]),
        np.array([OFFSET_X + ARENA_W, goal_bot]),
    ]:
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
        diff = obj.pos - CORNER_POINTS[i]
        dist = np.dot(diff, CORNER_NORMALS[i])
        if dist < r:
            obj.pos += CORNER_NORMALS[i] * (r - dist)
            bounce(obj, CORNER_NORMALS[i], elasticity)

def resolve_car_ball(car, ball):
    local_ball = rotate_vector(ball.pos - car.pos, -car.angle)
    hw, hh = car.width / 2, car.height / 2
    cx = max(-hw, min(local_ball[0], hw))
    cy = max(-hh, min(local_ball[1], hh))
    dist_vec = local_ball - np.array([cx, cy])
    dist_sq  = np.dot(dist_vec, dist_vec)

    if dist_sq < ball.radius ** 2:
        if dist_sq == 0:
            dist_vec = np.array([ball.radius, 0.0])
            dist_sq  = ball.radius ** 2
        dist         = np.sqrt(dist_sq)
        normal_world = rotate_vector(dist_vec / dist, car.angle)
        overlap      = ball.radius - dist

        total_inv = car.inv_mass + ball.inv_mass
        push      = normal_world * overlap / total_inv
        car.pos  -= push * car.inv_mass
        ball.pos += push * ball.inv_mass

        rel_vel = ball.vel - car.vel
        vel_n   = np.dot(rel_vel, normal_world)
        if vel_n > 0:
            return False
        j       = -(1 + BALL_ELASTICITY) * vel_n / total_inv
        impulse = j * normal_world
        ball.vel += impulse * ball.inv_mass
        car.vel  -= impulse * car.inv_mass
        return True
    return False

def resolve_car_car(c1, c2):
    diff = c1.pos - c2.pos
    dist = np.linalg.norm(diff)
    if dist < 40.0:
        if dist == 0:
            diff = np.array([1.0, 0.0]); dist = 1.0
        normal  = diff / dist
        overlap = 40.0 - dist
        c1.pos += normal * overlap * 0.5
        c2.pos -= normal * overlap * 0.5
        vel_n = np.dot(c1.vel - c2.vel, normal)
        if vel_n > 0: return
        j       = -(1 + 0.5) * vel_n / (c1.inv_mass + c2.inv_mass)
        impulse = j * normal
        c1.vel += impulse * c1.inv_mass
        c2.vel -= impulse * c2.inv_mass

# --- PPO Network ---
class Memory:
    def __init__(self):
        self.actions, self.states, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []
    def clear(self):
        del self.actions[:]; del self.states[:]; del self.logprobs[:]
        del self.rewards[:]; del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super().__init__()
        # Registered buffer so action_var is saved/loaded with state_dict
        self.register_buffer('action_var', torch.full((action_dim,), action_std_init ** 2))
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128),       nn.Tanh(),
            nn.Linear(128, action_dim), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 128),       nn.Tanh(),
            nn.Linear(128, 1),
        )

    def act(self, state):
        mu   = self.actor(state)
        dist = torch.distributions.MultivariateNormal(mu, torch.diag(self.action_var))
        # Clamp sampled action so stored action matches what the environment receives
        action = dist.sample().clamp(-1, 1)
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state, action):
        mu   = self.actor(state)
        cov  = torch.diag_embed(self.action_var.expand_as(mu))
        dist = torch.distributions.MultivariateNormal(mu, cov)
        return dist.log_prob(action), self.critic(state).squeeze(), dist.entropy()

class PPO:
    def __init__(self, state_dim, action_dim, action_std_init):
        self.gamma, self.eps_clip, self.K_epochs = GAMMA, EPS_CLIP, K_EPOCHS
        self.policy     = ActorCritic(state_dim, action_dim, action_std_init)
        self.optimizer  = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state  = torch.FloatTensor(state.reshape(1, -1))
        action, logprob = self.policy_old.act(state)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        return action.detach().numpy().flatten()

    def update(self, memory):
        rewards = []
        discounted = 0.0
        for reward, terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if terminal: discounted = 0.0
            discounted = reward + self.gamma * discounted
            rewards.insert(0, discounted)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states   = torch.squeeze(torch.stack(memory.states),   1).detach()
        old_actions  = torch.squeeze(torch.stack(memory.actions),  1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, entropy = self.policy.evaluate(old_states, old_actions)
            ratios     = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss  = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear()

# --- Human Behavior Tracking ---

class HumanMemory:
    """Ring buffer of (obs, action) pairs collected from the human player."""
    def __init__(self, maxlen: int = HUMAN_MEMORY_SIZE):
        self._states:  deque = deque(maxlen=maxlen)
        self._actions: deque = deque(maxlen=maxlen)

    def add(self, state: np.ndarray, action: np.ndarray):
        self._states.append(state.copy())
        self._actions.append(action.copy())

    def sample(self, n: int):
        n    = min(n, len(self))
        idxs = np.random.choice(len(self), n, replace=False)
        s = torch.FloatTensor(np.array([self._states[i]  for i in idxs]))
        a = torch.FloatTensor(np.array([self._actions[i] for i in idxs]))
        return s, a

    def __len__(self) -> int:
        return len(self._states)


class HumanModel(nn.Module):
    """Predicts the human's next action from their observation (behavioral cloning)."""
    def __init__(self, state_dim: int = 15, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),        nn.Tanh(),
            nn.Linear(64, action_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()


class HumanProfile:
    """Rolling behavioral statistics for display and adaptation reward shaping."""
    def __init__(self, window: int = 300):
        self._throttles: deque = deque([0.0], maxlen=window)
        self._boosts:    deque = deque([0.0], maxlen=window)
        self._drifts:    deque = deque([0.0], maxlen=window)
        self._x_pos:     deque = deque([0.5], maxlen=window)

    def update(self, obs1: np.ndarray, action: np.ndarray):
        self._x_pos.append(float(obs1[0]))           # my_x/W in car1 frame
        self._throttles.append(abs(float(action[0])))
        self._boosts.append(float(action[2] > 0))
        self._drifts.append(float(action[3] > 0))

    @property
    def aggression(self) -> float:
        return float(np.mean(self._throttles))

    @property
    def boost_rate(self) -> float:
        return float(np.mean(self._boosts))

    @property
    def drift_rate(self) -> float:
        return float(np.mean(self._drifts))

    @property
    def position_label(self) -> str:
        x = float(np.mean(self._x_pos))
        if x > 0.60: return "Forward"
        if x < 0.40: return "Back"
        return "Mid"


def train_human_model(
    model: HumanModel,
    optimizer: torch.optim.Optimizer,
    memory: HumanMemory,
) -> float:
    """One minibatch of behavioral cloning. Returns MSE loss, or nan if insufficient data."""
    if len(memory) < HUMAN_MODEL_BATCH:
        return float('nan')
    states, actions = memory.sample(HUMAN_MODEL_BATCH)
    loss = nn.functional.mse_loss(model(states), actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def compute_adaptation_reward(
    human_model: HumanModel,
    human_profile: HumanProfile,
    env: 'RocketLeagueEnv',
    model_ready: bool,
) -> float:
    """
    Small positional bonus that shapes the AI to counter the human's observed tendencies.
    Profile-based bonuses are always active; model-based bonuses activate once trained.
    Hard-capped at 0.01/step so it can't swamp the main reward signal.
    """
    bonus      = 0.0
    aggression = human_profile.aggression
    ball_x     = env.ball.pos[0]
    car2_x     = env.car2.pos[0]

    # Aggressive human → reward AI for staying between ball and its own goal (right side)
    if aggression > 0.55 and car2_x > ball_x:
        bonus += 0.005 * aggression

    # Passive human → reward AI for pressing close to ball
    if aggression < 0.30:
        if np.linalg.norm(env.car2.pos - env.ball.pos) < 250:
            bonus += 0.004

    if model_ready:
        pred          = human_model.predict(get_rel_obs(env.car1, env.car2, env.ball))
        pred_throttle = float(pred[0])   # +1 = human rushing toward AI goal
        pred_boost    = float(pred[2])

        if pred_throttle > 0.5 and car2_x > ball_x:
            bonus += 0.004 * pred_throttle

        # Human predicted to boost hard → conserving AI boost is valuable for a counter
        if pred_boost > 0.5 and env.car2.boost > 60.0:
            bonus += 0.002

    return min(bonus, 0.01)


# --- Game & Render Logic ---
class RocketLeagueEnv:
    def __init__(self):
        self.car1 = Car(250, SCREEN_H / 2)
        self.car2 = Car(SCREEN_W - 250, SCREEN_H / 2)
        self.ball = Ball(SCREEN_W / 2, SCREEN_H / 2)
        self.episode_steps    = 0
        self.wall_contact_steps = 0

    def reset(self):
        self.episode_steps    = 0
        self.wall_contact_steps = 0
        self.ball.pos = np.array([SCREEN_W / 2, SCREEN_H / 2]); self.ball.vel[:] = 0
        self.car1.pos = np.array([OFFSET_X + 100,            SCREEN_H / 2])
        self.car1.vel[:] = 0; self.car1.angle = 0.0;   self.car1.boost = 100
        self.car2.pos = np.array([OFFSET_X + ARENA_W - 100, SCREEN_H / 2])
        self.car2.vel[:] = 0; self.car2.angle = 180.0; self.car2.boost = 100
        return self._get_obs_pair()

    def _get_obs_pair(self):
        obs1 = get_rel_obs(self.car1, self.car2, self.ball)
        c2_fake       = Car(SCREEN_W - self.car2.pos[0], SCREEN_H - self.car2.pos[1], (self.car2.angle + 180) % 360)
        c2_fake.vel   = -self.car2.vel; c2_fake.boost = self.car2.boost
        c1_fake       = Car(SCREEN_W - self.car1.pos[0], SCREEN_H - self.car1.pos[1])
        c1_fake.vel   = -self.car1.vel
        b_fake        = Ball(SCREEN_W - self.ball.pos[0], SCREEN_H - self.ball.pos[1])
        b_fake.vel    = -self.ball.vel
        obs2 = get_rel_obs(c2_fake, c1_fake, b_fake)
        return obs1, obs2

    def _apply_action(self, car, act):
        car.throttle    = float(act[0])
        car.turn        = float(act[1])
        car.boost_active = act[2] > 0
        car.drifting    = act[3] > 0

    def get_human_action(self) -> np.ndarray:
        keys     = pygame.key.get_pressed()
        throttle = (1.0 if keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_s] else 0.0)
        turn     = (1.0 if keys[pygame.K_d] else 0.0) - (1.0 if keys[pygame.K_a] else 0.0)
        boost    = 1.0 if keys[pygame.K_LSHIFT] else 0.0
        drift    = 1.0 if keys[pygame.K_SPACE]  else 0.0
        return np.array([throttle, turn, boost, drift], dtype=np.float32)

    def step(self, action1, action2):
        self.episode_steps += 1
        self._apply_action(self.car1, action1)
        self._apply_action(self.car2, action2)

        dt           = 1.0 / PHYSICS_SUBSTEPS
        ai_hit_ball  = False
        human_hit_ball = False

        for _ in range(PHYSICS_SUBSTEPS):
            self.car1.update_controls()
            self.car2.update_controls()
            self.car1.pos += self.car1.vel * dt
            self.car2.pos += self.car2.vel * dt
            self.ball.pos += self.ball.vel * dt
            self.ball.vel *= BALL_DRAG
            resolve_arena_collisions(self.car1)
            resolve_arena_collisions(self.car2)
            resolve_arena_collisions(self.ball)
            if resolve_car_ball(self.car2, self.ball): ai_hit_ball    = True
            if resolve_car_ball(self.car1, self.ball): human_hit_ball = True
            resolve_car_car(self.car1, self.car2)

        c2 = self.car2
        on_wall = (
            c2.pos[0] - c2.radius <= OFFSET_X or
            c2.pos[0] + c2.radius >= OFFSET_X + ARENA_W or
            c2.pos[1] - c2.radius <= OFFSET_Y or
            c2.pos[1] + c2.radius >= OFFSET_Y + ARENA_H
        )
        self.wall_contact_steps = self.wall_contact_steps + 1 if on_wall else 0

        done = False
        r1, r2 = 0.0, 0.0
        info = {'ai_hit': ai_hit_ball, 'goal': False}

        # Goal detection: ball edge + 10px buffer, matching main_train.py
        ball_r = self.ball.radius
        if self.ball.pos[0] - ball_r > OFFSET_X + ARENA_W + 10:
            if abs(self.ball.pos[1] - SCREEN_H / 2) < GOAL_SIZE / 2:
                r1 += 30.0; r2 -= 30.0; done = True; info['goal'] = True
        elif self.ball.pos[0] + ball_r < OFFSET_X - 10:
            if abs(self.ball.pos[1] - SCREEN_H / 2) < GOAL_SIZE / 2:
                r1 -= 30.0; r2 += 30.0; done = True; info['goal'] = True

        # AI (car2) reward shaping — mirrors main_train.py car1 rewards
        d2 = np.linalg.norm(self.car2.pos - self.ball.pos)
        r2 -= 0.001
        if d2 < 500:
            r2 += (500 - d2) * 0.0001

        vec_to_ball = (self.ball.pos - self.car2.pos) / (d2 + 1e-5)
        if np.dot(self.car2.vel, vec_to_ball) > 2.0:
            r2 += np.dot(self.car2.vel, vec_to_ball) * 0.001

        if self.car2.boost > 30.0:
            r2 += 0.005

        if self.wall_contact_steps > 30:
            r2 -= 0.05

        ai_goal_center = np.array([OFFSET_X + ARENA_W, SCREEN_H / 2])
        ball_toward_ai = np.dot(self.ball.vel, normalize(ai_goal_center - self.ball.pos))
        if ball_toward_ai > 0:
            r2 -= ball_toward_ai * 0.05

        if human_hit_ball:
            r2 -= 0.5

        if ai_hit_ball:
            r2 += 1.0
            left_goal = np.array([OFFSET_X, SCREEN_H / 2])
            shoot_vel = np.dot(self.ball.vel, normalize(left_goal - self.ball.pos))
            if shoot_vel > 0:
                r2 += shoot_vel * 1.0
            if np.linalg.norm(self.ball.pos - ai_goal_center) < 300:
                r2 += 5.0   # save bonus

        if self.episode_steps > 1500:
            done = True

        return self._get_obs_pair(), (r1, r2), done, info


def draw_rotated_rect(screen, color, pos, angle, width, height):
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (0, 0, width, height))
    rot  = pygame.transform.rotate(surf, -angle)
    rect = rot.get_rect(center=(int(pos[0]), int(pos[1])))
    screen.blit(rot, rect.topleft)
    rad   = np.radians(angle)
    front = pos + np.array([np.cos(rad), np.sin(rad)]) * (width / 2)
    pygame.draw.line(screen, (255, 255, 255),
                     (int(pos[0]), int(pos[1])), (int(front[0]), int(front[1])), 2)

def _bar(screen, x, y, w, h, frac, color):
    """Render a filled progress bar."""
    pygame.draw.rect(screen, (40, 40, 55), (x, y, w, h))
    pygame.draw.rect(screen, color, (x, y, int(w * max(0.0, min(1.0, frac))), h))
    pygame.draw.rect(screen, (80, 80, 100), (x, y, w, h), 1)


# --- Main ---
if __name__ == "__main__":
    pygame.init()
    screen   = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Human (Blue) vs AI (Red) — Adaptive Mode")
    clock    = pygame.time.Clock()
    font     = pygame.font.Font(None, 28)
    font_sm  = pygame.font.Font(None, 22)
    font_hdr = pygame.font.Font(None, 24)

    env = RocketLeagueEnv()
    ai_agent = PPO(15, 4, ACTION_STD_INIT)

    human_mem     = HumanMemory()
    human_model   = HumanModel()
    human_opt     = torch.optim.Adam(human_model.parameters(), lr=HUMAN_MODEL_LR)
    human_profile = HumanProfile()
    human_model_ready = False
    human_model_loss  = float('nan')

    if os.path.exists("curriculum_brain.pth"):
        print(">>> Loading Saved Brain...")
        ai_agent.policy.load_state_dict(
            torch.load("curriculum_brain.pth", weights_only=True)
        )
        ai_agent.policy_old.load_state_dict(ai_agent.policy.state_dict())
    else:
        print(">>> No brain found. AI starts untrained.")

    memory = Memory()
    step   = 0
    score_human, score_ai = 0, 0
    running  = True
    obs1, obs2 = env.reset()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            step += 1
            action_human = env.get_human_action()
            action_ai    = ai_agent.select_action(obs2, memory)

            # Collect human behavior before executing the step
            human_mem.add(obs1, action_human)
            human_profile.update(obs1, action_human)

            (next_obs1, next_obs2), (r1, r2), done, info = env.step(action_human, action_ai)

            r2 += compute_adaptation_reward(human_model, human_profile, env, human_model_ready)

            memory.rewards.append(r2)
            memory.is_terminals.append(done)
            obs1, obs2 = next_obs1, next_obs2

            if info['goal']:
                if r1 > 0: score_human += 1
                else:       score_ai    += 1

            # --- Render ---
            screen.fill((20, 20, 30))

            pygame.draw.rect(screen, (50, 50, 60), (OFFSET_X, OFFSET_Y, ARENA_W, ARENA_H), 2)
            pygame.draw.rect(screen, (0, 100, 255),
                             (OFFSET_X - GOAL_DEPTH, SCREEN_H // 2 - GOAL_SIZE // 2, GOAL_DEPTH, GOAL_SIZE), 2)
            pygame.draw.rect(screen, (255, 50, 50),
                             (OFFSET_X + ARENA_W,    SCREEN_H // 2 - GOAL_SIZE // 2, GOAL_DEPTH, GOAL_SIZE), 2)

            draw_rotated_rect(screen, (50, 150, 255), env.car1.pos, env.car1.angle, 40, 24)
            draw_rotated_rect(screen, (255, 80,  80), env.car2.pos, env.car2.angle, 40, 24)
            pygame.draw.circle(screen, (240, 240, 240),
                               (int(env.ball.pos[0]), int(env.ball.pos[1])), 22)

            # --- HUD ---
            hx = 12

            score_s = font.render(f"You  {score_human}  —  {score_ai}  AI", True, (220, 220, 220))
            screen.blit(score_s, (hx, 10))

            step_s = font_sm.render(
                f"Step {step}   AI reward: {r2:+.3f}   Boost: {int(env.car2.boost)}",
                True, (140, 140, 180))
            screen.blit(step_s, (hx, 36))

            # Human profile panel
            px, py, pw, ph = hx, 62, 234, 114
            pygame.draw.rect(screen, (26, 26, 40), (px, py, pw, ph), border_radius=5)
            pygame.draw.rect(screen, (55, 55, 85), (px, py, pw, ph), 1, border_radius=5)

            screen.blit(font_hdr.render("HUMAN PROFILE", True, (120, 120, 210)), (px + 6, py + 5))

            bar_x, bar_w = px + 94, pw - 102
            row = py + 24
            for label, val, col in [
                ("Aggression", human_profile.aggression, (255, 110, 70)),
                ("Boost use",  human_profile.boost_rate, (70,  190, 255)),
                ("Drift rate", human_profile.drift_rate, (160, 255, 110)),
            ]:
                screen.blit(font_sm.render(f"{label}:", True, (160, 160, 185)), (px + 6, row + 1))
                _bar(screen, bar_x, row + 3, bar_w, 10, val, col)
                row += 19

            screen.blit(
                font_sm.render(f"Position:   {human_profile.position_label}", True, (160, 215, 160)),
                (px + 6, row + 2))
            row += 20

            if human_model_ready:
                st_col = (90, 210, 110)
                st_txt = f"Adapting  (loss={human_model_loss:.3f})"
            elif len(human_mem) > 0:
                pct    = int(len(human_mem) / HUMAN_MODEL_BATCH * 100)
                st_col = (200, 175, 70)
                st_txt = f"Observing...  {min(pct, 100)}%"
            else:
                st_col = (130, 130, 130)
                st_txt = "Watching you play..."
            screen.blit(font_sm.render(st_txt, True, st_col), (px + 6, row + 2))

            pygame.display.flip()
            clock.tick(FPS)

            if step % UPDATE_TIMESTEP == 0:
                print(">>> Updating AI brain...")
                ai_agent.update(memory)
                torch.save(ai_agent.policy.state_dict(), "curriculum_brain.pth")

                human_model_loss = train_human_model(human_model, human_opt, human_mem)
                if not np.isnan(human_model_loss):
                    human_model_ready = True
                    print(f">>> Human model updated  loss={human_model_loss:.4f}")

            if done:
                obs1, obs2 = env.reset()

    except KeyboardInterrupt:
        print("Saving brain...")
        torch.save(ai_agent.policy.state_dict(), "curriculum_brain.pth")

    pygame.quit()
