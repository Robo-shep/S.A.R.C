import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import pygame
import math
import os
import sys

# --- Configuration ---
SCREEN_W, SCREEN_H = 1200, 1000
FPS = 60
UPDATE_TIMESTEP = 2000  # Train AI every ~30 seconds of gameplay
LR = 0.0003
GAMMA = 0.99
K_EPOCHS = 10
EPS_CLIP = 0.2
ACTION_STD_INIT = 0.6

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
def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def rotate_vector(v, angle_degrees):
    rad = np.radians(angle_degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])

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
        elif self.boost < MAX_BOOST: self.boost += 0.03
            
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
        if np.linalg.norm(self.vel) > current_max: self.vel = (self.vel / np.linalg.norm(self.vel)) * current_max

# --- Physics Solver ---
def bounce(obj, normal, elasticity):
    vel_along_normal = np.dot(obj.vel, normal)
    if vel_along_normal < 0: 
        j = -(1 + elasticity) * vel_along_normal
        impulse = j * normal
        obj.vel += impulse

def resolve_arena_collisions(obj):
    r = obj.radius
    elasticity = BALL_ELASTICITY if isinstance(obj, Ball) else CAR_WALL_ELASTICITY

    goal_top = SCREEN_H/2 - GOAL_SIZE/2
    goal_bot = SCREEN_H/2 + GOAL_SIZE/2
    
    tl_post = np.array([OFFSET_X, goal_top])
    bl_post = np.array([OFFSET_X, goal_bot])
    tr_post = np.array([OFFSET_X + ARENA_W, goal_top])
    br_post = np.array([OFFSET_X + ARENA_W, goal_bot])

    for post in [tl_post, bl_post, tr_post, br_post]:
        diff = obj.pos - post
        dist_sq = np.dot(diff, diff)
        if dist_sq < r*r:
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
    
    hw, hh = car.width/2, car.height/2
    cx = max(-hw, min(local_ball[0], hw))
    cy = max(-hh, min(local_ball[1], hh))
    
    closest_local = np.array([cx, cy])
    dist_vec = local_ball - closest_local
    dist_sq = np.dot(dist_vec, dist_vec)
    
    if dist_sq < ball.radius * ball.radius:
        if dist_sq == 0: 
            dist_vec = np.array([ball.radius, 0.0])
            dist_sq = ball.radius**2
            
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
        if vel_along_normal > 0: return 
        
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
            dist=1.0
        normal = dist_vec / dist
        
        c1.pos += normal * (overlap * 0.5)
        c2.pos -= normal * (overlap * 0.5)
        
        rel_vel = c1.vel - c2.vel
        vel_along = np.dot(rel_vel, normal)
        if vel_along > 0: return
        
        j = -(1 + 0.5) * vel_along
        j /= (c1.inv_mass + c2.inv_mass)
        impulse = j * normal
        c1.vel += impulse * c1.inv_mass
        c2.vel -= impulse * c2.inv_mass


# --- PPO Network ---
class Memory:
    def __init__(self): self.actions, self.states, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []
    def clear(self): del self.actions[:]; del self.states[:]; del self.logprobs[:]; del self.rewards[:]; del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        self.actor = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, action_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    def act(self, state):
        mu = self.actor(state); std = torch.diag(self.action_var).unsqueeze(0)
        dist = torch.distributions.MultivariateNormal(mu, std); action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()
    def evaluate(self, state, action):
        mu = self.actor(state); std = self.action_var.expand_as(mu); cov = torch.diag_embed(std)
        dist = torch.distributions.MultivariateNormal(mu, cov)
        return dist.log_prob(action), self.critic(state).squeeze(), dist.entropy()

class PPO:
    def __init__(self, state_dim, action_dim, action_std_init):
        self.lr, self.gamma, self.eps_clip, self.K_epochs = LR, GAMMA, EPS_CLIP, K_EPOCHS
        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1))
        action, action_logprob = self.policy_old.act(state)
        memory.states.append(state); memory.actions.append(action); memory.logprobs.append(action_logprob)
        return action.detach().numpy().flatten()
    def update(self, memory):
        rewards = []; discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.squeeze(torch.stack(memory.states), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).detach()
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs); advantages = rewards - state_values.detach()
            surr1 = ratios * advantages; surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad(); loss.mean().backward(); self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict()); memory.clear()

# --- Game & Render Logic ---
class RocketLeagueEnv:
    def __init__(self):
        self.car1 = Car(250, SCREEN_H/2)
        self.car2 = Car(SCREEN_W - 250, SCREEN_H/2)
        self.ball = Ball(SCREEN_W/2, SCREEN_H/2)
        self.episode_steps = 0

    def reset(self):
        self.episode_steps = 0
        self.ball.pos = np.array([SCREEN_W/2, SCREEN_H/2]); self.ball.vel[:] = 0
        self.car1.pos = np.array([OFFSET_X + 100, SCREEN_H/2]); self.car1.vel[:] = 0; self.car1.angle = 0.0; self.car1.boost = 100
        self.car2.pos = np.array([OFFSET_X + ARENA_W - 100, SCREEN_H/2]); self.car2.vel[:] = 0; self.car2.angle = 180.0; self.car2.boost = 100
        return self._get_obs_pair()

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
        car.throttle = np.clip(act[0], -1, 1); car.turn = np.clip(act[1], -1, 1)
        car.boost_active = act[2] > 0; car.drifting = act[3] > 0

    def get_human_action(self):
        keys = pygame.key.get_pressed()
        throttle = (1.0 if keys[pygame.K_w] else 0.0) - (1.0 if keys[pygame.K_s] else 0.0)
        turn = (1.0 if keys[pygame.K_d] else 0.0) - (1.0 if keys[pygame.K_a] else 0.0)
        boost = 1.0 if keys[pygame.K_LSHIFT] else 0.0
        drift = 1.0 if keys[pygame.K_SPACE] else 0.0
        return np.array([throttle, turn, boost, drift], dtype=np.float32)

    def step(self, action1, action2):
        self.episode_steps += 1
        self._apply_action(self.car1, action1); self._apply_action(self.car2, action2)
        
        dt = 1.0 / PHYSICS_SUBSTEPS; hit_ball = False
        for _ in range(PHYSICS_SUBSTEPS):
            if self.car1.pos[0] > -500: self.car1.update_controls()
            if self.car2.pos[0] > -500: self.car2.update_controls()
            self.car1.pos += self.car1.vel * dt; self.car2.pos += self.car2.vel * dt
            self.ball.pos += self.ball.vel * dt; self.ball.vel *= BALL_DRAG
            resolve_arena_collisions(self.car1); resolve_arena_collisions(self.car2); resolve_arena_collisions(self.ball)
            if resolve_car_ball(self.car1, self.ball): hit_ball = True
            if resolve_car_ball(self.car2, self.ball): hit_ball = True
            resolve_car_car(self.car1, self.car2)

        done = False; r1, r2 = 0.0, 0.0
        info = {'goal': False}
        
        # Goals
        if self.ball.pos[0] > OFFSET_X + ARENA_W and abs(self.ball.pos[1] - SCREEN_H/2) < 100:
            r1 += 10.0; r2 -= 10.0; done = True; info['goal'] = True
        elif self.ball.pos[0] < OFFSET_X and abs(self.ball.pos[1] - SCREEN_H/2) < 100:
            r1 -= 10.0; r2 += 10.0; done = True; info['goal'] = True

        # AI Rewards (Car 2)
        d2 = np.linalg.norm(self.car2.pos - self.ball.pos)
        r2 -= 0.001
        if d2 < 500: r2 += (500 - d2) * 0.0001
        
        if self.episode_steps > 1200: done = True # 20 second timeout

        return self._get_obs_pair(), (r1, r2), done, info

def draw_rotated_rect(screen, color, pos, angle, width, height):
    car_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(car_surf, color, (0, 0, width, height))
    rotated_surf = pygame.transform.rotate(car_surf, -angle)
    rect = rotated_surf.get_rect(center=(pos[0], pos[1]))
    screen.blit(rotated_surf, rect.topleft)
    rad = np.radians(angle)
    front = pos + np.array([np.cos(rad), np.sin(rad)]) * (width/2)
    pygame.draw.line(screen, (255, 255, 255), pos, front, 2)

# --- Main ---
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Rocket League: Human (Blue) vs AI (Red)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    env = RocketLeagueEnv()
    ai_agent = PPO(15, 4, ACTION_STD_INIT)
    
    if os.path.exists("curriculum_brain.pth"):
        print(">>> Loading Saved Brain...")
        ai_agent.policy.load_state_dict(torch.load("curriculum_brain.pth"))
        ai_agent.policy_old.load_state_dict(ai_agent.policy.state_dict())
    else:
        print(">>> No brain found. AI will be untaught.")

    memory = Memory()
    step = 0
    running = True
    obs1, obs2 = env.reset()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            
            step += 1
            action_human = env.get_human_action()
            action_ai = ai_agent.select_action(obs2, memory)
            
            (next_obs1, next_obs2), (r1, r2), done, info = env.step(action_human, action_ai)
            memory.rewards.append(r2); memory.is_terminals.append(done)
            obs1, obs2 = next_obs1, next_obs2
            
            # Render
            screen.fill((20, 20, 30))
            pygame.draw.rect(screen, (50, 50, 60), (OFFSET_X, OFFSET_Y, ARENA_W, ARENA_H), 2)
            pygame.draw.rect(screen, (0, 100, 255), (OFFSET_X-GOAL_DEPTH, SCREEN_H/2-GOAL_SIZE/2, GOAL_DEPTH, GOAL_SIZE), 2)
            pygame.draw.rect(screen, (255, 50, 50), (OFFSET_X+ARENA_W, SCREEN_H/2-GOAL_SIZE/2, GOAL_DEPTH, GOAL_SIZE), 2)
            
            draw_rotated_rect(screen, (50, 150, 255), env.car1.pos, env.car1.angle, 40, 24)
            draw_rotated_rect(screen, (255, 80, 80), env.car2.pos, env.car2.angle, 40, 24)
            pygame.draw.circle(screen, (240, 240, 240), (int(env.ball.pos[0]), int(env.ball.pos[1])), 22)
            
            t = font.render(f"AI Reward: {r2:.3f}", True, (200, 200, 200))
            screen.blit(t, (20, 20))
            pygame.display.flip()
            clock.tick(FPS)
            
            if step % UPDATE_TIMESTEP == 0:
                print(">>> Updating Brain...")
                ai_agent.update(memory)
                torch.save(ai_agent.policy.state_dict(), "curriculum_brain.pth")
            
            if done: obs1, obs2 = env.reset()

    except KeyboardInterrupt:
        print("Saving...")
        torch.save(ai_agent.policy.state_dict(), "curriculum_brain.pth")
    
    pygame.quit()