import pygame
import torch
import torch.nn as nn
import numpy as np
import math
import sys
import os

# ==========================================
# --- CONFIGURATION & CONSTANTS (Synced) ---
# ==========================================
SCREEN_W, SCREEN_H = 1200, 1000
TARGET_FPS = 60
PHYSICS_SUBSTEPS = 8 

# Colors
BG_COLOR = (20, 20, 30)
WALL_COLOR = (50, 50, 70)
P1_COLOR = (50, 150, 255)  # Blue (AI)
P2_COLOR = (255, 100, 50)  # Orange (Human/Opponent)
BALL_COLOR = (220, 220, 220)
BOOST_COLOR = (255, 255, 0)

# --- Physics Constants (From enginenumpy.py) ---
DRAG = 0.99 
TIRE_GRIP_NORMAL = 0.95 
TIRE_GRIP_DRIFT = 0.05
ACCEL = 0.08
TURN_SPEED = 3.5
MAX_SPEED = 6.0
BOOST_ACCEL = 0.15
BOOST_MAX_SPEED = 9.0
MAX_BOOST = 100
BALL_ELASTICITY = 0.8
CAR_WALL_ELASTICITY = 0.5 
BALL_DRAG = 0.999

# --- Arena Dimensions (From enginenumpy.py) ---
OFFSET_X, OFFSET_Y = 100, 100
ARENA_W, ARENA_H = SCREEN_W - 200, SCREEN_H - 200
GOAL_SIZE = 200
GOAL_DEPTH = 80
CORNER_SIZE = 80

# ==========================================
# --- HELPER FUNCTIONS ---
# ==========================================
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

# Pre-compute corner normals/points as numpy arrays
CORNER_NORMALS = [
    normalize(np.array([0.707, 0.707])),   # TL
    normalize(np.array([-0.707, 0.707])),  # TR
    normalize(np.array([-0.707, -0.707])), # BR
    normalize(np.array([0.707, -0.707]))   # BL
]

CORNER_POINTS = [
    np.array([OFFSET_X + CORNER_SIZE, OFFSET_Y]),           
    np.array([OFFSET_X + ARENA_W - CORNER_SIZE, OFFSET_Y]), 
    np.array([OFFSET_X + ARENA_W - CORNER_SIZE, OFFSET_Y + ARENA_H]), 
    np.array([OFFSET_X + CORNER_SIZE, OFFSET_Y + ARENA_H])  
]

# ==========================================
# --- PHYSICS CLASSES ---
# ==========================================
class PhysicsObject:
    def __init__(self, x, y, mass, radius):
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        self.mass = float(mass)
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0
        self.radius = float(radius)

class Ball(PhysicsObject):
    def __init__(self, x, y):
        super().__init__(x, y, mass=1.0, radius=22)

class Car(PhysicsObject):
    def __init__(self, x, y, angle=0, color=P1_COLOR):
        super().__init__(x, y, mass=10.0, radius=20) 
        self.angle = float(angle)
        self.color = color
        
        # Hitbox dimensions from enginenumpy.py
        self.width = 40.0 
        self.height = 24.0 
        
        self.boost = 100.0
        self.boosting = False
        self.drifting = False
        self.throttle = 0.0
        self.turn = 0.0
        self.boost_active = False

    def update_controls(self):
        current_accel = ACCEL
        current_max = MAX_SPEED
        self.boosting = False
        
        if self.boost_active and self.boost > 0:
            self.boosting = True
            current_accel = BOOST_ACCEL
            current_max = BOOST_MAX_SPEED
            self.boost -= 0.15 
        elif self.boost < MAX_BOOST:
            self.boost += 0.03
            
        vel_mag = np.linalg.norm(self.vel)
        
        # Turn logic
        if vel_mag > 0.5 or self.drifting:
            orientation = rotate_vector(np.array([1.0, 0.0]), self.angle)
            dot = np.dot(self.vel, orientation)
            direction = 1.0 if dot > 0 else -1.0
            if self.drifting: direction = 1.0
            self.angle += self.turn * TURN_SPEED * direction * 0.15 

        # Forward vector
        forward = rotate_vector(np.array([1.0, 0.0]), self.angle)
        self.vel += forward * self.throttle * current_accel

        # Lateral friction
        right = rotate_vector(forward, 90) # Rotate 90 deg
        lateral_vel = np.dot(self.vel, right)
        grip = TIRE_GRIP_DRIFT if self.drifting else TIRE_GRIP_NORMAL
        self.vel -= right * lateral_vel * grip * 0.2 
        self.vel *= DRAG

        # Cap Speed
        vel_mag = np.linalg.norm(self.vel)
        if vel_mag > current_max:
            self.vel = (self.vel / vel_mag) * current_max

# ==========================================
# --- PHYSICS SOLVER (From enginenumpy.py) ---
# ==========================================
def bounce(obj, normal, elasticity):
    """Reflects velocity off a normal with elasticity."""
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
    
    # Post locations
    tl_post = np.array([OFFSET_X, goal_top])
    bl_post = np.array([OFFSET_X, goal_bot])
    tr_post = np.array([OFFSET_X + ARENA_W, goal_top])
    br_post = np.array([OFFSET_X + ARENA_W, goal_bot])

    # 1. Post Collisions
    for post in [tl_post, bl_post, tr_post, br_post]:
        diff = obj.pos - post
        dist_sq = np.dot(diff, diff)
        if dist_sq < r*r:
            dist = np.sqrt(dist_sq)
            normal = diff / dist if dist > 0 else np.array([1.0, 0.0])
            obj.pos = post + normal * r
            bounce(obj, normal, elasticity)
            return

    # 2. Left Side
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

    # 3. Right Side
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

    # 4. Top & Bottom
    if obj.pos[1] - r < OFFSET_Y:
        obj.pos[1] = OFFSET_Y + r
        bounce(obj, np.array([0.0, 1.0]), elasticity)
    elif obj.pos[1] + r > OFFSET_Y + ARENA_H:
        obj.pos[1] = OFFSET_Y + ARENA_H - r
        bounce(obj, np.array([0.0, -1.0]), elasticity)

    # 5. Corners
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
    # Rotate ball into car's local space
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


# ==========================================
# --- MODEL ARCHITECTURES ---
# ==========================================
class OldPPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(15, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4),
            nn.Tanh()
        )
    def forward(self, x):
        return self.actor(x)

class CurriculumAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(15, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 4),
            nn.Tanh()
        )
    def forward(self, x):
        return self.actor(x)

# ==========================================
# --- OBSERVATION LOGIC ---
# ==========================================
def get_obs_old_ppo(engine):
    c1, c2, b = engine.car1, engine.car2, engine.ball
    return np.array([
        c1.pos[0]/SCREEN_W, c1.pos[1]/SCREEN_H, 
        c1.vel[0]/10, c1.vel[1]/10, 
        np.cos(np.radians(c1.angle)), np.sin(np.radians(c1.angle)), 
        c1.boost/100.0,
        b.pos[0]/SCREEN_W, b.pos[1]/SCREEN_H, 
        b.vel[0]/10, b.vel[1]/10,
        c2.pos[0]/SCREEN_W, c2.pos[1]/SCREEN_H, 
        0, 0 
    ], dtype=np.float32)

def get_obs_curriculum(engine):
    my_car = engine.car1
    opp_car = engine.car2
    ball = engine.ball
    
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

# ==========================================
# --- GAME ENGINE ---
# ==========================================
class GameEngine:
    def __init__(self):
        self.car1 = Car(250, SCREEN_H/2, 0, P1_COLOR) 
        self.car2 = Car(SCREEN_W - 250, SCREEN_H/2, 180, P2_COLOR) 
        self.ball = Ball(SCREEN_W/2, SCREEN_H/2)
        self.scores = [0, 0]
        self.reset_positions()

    def reset_positions(self):
        self.car1.pos[:] = [250.0, SCREEN_H/2]
        self.car1.vel[:] = 0
        self.car1.angle = 0.0
        self.car1.boost = 100.0
        
        self.car2.pos[:] = [float(SCREEN_W - 250), SCREEN_H/2]
        self.car2.vel[:] = 0
        self.car2.angle = 180.0
        self.car2.boost = 100.0
        
        self.ball.pos[:] = [SCREEN_W/2, SCREEN_H/2]
        self.ball.vel[:] = 0

    def step_physics(self):
        dt = 1.0 / PHYSICS_SUBSTEPS
        
        for _ in range(PHYSICS_SUBSTEPS):
            self.car1.update_controls()
            self.car2.update_controls()
            
            self.car1.pos += self.car1.vel * dt
            self.car2.pos += self.car2.vel * dt
            self.ball.pos += self.ball.vel * dt
            
            # Apply Ball Drag (From enginenumpy.py)
            self.ball.vel *= BALL_DRAG

            # Resolve Collisions using enginenumpy logic
            resolve_arena_collisions(self.car1)
            resolve_arena_collisions(self.car2)
            resolve_arena_collisions(self.ball)
            
            resolve_car_ball(self.car1, self.ball)
            resolve_car_ball(self.car2, self.ball)
            resolve_car_car(self.car1, self.car2)

        # Goal Check (Expanded from enginenumpy logic)
        if self.ball.pos[0] + self.ball.radius < OFFSET_X - 10:
             if SCREEN_H/2 - GOAL_SIZE/2 < self.ball.pos[1] < SCREEN_H/2 + GOAL_SIZE/2:
                self.scores[1] += 1
                return True
        elif self.ball.pos[0] - self.ball.radius > OFFSET_X + ARENA_W + 10:
             if SCREEN_H/2 - GOAL_SIZE/2 < self.ball.pos[1] < SCREEN_H/2 + GOAL_SIZE/2:
                self.scores[0] += 1
                return True
        
        return False

# ==========================================
# --- RENDERING HELPERS ---
# ==========================================
def draw_car(screen, car):
    # Visual size (can differ slightly from hitbox if desired, but kept close here)
    surf = pygame.Surface((55, 28), pygame.SRCALPHA)
    surf.fill(car.color)
    # Windshield
    pygame.draw.rect(surf, (200, 200, 200), (35, 2, 10, 24))
    
    rotated = pygame.transform.rotate(surf, -car.angle)
    rect = rotated.get_rect(center=(car.pos[0], car.pos[1]))
    screen.blit(rotated, rect.topleft)
    
    # Boost bar
    bx, by = car.pos[0] - 20, car.pos[1] - 40
    pygame.draw.rect(screen, (50,50,50), (bx, by, 40, 4))
    pygame.draw.rect(screen, BOOST_COLOR, (bx, by, 40 * (car.boost/100), 4))

def draw_game(screen, engine, font, mode_name):
    screen.fill(BG_COLOR)
    
    # Draw Arena
    pygame.draw.rect(screen, (30,30,40), (OFFSET_X, OFFSET_Y, ARENA_W, ARENA_H))
    pygame.draw.rect(screen, (25,25,35), (OFFSET_X-GOAL_DEPTH, SCREEN_H/2-GOAL_SIZE/2, GOAL_DEPTH, GOAL_SIZE))
    pygame.draw.rect(screen, (25,25,35), (OFFSET_X+ARENA_W, SCREEN_H/2-GOAL_SIZE/2, GOAL_DEPTH, GOAL_SIZE))
    
    # Walls
    pygame.draw.lines(screen, WALL_COLOR, True, [
        (OFFSET_X+CORNER_SIZE, OFFSET_Y), (OFFSET_X+ARENA_W-CORNER_SIZE, OFFSET_Y), 
        (OFFSET_X+ARENA_W, OFFSET_Y+CORNER_SIZE), (OFFSET_X+ARENA_W, OFFSET_Y+ARENA_H-CORNER_SIZE), 
        (OFFSET_X+ARENA_W-CORNER_SIZE, OFFSET_Y+ARENA_H), (OFFSET_X+CORNER_SIZE, OFFSET_Y+ARENA_H), 
        (OFFSET_X, OFFSET_Y+ARENA_H-CORNER_SIZE), (OFFSET_X, OFFSET_Y+CORNER_SIZE)
    ], 10)
    
    # Posts
    goal_top = SCREEN_H//2 - GOAL_SIZE//2
    goal_bot = SCREEN_H//2 + GOAL_SIZE//2
    pygame.draw.circle(screen, WALL_COLOR, (OFFSET_X, goal_top), 5)
    pygame.draw.circle(screen, WALL_COLOR, (OFFSET_X, goal_bot), 5)
    pygame.draw.circle(screen, WALL_COLOR, (OFFSET_X+ARENA_W, goal_top), 5)
    pygame.draw.circle(screen, WALL_COLOR, (OFFSET_X+ARENA_W, goal_bot), 5)

    draw_car(screen, engine.car1)
    draw_car(screen, engine.car2)
    
    # Draw Ball
    bx, by = int(engine.ball.pos[0]), int(engine.ball.pos[1])
    pygame.draw.circle(screen, BALL_COLOR, (bx, by), int(engine.ball.radius))
    pygame.draw.circle(screen, (0,0,0), (bx, by), int(engine.ball.radius), 2)
    
    # UI
    score_text = font.render(f"AI: {engine.scores[0]}   YOU: {engine.scores[1]}", True, (255, 255, 255))
    screen.blit(score_text, (SCREEN_W/2 - score_text.get_width()/2, 30))
    
    info_text = font.render(f"Mode: {mode_name}", True, (150, 150, 150))
    screen.blit(info_text, (20, 20))
    
    controls = font.render("Arrows: Move | Shift: Boost | Space: Drift | R: Reset", True, (100, 100, 100))
    screen.blit(controls, (SCREEN_W/2 - controls.get_width()/2, SCREEN_H - 40))

def draw_nn_hud(screen, obs, action, x, y):
    """
    Draws a visual representation of the Neural Network state.
    x, y: Top-left coordinates of the HUD
    """
    # Config
    WIDTH, HEIGHT = 320, 400
    bg_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    bg_surface.fill((10, 10, 15, 220)) # Semi-transparent dark background
    
    # Fonts (create locally to ensure availability)
    font_small = pygame.font.SysFont("consolas", 12)
    font_header = pygame.font.SysFont("consolas", 16, bold=True)

    # Input Labels (Mapped to your 15 observation values)
    input_names = [
        "My X", "My Y", "My VX", "My VY", "Ang Cos", "Ang Sin", "Boost", # 0-6
        "Ball X", "Ball Y", "Ball VX", "Ball VY",                        # 7-10
        "Opp X", "Opp Y", "Opp VX", "Opp VY"                             # 11-14
    ]
    
    # Output Labels (Mapped to your 4 action values)
    output_names = ["Throttle", "Steer", "Boost", "Drift"]

    # --- Helper to get color from value ---
    def get_color_val(val):
        # Clamp -1 to 1
        v = max(-1.0, min(1.0, float(val)))
        intensity = int(abs(v) * 255)
        if v > 0: return (0, intensity, 255)  # Blue for positive
        if v < 0: return (255, 50, 50)        # Red for negative
        return (50, 50, 50)                   # Grey for zero

    # Layout
    in_x, out_x = 80, 240
    start_y = 40
    step_y_in = 22
    step_y_out = 60

    # Draw Header
    title = font_header.render("NN INTERNALS", True, (200, 200, 200))
    bg_surface.blit(title, (WIDTH//2 - title.get_width()//2, 10))

    # 1. Draw Inputs (Left Side)
    node_positions_in = []
    for i, val in enumerate(obs):
        py = start_y + i * step_y_in
        color = get_color_val(val)
        
        # Text
        txt = font_small.render(f"{input_names[i][:7]}", True, (150, 150, 150))
        bg_surface.blit(txt, (5, py - 6))
        
        # Node Circle
        pygame.draw.circle(bg_surface, color, (in_x, py), 6)
        pygame.draw.circle(bg_surface, (200,200,200), (in_x, py), 7, 1) # Rim
        
        # Value Bar (mini visualizer next to text)
        bar_len = int(val * 20)
        pygame.draw.rect(bg_surface, color, (in_x - 40, py + 2, bar_len, 2))
        
        node_positions_in.append((in_x, py))

    # 2. Draw Outputs (Right Side)
    node_positions_out = []
    for i, val in enumerate(action):
        py = start_y + 40 + i * step_y_out
        
        # Logic for boolean actions (Boost/Drift) vs continuous
        if i >= 2: val = 1.0 if val > 0 else 0.0 # Binary display for buttons
        
        color = get_color_val(val)

        # Node Circle
        pygame.draw.circle(bg_surface, color, (out_x, py), 10)
        pygame.draw.circle(bg_surface, (255,255,255), (out_x, py), 11, 2)
        
        # Text
        txt = font_small.render(output_names[i], True, (255, 255, 255))
        bg_surface.blit(txt, (out_x + 15, py - 6))
        
        # Value Text
        val_txt = font_small.render(f"{val:.2f}", True, color)
        bg_surface.blit(val_txt, (out_x + 15, py + 6))

        node_positions_out.append((out_x, py))

    # 3. Draw Synapses (Visual Fake-out for aesthetics)
    # We draw faint lines from active inputs to active outputs to visualize "thought"
    for i, in_pos in enumerate(node_positions_in):
        in_val = obs[i]
        if abs(in_val) < 0.1: continue # Skip weak inputs to reduce clutter
        
        for j, out_pos in enumerate(node_positions_out):
            out_val = action[j]
            if abs(out_val) < 0.2: continue 

            # Line opacity based on signal strength
            alpha = int(abs(in_val * out_val) * 100)
            if alpha > 255: alpha = 255
            
            line_color = (100, 200, 255, alpha) if in_val * out_val > 0 else (255, 100, 100, alpha)
            
            # Draw line on a temp surface to support alpha
            pygame.draw.line(bg_surface, line_color, in_pos, out_pos, 1)

    screen.blit(bg_surface, (x, y))

# ==========================================
# --- MAIN LOOP ---
# ==========================================
def main():
    print("=========================================")
    print("       ROCKET LEAGUE AI VISUALIZER       ")
    print("=========================================")
    print("1. Old PPO Model (ppo_rocket_model.pth)")
    print("2. New Curriculum Model (curriculum_brain.pth)")
    choice = input("Select Model (1 or 2): ")

    device = torch.device("cpu")
    model = None
    mode_name = ""
    obs_func = None

    if choice == "1":
        print(">> Loading Old PPO Model...")
        model = OldPPOAgent().to(device)
        path = "ppo_rocket_model.pth"
        obs_func = get_obs_old_ppo
        mode_name = "Old PPO (Standard)"
        try:
            full_state = torch.load(path, map_location=device)
            model.load_state_dict(full_state, strict=False) 
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return
            
    elif choice == "2":
        print(">> Loading Curriculum Model...")
        model = CurriculumAgent().to(device)
        path = "curriculum_brain.pth"
        obs_func = get_obs_curriculum
        mode_name = "Curriculum (Self-Play)"
        try:
            model.load_state_dict(torch.load(path, map_location=device), strict=False)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return
    else:
        print("Invalid choice.")
        return

    print(">> Model Loaded. Starting Pygame...")
    model.eval()
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"Rocket League AI - {mode_name}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 40)
    
    engine = GameEngine()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    engine.reset_positions()
                    engine.scores = [0, 0]

        # 1. AI Logic
        obs = obs_func(engine)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = model(obs_tensor).squeeze(0).numpy()
        
        # Apply AI
        engine.car1.throttle = float(np.clip(action[0], -1, 1))
        engine.car1.turn = float(np.clip(action[1], -1, 1))
        engine.car1.boost_active = action[2] > 0
        engine.car1.drifting = action[3] > 0

        # 2. Human Logic (Car 2)
        keys = pygame.key.get_pressed()
        throttle = 0.0
        if keys[pygame.K_w]: throttle += 1.0
        if keys[pygame.K_s]: throttle -= 1.0
        
        turn = 0.0
        if keys[pygame.K_d]: turn += 1.0
        if keys[pygame.K_a]: turn -= 1.0
        
        engine.car2.throttle = throttle
        engine.car2.turn = turn
        engine.car2.boost_active = keys[pygame.K_RSHIFT] or keys[pygame.K_LSHIFT]
        engine.car2.drifting = keys[pygame.K_SPACE]

        # 3. Physics & Render
        if engine.step_physics():
            draw_game(screen, engine, font, mode_name)
            pygame.display.flip()
            pygame.time.delay(500)
            engine.reset_positions()

        draw_game(screen, engine, font, mode_name)
        draw_nn_hud(screen, obs, action, 10, 10)
        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()

if __name__ == "__main__":
    main()