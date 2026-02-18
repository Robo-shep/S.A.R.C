#!/usr/bin/env python3
import pygame
import numpy as np
import math

# --- Configuration ---
SCREEN_W, SCREEN_H = 1200, 1000
TARGET_FPS = 60
PHYSICS_SUBSTEPS = 8 

# Colors
BG_COLOR = (20, 20, 30)
WALL_COLOR = (50, 50, 70)
P1_COLOR = (50, 150, 255)
P2_COLOR = (255, 100, 50)
BALL_COLOR = (220, 220, 220)
BOOST_COLOR = (255, 255, 0)

# --- Physics Constants ---
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

# --- Arena ---
OFFSET_X, OFFSET_Y = 100, 100
ARENA_W, ARENA_H = SCREEN_W - 200, SCREEN_H - 200
GOAL_SIZE = 200
GOAL_DEPTH = 80
CORNER_SIZE = 80

# --- Numpy Helper Functions ---
def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def rotate_vector(v, angle_degrees):
    # Pygame rotates clockwise for positive angles, so we use -angle for standard trig or flip y
    # Standard 2D rotation matrix:
    # [ cos -sin ]
    # [ sin  cos ]
    # However, since y is down in screen coords, clockwise rotation (positive angle) 
    # behaves like standard math rotation if we interpret the angle correctly.
    # Pygame: (1,0) rot 90 -> (0,1). 
    rad = np.radians(angle_degrees)
    c, s = np.cos(rad), np.sin(rad)
    # Rotation matrix for clockwise rotation in screen coords (y-down)
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
    
    def draw(self, screen):
        # Convert numpy to int tuple for pygame
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        pygame.draw.circle(screen, BALL_COLOR, pos_int, int(self.radius))
        pygame.draw.circle(screen, (0,0,0), pos_int, int(self.radius), 2)

class Car(PhysicsObject):
    def __init__(self, x, y, image_path, angle=0):
        super().__init__(x, y, mass=10.0, radius=20) 
        self.angle = float(angle)
        
        # Load sprite (Visualization only)
        try:
            img = pygame.image.load(image_path).convert_alpha()
            self.width, self.height = 55, 28 # Physics dims
            self.original_image = pygame.transform.scale(img, (self.width, self.height))
        except:
            # Fallback if image not found
            self.width, self.height = 55, 28
            self.original_image = pygame.Surface((self.width, self.height))
            self.original_image.fill(P1_COLOR)

        self.width = 40.0 # Hitbox
        self.height = 24.0 # Hitbox
        self.boost = 100.0
        self.boosting = False
        self.drifting = False
        self.throttle = 0.0
        self.turn = 0.0
        self.boost_active = False # Added missing init
    
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
            # Dot product to determine direction (forward/backward)
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

    def draw(self, screen):
        # Rotate image (Visuals only)
        rotated = pygame.transform.rotate(self.original_image, -self.angle)
        rect = rotated.get_rect(center=(self.pos[0], self.pos[1]))
        screen.blit(rotated, rect.topleft)

        # Boost bar
        bar_x, bar_y = self.pos[0] - 20, self.pos[1] - 40
        pygame.draw.rect(screen, (50,50,50), (bar_x, bar_y, 40, 4))
        pygame.draw.rect(screen, BOOST_COLOR, (bar_x, bar_y, 40 * (self.boost/100), 4))

# --- Physics Solver (Numpy) ---

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

# --- Main Loop ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Rocket League: Numpy Physics Engine")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 74)

    # Create assets or placeholders if not found
    car1 = Car(250, SCREEN_H/2, "assets/blue_car.png", 0)
    car2 = Car(SCREEN_W - 250, SCREEN_H/2, "assets/red_car.png", 180)
    ball = Ball(SCREEN_W/2, SCREEN_H/2)
    
    scores = [0, 0]

    def reset(scorer):
        car1.pos = np.array([250.0, SCREEN_H/2])
        car1.vel[:] = 0
        car1.angle = 0.0
        car1.boost = 100
        
        car2.pos = np.array([float(SCREEN_W-250), SCREEN_H/2])
        car2.vel[:] = 0
        car2.angle = 180.0
        car2.boost = 100
        
        ball.pos = np.array([SCREEN_W/2, SCREEN_H/2])
        ball.vel[:] = 0
        ball.vel[0] = -2.0 if scorer == 1 else 2.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r: 
                reset(0); scores=[0,0]

        keys = pygame.key.get_pressed()
        
        # P1 Input
        car1.throttle = (1 if keys[pygame.K_w] else 0) - (1 if keys[pygame.K_s] else 0)
        car1.turn = (1 if keys[pygame.K_d] else 0) - (1 if keys[pygame.K_a] else 0)
        car1.boost_active = keys[pygame.K_LSHIFT]
        car1.drifting = keys[pygame.K_SPACE]

        # P2 Input
        car2.throttle = (1 if keys[pygame.K_UP] else 0) - (1 if keys[pygame.K_DOWN] else 0)
        car2.turn = (1 if keys[pygame.K_RIGHT] else 0) - (1 if keys[pygame.K_LEFT] else 0)
        car2.boost_active = keys[pygame.K_RCTRL] or keys[pygame.K_RSHIFT]
        car2.drifting = keys[pygame.K_KP_0]  

        # Sub-stepping
        dt = 1.0 / PHYSICS_SUBSTEPS 
        for _ in range(PHYSICS_SUBSTEPS):
            car1.update_controls()
            car2.update_controls()
            
            car1.pos += car1.vel * dt
            car2.pos += car2.vel * dt
            ball.pos += ball.vel * dt
            
            # Apply Ball Drag
            ball.vel *= BALL_DRAG

            resolve_arena_collisions(car1)
            resolve_arena_collisions(car2)
            resolve_arena_collisions(ball)
            
            resolve_car_ball(car1, ball)
            resolve_car_ball(car2, ball)
            resolve_car_car(car1, car2)

        # Game Logic
        if ball.pos[0] + ball.radius < OFFSET_X - 10:
             if SCREEN_H/2 - GOAL_SIZE/2 < ball.pos[1] < SCREEN_H/2 + GOAL_SIZE/2:
                scores[1] += 1; reset(2); pygame.time.delay(500)
        elif ball.pos[0] - ball.radius > OFFSET_X + ARENA_W + 10:
             if SCREEN_H/2 - GOAL_SIZE/2 < ball.pos[1] < SCREEN_H/2 + GOAL_SIZE/2:
                scores[0] += 1; reset(1); pygame.time.delay(500)

        # Render
        screen.fill(BG_COLOR)
        
        # Draw Arena
        pygame.draw.rect(screen, (30,30,40), (OFFSET_X, OFFSET_Y, ARENA_W, ARENA_H))
        pygame.draw.rect(screen, (25,25,35), (OFFSET_X-GOAL_DEPTH, SCREEN_H/2-GOAL_SIZE/2, GOAL_DEPTH, GOAL_SIZE))
        pygame.draw.rect(screen, (25,25,35), (OFFSET_X+ARENA_W, SCREEN_H/2-GOAL_SIZE/2, GOAL_DEPTH, GOAL_SIZE))
        pygame.draw.lines(screen, WALL_COLOR, True, [
            (OFFSET_X+CORNER_SIZE, OFFSET_Y), (OFFSET_X+ARENA_W-CORNER_SIZE, OFFSET_Y), 
            (OFFSET_X+ARENA_W, OFFSET_Y+CORNER_SIZE), (OFFSET_X+ARENA_W, OFFSET_Y+ARENA_H-CORNER_SIZE), 
            (OFFSET_X+ARENA_W-CORNER_SIZE, OFFSET_Y+ARENA_H), (OFFSET_X+CORNER_SIZE, OFFSET_Y+ARENA_H), 
            (OFFSET_X, OFFSET_Y+ARENA_H-CORNER_SIZE), (OFFSET_X, OFFSET_Y+CORNER_SIZE)
        ], 10)
        
        goal_top = SCREEN_H//2 - GOAL_SIZE//2
        goal_bot = SCREEN_H//2 + GOAL_SIZE//2
        pygame.draw.circle(screen, WALL_COLOR, (OFFSET_X, goal_top), 5)
        pygame.draw.circle(screen, WALL_COLOR, (OFFSET_X, goal_bot), 5)
        pygame.draw.circle(screen, WALL_COLOR, (OFFSET_X+ARENA_W, goal_top), 5)
        pygame.draw.circle(screen, WALL_COLOR, (OFFSET_X+ARENA_W, goal_bot), 5)

        car1.draw(screen)
        car2.draw(screen)
        ball.draw(screen)
        
        sc = font.render(f"{scores[0]}   {scores[1]}", True, (255,255,255))
        screen.blit(sc, (SCREEN_W/2 - sc.get_width()/2, 30))

        pygame.display.flip()
        clock.tick(TARGET_FPS)

    pygame.quit()

if __name__ == "__main__":
    main()