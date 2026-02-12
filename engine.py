#!/usr/bin/env python3

import pygame
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

# --- Ball Physics ---
# Lower number = Higher Drag (stops faster)
# 0.995 was ice, 0.985 is short grass
BALL_DRAG = 0.999

# --- Arena ---
OFFSET_X, OFFSET_Y = 100, 100
ARENA_W, ARENA_H = SCREEN_W - 200, SCREEN_H - 200
GOAL_SIZE = 160
GOAL_DEPTH = 60
CORNER_SIZE = 80

# Normals and Points for Corners
CORNER_NORMALS = [
    pygame.math.Vector2(0.707, 0.707),   # TL
    pygame.math.Vector2(-0.707, 0.707),  # TR
    pygame.math.Vector2(-0.707, -0.707), # BR
    pygame.math.Vector2(0.707, -0.707)   # BL
]
CORNER_POINTS = [
    pygame.math.Vector2(OFFSET_X + CORNER_SIZE, OFFSET_Y),           
    pygame.math.Vector2(OFFSET_X + ARENA_W - CORNER_SIZE, OFFSET_Y), 
    pygame.math.Vector2(OFFSET_X + ARENA_W - CORNER_SIZE, OFFSET_Y + ARENA_H), 
    pygame.math.Vector2(OFFSET_X + CORNER_SIZE, OFFSET_Y + ARENA_H)  
]

class PhysicsObject:
    def __init__(self, x, y, mass, radius):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0
        self.radius = radius 

class Ball(PhysicsObject):
    def __init__(self, x, y):
        super().__init__(x, y, mass=1.0, radius=18)
    
    def draw(self, screen):
        pygame.draw.circle(screen, BALL_COLOR, (int(self.pos.x), int(self.pos.y)), self.radius)
        pygame.draw.circle(screen, (0,0,0), (int(self.pos.x), int(self.pos.y)), self.radius, 2)

class Car(PhysicsObject):
    def __init__(self, x, y, color, angle=0):
        super().__init__(x, y, mass=10.0, radius=20) 
        self.angle = angle
        self.color = color
        self.width = 40
        self.height = 24
        self.boost = 100.0
        self.boosting = False
        self.drifting = False
        self.throttle = 0
        self.turn = 0
    
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
            
        if self.vel.length() > 0.5 or self.drifting:
            direction = 1 if self.vel.dot(pygame.math.Vector2(1,0).rotate(self.angle)) > 0 else -1
            if self.drifting: direction = 1 
            self.angle += self.turn * TURN_SPEED * direction * 0.15 

        forward = pygame.math.Vector2(1, 0).rotate(self.angle)
        self.vel += forward * self.throttle * current_accel

        right = forward.rotate(90)
        lateral_vel = self.vel.dot(right)
        grip = TIRE_GRIP_DRIFT if self.drifting else TIRE_GRIP_NORMAL
        self.vel -= right * lateral_vel * grip * 0.2 
        self.vel *= DRAG

        if self.vel.length() > current_max:
            self.vel.scale_to_length(current_max)

    def get_corners(self):
        corners = []
        offsets = [
            pygame.math.Vector2(self.width/2, self.height/2),
            pygame.math.Vector2(-self.width/2, self.height/2),
            pygame.math.Vector2(-self.width/2, -self.height/2),
            pygame.math.Vector2(self.width/2, -self.height/2),
        ]
        for offset in offsets:
            corners.append(self.pos + offset.rotate(self.angle))
        return corners

    def draw(self, screen):
        corners = self.get_corners()
        pygame.draw.polygon(screen, self.color, corners)
        front = (corners[0] + corners[3]) / 2
        pygame.draw.circle(screen, (255,255,255), (int(front.x), int(front.y)), 3)
        
        bar_x, bar_y = self.pos.x - 20, self.pos.y - 30
        pygame.draw.rect(screen, (50,50,50), (bar_x, bar_y, 40, 4))
        pygame.draw.rect(screen, BOOST_COLOR, (bar_x, bar_y, 40 * (self.boost/100), 4))
        
        if self.boosting:
            ex = (corners[1] + corners[2]) / 2
            pygame.draw.circle(screen, (255, 100, 0), (int(ex.x), int(ex.y)), 6)

# --- Strict Physics Solver ---

def bounce(obj, normal, elasticity):
    """Reflects velocity off a normal with elasticity."""
    vel_along_normal = obj.vel.dot(normal)
    if vel_along_normal < 0: 
        j = -(1 + elasticity) * vel_along_normal
        impulse = j * normal
        obj.vel += impulse

def resolve_arena_collisions(obj):
    r = obj.radius
    elasticity = BALL_ELASTICITY if isinstance(obj, Ball) else CAR_WALL_ELASTICITY

    # Goal Y-Bounds
    goal_top = SCREEN_H/2 - GOAL_SIZE/2
    goal_bot = SCREEN_H/2 + GOAL_SIZE/2

    # --- 1. LEFT SIDE ---
    if obj.pos.x - r < OFFSET_X:
        if goal_top - 20 <= obj.pos.y <= goal_bot + 20:
            # Inside Goal Mouth
            # CHECK BACK WALL OF NET (FIXED)
            back_net_x = OFFSET_X - GOAL_DEPTH
            if obj.pos.x - r < back_net_x:
                obj.pos.x = back_net_x + r
                bounce(obj, pygame.math.Vector2(1, 0), elasticity)
            
            # Goal Posts (Corners)
            if obj.pos.y < goal_top + 10 and obj.pos.x > OFFSET_X - 10:
                 diff = obj.pos - pygame.math.Vector2(OFFSET_X, goal_top)
                 if diff.length() < r:
                     normal = diff.normalize()
                     obj.pos = pygame.math.Vector2(OFFSET_X, goal_top) + normal * r
                     bounce(obj, normal, elasticity)
            elif obj.pos.y > goal_bot - 10 and obj.pos.x > OFFSET_X - 10:
                 diff = obj.pos - pygame.math.Vector2(OFFSET_X, goal_bot)
                 if diff.length() < r:
                     normal = diff.normalize()
                     obj.pos = pygame.math.Vector2(OFFSET_X, goal_bot) + normal * r
                     bounce(obj, normal, elasticity)
        else:
            # Solid Wall
            obj.pos.x = OFFSET_X + r
            bounce(obj, pygame.math.Vector2(1, 0), elasticity)

    # --- 2. RIGHT SIDE ---
    elif obj.pos.x + r > OFFSET_X + ARENA_W:
        if goal_top - 20 <= obj.pos.y <= goal_bot + 20:
            # Inside Goal Mouth
            # CHECK BACK WALL OF NET (FIXED)
            back_net_x = OFFSET_X + ARENA_W + GOAL_DEPTH
            if obj.pos.x + r > back_net_x:
                obj.pos.x = back_net_x - r
                bounce(obj, pygame.math.Vector2(-1, 0), elasticity)
            
            # Goal Posts (Corners)
            if obj.pos.y < goal_top + 10 and obj.pos.x < OFFSET_X + ARENA_W + 10:
                 diff = obj.pos - pygame.math.Vector2(OFFSET_X + ARENA_W, goal_top)
                 if diff.length() < r:
                     normal = diff.normalize()
                     obj.pos = pygame.math.Vector2(OFFSET_X + ARENA_W, goal_top) + normal * r
                     bounce(obj, normal, elasticity)
            elif obj.pos.y > goal_bot - 10 and obj.pos.x < OFFSET_X + ARENA_W + 10:
                 diff = obj.pos - pygame.math.Vector2(OFFSET_X + ARENA_W, goal_bot)
                 if diff.length() < r:
                     normal = diff.normalize()
                     obj.pos = pygame.math.Vector2(OFFSET_X + ARENA_W, goal_bot) + normal * r
                     bounce(obj, normal, elasticity)
        else:
            # Solid Wall
            obj.pos.x = OFFSET_X + ARENA_W - r
            bounce(obj, pygame.math.Vector2(-1, 0), elasticity)

    # --- 3. TOP & BOTTOM WALLS ---
    if obj.pos.y - r < OFFSET_Y:
        obj.pos.y = OFFSET_Y + r
        bounce(obj, pygame.math.Vector2(0, 1), elasticity)
    elif obj.pos.y + r > OFFSET_Y + ARENA_H:
        obj.pos.y = OFFSET_Y + ARENA_H - r
        bounce(obj, pygame.math.Vector2(0, -1), elasticity)

    # --- 4. CORNERS ---
    for i in range(4):
        anchor = CORNER_POINTS[i]
        normal = CORNER_NORMALS[i]
        dist = (obj.pos - anchor).dot(normal)
        if dist < r:
            overlap = r - dist
            obj.pos += normal * overlap
            bounce(obj, normal, elasticity)

def resolve_car_ball(car, ball):
    diff = ball.pos - car.pos
    local_ball = diff.rotate(-car.angle)
    hw, hh = car.width/2, car.height/2
    cx = max(-hw, min(local_ball.x, hw))
    cy = max(-hh, min(local_ball.y, hh))
    closest_local = pygame.math.Vector2(cx, cy)
    dist_vec = local_ball - closest_local
    dist_sq = dist_vec.length_squared()
    
    if dist_sq < ball.radius * ball.radius:
        if dist_sq == 0: dist_vec = pygame.math.Vector2(ball.radius, 0); dist_sq = ball.radius**2
        dist = math.sqrt(dist_sq)
        normal_local = dist_vec / dist
        normal_world = normal_local.rotate(car.angle)
        overlap = ball.radius - dist
        
        total_inv_mass = car.inv_mass + ball.inv_mass
        move_per_inv_mass = (normal_world * overlap) / total_inv_mass
        car.pos -= move_per_inv_mass * car.inv_mass
        ball.pos += move_per_inv_mass * ball.inv_mass
        
        rel_vel = ball.vel - car.vel
        vel_along_normal = rel_vel.dot(normal_world)
        if vel_along_normal > 0: return 
        
        j = -(1 + BALL_ELASTICITY) * vel_along_normal
        j /= total_inv_mass
        impulse = j * normal_world
        ball.vel += impulse * ball.inv_mass
        car.vel -= impulse * car.inv_mass

def resolve_car_car(c1, c2):
    dist_vec = c1.pos - c2.pos
    dist = dist_vec.length()
    min_dist = 40 
    
    if dist < min_dist:
        overlap = min_dist - dist
        if dist == 0: dist_vec = pygame.math.Vector2(1,0); dist=1
        normal = dist_vec / dist
        
        c1.pos += normal * (overlap * 0.5)
        c2.pos -= normal * (overlap * 0.5)
        
        rel_vel = c1.vel - c2.vel
        vel_along = rel_vel.dot(normal)
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
    pygame.display.set_caption("Rocket League: Final Fixed Physics")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 74)

    car1 = Car(250, SCREEN_H/2, P1_COLOR, 0)
    car2 = Car(SCREEN_W - 250, SCREEN_H/2, P2_COLOR, 180)
    ball = Ball(SCREEN_W/2, SCREEN_H/2)
    
    scores = [0, 0]

    def reset(scorer):
        car1.pos.update(250, SCREEN_H/2); car1.vel.update(0,0); car1.angle=0
        car2.pos.update(SCREEN_W-250, SCREEN_H/2); car2.vel.update(0,0); car2.angle=180
        ball.pos.update(SCREEN_W/2, SCREEN_H/2); ball.vel.update(0,0)
        ball.vel.x = -2 if scorer == 1 else 2

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
        car2.drifting = keys[pygame.K_KP_0] or keys[pygame.K_RSHIFT] 

        # Sub-stepping
        dt = 1.0 / PHYSICS_SUBSTEPS 
        for _ in range(PHYSICS_SUBSTEPS):
            car1.update_controls()
            car2.update_controls()
            
            car1.pos += car1.vel * dt
            car2.pos += car2.vel * dt
            ball.pos += ball.vel * dt
            
            # Apply Ball Drag specifically (not in PhysicsObject)
            ball.vel *= BALL_DRAG

            resolve_arena_collisions(car1)
            resolve_arena_collisions(car2)
            resolve_arena_collisions(ball)
            
            resolve_car_ball(car1, ball)
            resolve_car_ball(car2, ball)
            resolve_car_car(car1, car2)

        # Game Logic
        # Added buffer (+15) so goal only counts if ball hits back of net roughly
        if ball.pos.x + ball.radius < OFFSET_X - 10:
             if SCREEN_H/2 - GOAL_SIZE/2 < ball.pos.y < SCREEN_H/2 + GOAL_SIZE/2:
                scores[1] += 1; reset(2); pygame.time.delay(500)
        elif ball.pos.x - ball.radius > OFFSET_X + ARENA_W + 10:
             if SCREEN_H/2 - GOAL_SIZE/2 < ball.pos.y < SCREEN_H/2 + GOAL_SIZE/2:
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
        
        # Post Indicators
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
