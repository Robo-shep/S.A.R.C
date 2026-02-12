import pygame
import math

# --- Configuration ---
SCREEN_W, SCREEN_H = 1000, 700
ARENA_W, ARENA_H = 850, 550 # The playable area inside walls
GOAL_WIDTH = 180    # How wide the goal opening is
GOAL_DEPTH = 40     # How deep the goal recess is
FPS = 60

# Colors
BG_COLOR = (20, 20, 30)
WALL_COLOR = (50, 50, 70)
LINE_COLOR = (100, 100, 120)
P1_COLOR = (50, 150, 255)   # Blue Team
P2_COLOR = (255, 100, 50)   # Orange Team
BALL_COLOR = (220, 220, 220)

# Calculate Arena Offsets to center it
OFFSET_X = (SCREEN_W - ARENA_W) // 2
OFFSET_Y = (SCREEN_H - ARENA_H) // 2
GOAL_TOP_Y = (SCREEN_H // 2) - (GOAL_WIDTH // 2)
GOAL_BOTTOM_Y = (SCREEN_H // 2) + (GOAL_WIDTH // 2)

class PhysicsObject:
    def __init__(self, x, y, radius, mass, elasticity, color):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.radius = radius
        self.mass = mass
        self.elasticity = elasticity
        self.color = color
        self.inv_mass = 1.0 / mass if mass > 0 else 0
        # Drag: 1.0 = no friction, 0.9 = quickly slows down
        self.drag = 0.985 

    def update_physics(self):
        """Applies drag and updates position."""
        # Apply air/ground friction
        self.vel *= self.drag
        
        # Terminal velocity cap to prevent physics glitches at extreme speeds
        if self.vel.length_squared() > 40000: # cap speed around 200
             self.vel.scale_to_length(200)

        self.pos += self.vel
        self.handle_arena_collisions()

    def handle_arena_collisions(self):
        # --- Top & Bottom Walls ---
        if self.pos.y - self.radius < OFFSET_Y:
            self.pos.y = OFFSET_Y + self.radius
            self.vel.y *= -self.elasticity
        elif self.pos.y + self.radius > OFFSET_Y + ARENA_H:
            self.pos.y = OFFSET_Y + ARENA_H - self.radius
            self.vel.y *= -self.elasticity

        # --- Side Walls and Goals ---
        # Is the object vertically within the goal opening?
        in_goal_opening = GOAL_TOP_Y < self.pos.y < GOAL_BOTTOM_Y

        # Left Side
        if self.pos.x - self.radius < OFFSET_X:
            if in_goal_opening:
                # Check back of goal net
                if self.pos.x - self.radius < OFFSET_X - GOAL_DEPTH:
                     self.pos.x = OFFSET_X - GOAL_DEPTH + self.radius
                     self.vel.x *= -self.elasticity
            else:
                # Normal wall bounce
                self.pos.x = OFFSET_X + self.radius
                self.vel.x *= -self.elasticity

        # Right Side
        elif self.pos.x + self.radius > OFFSET_X + ARENA_W:
            if in_goal_opening:
                # Check back of goal net
                if self.pos.x + self.radius > OFFSET_X + ARENA_W + GOAL_DEPTH:
                     self.pos.x = OFFSET_X + ARENA_W + GOAL_DEPTH - self.radius
                     self.vel.x *= -self.elasticity
            else:
                 # Normal wall bounce
                self.pos.x = OFFSET_X + ARENA_W - self.radius
                self.vel.x *= -self.elasticity
                
    def apply_force(self, force_vec):
        # F = ma  ->  a = F / m.  We add acceleration to velocity.
        self.vel += force_vec * self.inv_mass

    def draw(self, screen):
        # Draw main body
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)
        # Draw a subtle outline for contrast
        pygame.draw.circle(screen, (0,0,0), (int(self.pos.x), int(self.pos.y)), self.radius, 2)


def resolve_collision(obj1, obj2):
    # (Standard Impulse Resolution - Same as previous code)
    collision_vector = obj1.pos - obj2.pos
    distance = collision_vector.length()
    min_dist = obj1.radius + obj2.radius

    if distance < min_dist:
        if distance == 0:
            collision_vector = pygame.math.Vector2(1, 0)
            distance = 1
        normal = collision_vector / distance

        # Positional Correction
        correction_percent = 0.5 
        overlap = min_dist - distance
        correction = (normal * overlap * correction_percent) / (obj1.inv_mass + obj2.inv_mass)
        obj1.pos += correction * obj1.inv_mass
        obj2.pos -= correction * obj2.inv_mass

        # Impulse Resolution
        rel_vel = obj1.vel - obj2.vel
        vel_along_normal = rel_vel.dot(normal)
        if vel_along_normal > 0: return

        e = min(obj1.elasticity, obj2.elasticity)
        j = -(1 + e) * vel_along_normal
        j /= (obj1.inv_mass + obj2.inv_mass)
        impulse = j * normal
        obj1.vel += impulse * obj1.inv_mass
        obj2.vel -= impulse * obj2.inv_mass

# --- Game Specific Functions ---

def draw_arena(screen):
    # Main floor
    pygame.draw.rect(screen, (35, 35, 45), (OFFSET_X, OFFSET_Y, ARENA_W, ARENA_H))
    # Center Line
    mid_x = SCREEN_W // 2
    pygame.draw.line(screen, LINE_COLOR, (mid_x, OFFSET_Y), (mid_x, OFFSET_Y + ARENA_H), 3)
    pygame.draw.circle(screen, LINE_COLOR, (mid_x, SCREEN_H//2), 70, 3)
    
    # Walls (Top/Bottom)
    pygame.draw.rect(screen, WALL_COLOR, (OFFSET_X-20, OFFSET_Y-20, ARENA_W+40, 20))
    pygame.draw.rect(screen, WALL_COLOR, (OFFSET_X-20, OFFSET_Y+ARENA_H, ARENA_W+40, 20))

    # Side Walls (Top parts)
    pygame.draw.rect(screen, WALL_COLOR, (OFFSET_X-20, OFFSET_Y, 20, GOAL_TOP_Y - OFFSET_Y))
    pygame.draw.rect(screen, WALL_COLOR, (OFFSET_X+ARENA_W, OFFSET_Y, 20, GOAL_TOP_Y - OFFSET_Y))
    # Side Walls (Bottom parts)
    pygame.draw.rect(screen, WALL_COLOR, (OFFSET_X-20, GOAL_BOTTOM_Y, 20, OFFSET_Y+ARENA_H - GOAL_BOTTOM_Y))
    pygame.draw.rect(screen, WALL_COLOR, (OFFSET_X+ARENA_W, GOAL_BOTTOM_Y, 20, OFFSET_Y+ARENA_H - GOAL_BOTTOM_Y))

    # Goal areas (recessed)
    pygame.draw.rect(screen, (20,20,20), (OFFSET_X-GOAL_DEPTH, GOAL_TOP_Y, GOAL_DEPTH, GOAL_WIDTH))
    pygame.draw.rect(screen, (20,20,20), (OFFSET_X+ARENA_W, GOAL_TOP_Y, GOAL_DEPTH, GOAL_WIDTH))
    
def reset_positions(p1, p2, ball):
    p1.pos = pygame.math.Vector2(OFFSET_X + 150, SCREEN_H // 2)
    p1.vel = pygame.math.Vector2(0, 0)
    p2.pos = pygame.math.Vector2(OFFSET_X + ARENA_W - 150, SCREEN_H // 2)
    p2.vel = pygame.math.Vector2(0, 0)
    ball.pos = pygame.math.Vector2(SCREEN_W // 2, SCREEN_H // 2)
    ball.vel = pygame.math.Vector2(0, 0)

def check_goal(ball):
    # Returns 1 if Blue scores, 2 if Orange scores, 0 otherwise
    if ball.pos.x < OFFSET_X:
        return 2 # Orange scores on left goal
    elif ball.pos.x > OFFSET_X + ARENA_W:
        return 1 # Blue scores on right goal
    return 0

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Top-Down Physics Soccer")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 74)

    # --- Object Initialization (Tuning is key here!) ---
    # Cars: Small radius, High mass, Low bounce, High drag
    p1_car = PhysicsObject(0, 0, radius=22, mass=40, elasticity=0.3, color=P1_COLOR)
    p1_car.drag = 0.95 

    p2_car = PhysicsObject(0, 0, radius=22, mass=40, elasticity=0.3, color=P2_COLOR)
    p2_car.drag = 0.95

    # Ball: Larger radius, lighter mass relative to size, high bounce, lower drag
    ball = PhysicsObject(0, 0, radius=30, mass=5, elasticity=0.85, color=BALL_COLOR)
    ball.drag = 0.99

    reset_positions(p1_car, p2_car, ball)

    objects = [p1_car, p2_car, ball]
    scores = [0, 0] # [Blue, Orange]

    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_positions(p1_car, p2_car, ball)
                    scores = [0,0]

        # --- Player Controls ---
        keys = pygame.key.get_pressed()
        accel_power = 35 # Force applied per frame

        # Player 1 (WASD)
        input_p1 = pygame.math.Vector2(0,0)
        if keys[pygame.K_a]: input_p1.x -= 1
        if keys[pygame.K_d]: input_p1.x += 1
        if keys[pygame.K_w]: input_p1.y -= 1
        if keys[pygame.K_s]: input_p1.y += 1
        if input_p1.length_squared() > 0:
            p1_car.apply_force(input_p1.normalize() * accel_power)

        # Player 2 (Arrows)
        input_p2 = pygame.math.Vector2(0,0)
        if keys[pygame.K_LEFT]: input_p2.x -= 1
        if keys[pygame.K_RIGHT]: input_p2.x += 1
        if keys[pygame.K_UP]: input_p2.y -= 1
        if keys[pygame.K_DOWN]: input_p2.y += 1
        if input_p2.length_squared() > 0:
            p2_car.apply_force(input_p2.normalize() * accel_power)

        # 2. Update Physics
        for obj in objects:
            obj.update_physics()

        # Resolve Object Collisions (Car vs Car, Car vs Ball)
        # Using the same function from the previous engine
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                resolve_collision(objects[i], objects[j])

        # Check Goals
        goal_status = check_goal(ball)
        if goal_status > 0:
            scores[goal_status - 1] += 1
            reset_positions(p1_car, p2_car, ball)
            # A brief pause after a goal
            pygame.time.delay(500) 

        # 3. Render
        screen.fill(BG_COLOR)
        draw_arena(screen)
        
        # Draw Score
        score_text = font.render(f"{scores[0]} - {scores[1]}", True, (255,255,255))
        screen.blit(score_text, (SCREEN_W//2 - score_text.get_width()//2, 20))

        for obj in objects:
            obj.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
