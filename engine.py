import pygame
import math
import random

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
FPS = 60
BACKGROUND_COLOR = (30, 30, 30)

class PhysicsObject:
    def __init__(self, x, y, radius, mass, elasticity, color):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.radius = radius
        self.mass = mass
        # Elasticity: 0.0 = no bounce (mud), 1.0 = perfect bounce (superball)
        self.elasticity = elasticity 
        self.color = color
        # Inverse mass is pre-calculated because we divide by mass often.
        # If mass is 0 (infinite mass/static object), inv_mass is 0.
        self.inv_mass = 1.0 / mass if mass > 0 else 0

    def move(self):
        """Updates position based on velocity."""
        self.pos += self.vel
        
        # --- Wall Collisions ---
        # Bounce off left/right
        if self.pos.x - self.radius < 0:
            self.pos.x = self.radius
            self.vel.x *= -self.elasticity
        elif self.pos.x + self.radius > WIDTH:
            self.pos.x = WIDTH - self.radius
            self.vel.x *= -self.elasticity
            
        # Bounce off top/bottom
        if self.pos.y - self.radius < 0:
            self.pos.y = self.radius
            self.vel.y *= -self.elasticity
        elif self.pos.y + self.radius > HEIGHT:
            self.pos.y = HEIGHT - self.radius
            self.vel.y *= -self.elasticity

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

def resolve_collision(obj1, obj2):
    """
    Resolves collision between two circular PhysicsObjects using Impulse.
    """
    collision_vector = obj1.pos - obj2.pos
    distance = collision_vector.length()
    min_dist = obj1.radius + obj2.radius

    # 1. Check if colliding
    if distance < min_dist:
        # Avoid division by zero
        if distance == 0:
            collision_vector = pygame.math.Vector2(1, 0)
            distance = 1

        # Normal vector (direction of collision)
        normal = collision_vector / distance

        # --- Positional Correction (prevent sticking) ---
        # Push objects apart so they no longer overlap
        correction_percent = 0.8 # Stabilizes jitter
        overlap = min_dist - distance
        correction = (normal * overlap * correction_percent) / (obj1.inv_mass + obj2.inv_mass)
        
        obj1.pos += correction * obj1.inv_mass
        obj2.pos -= correction * obj2.inv_mass

        # --- Impulse Resolution (Velocity change) ---
        # Relative velocity
        rel_vel = obj1.vel - obj2.vel
        
        # Calculate velocity along the normal
        vel_along_normal = rel_vel.dot(normal)

        # Do not resolve if velocities are separating
        if vel_along_normal > 0:
            return

        # Calculate elasticity (using the lower of the two for realism)
        e = min(obj1.elasticity, obj2.elasticity)

        # Calculate impulse scalar
        j = -(1 + e) * vel_along_normal
        j /= (obj1.inv_mass + obj2.inv_mass)

        # Apply impulse
        impulse = j * normal
        obj1.vel += impulse * obj1.inv_mass
        obj2.vel -= impulse * obj2.inv_mass

# --- Main Simulation ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Physics Engine: Impulse & Elasticity")
    clock = pygame.time.Clock()

    objects = []

    # Example 1: Large, heavy, bouncy ball (The "Planet")
    objects.append(PhysicsObject(x=400, y=300, radius=50, mass=100, elasticity=0.9, color=(255, 100, 100)))

    # Example 2: Small, light, fast balls
    for i in range(10):
        obj = PhysicsObject(
            x=random.randint(50, WIDTH-50),
            y=random.randint(50, HEIGHT-50),
            radius=random.randint(10, 20),
            mass=random.randint(5, 15), # Lighter mass
            elasticity=0.7,
            color=(100, 200, 255)
        )
        obj.vel = pygame.math.Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
        objects.append(obj)

    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Click to add a heavy rock (low elasticity)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                rock = PhysicsObject(mx, my, 30, 50, 0.2, (150, 150, 150)) # Elasticity 0.2
                rock.vel = pygame.math.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
                objects.append(rock)

        # 2. Update Physics
        # Move everything
        for obj in objects:
            obj.move()

        # Check collisions (Naive O(N^2) approach)
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                resolve_collision(objects[i], objects[j])

        # 3. Render
        screen.fill(BACKGROUND_COLOR)
        for obj in objects:
            obj.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
