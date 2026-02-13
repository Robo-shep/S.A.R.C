import pygame
import torch
import numpy as np
import math

# ------------------------
# Same constants as training
# ------------------------

SCREEN_W, SCREEN_H = 800, 600
ARENA_MARGIN = 60
GOAL_SIZE = 160

CAR_RADIUS = 18
BALL_RADIUS = 14

MAX_SPEED = 4.0
ACCEL = 0.2
TURN_SPEED = 4
BALL_DRAG = 0.995


# ------------------------
# Neural Network
# ------------------------

STATE_SIZE = 11
ACTION_SIZE = 4

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(STATE_SIZE, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, ACTION_SIZE),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# ------------------------
# Game Objects
# ------------------------

class Car:
    def __init__(self, x, y, angle):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.angle = angle


class Ball:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)


class Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.car1 = Car(150, SCREEN_H / 2, 0)
        self.car2 = Car(SCREEN_W - 150, SCREEN_H / 2, 180)
        self.ball = Ball(SCREEN_W / 2, SCREEN_H / 2)
        return self.get_state()

    def get_state(self):
        return np.concatenate([
            self.car1.pos / SCREEN_W,
            self.car1.vel / MAX_SPEED,
            [self.car1.angle / 360],
            self.ball.pos / SCREEN_W,
            self.ball.vel / MAX_SPEED,
            self.car2.pos / SCREEN_W
        ])

    def apply_action(self, car, action):
        throttle, turn, boost, drift = action

        car.angle += turn * TURN_SPEED
        forward = np.array([
            math.cos(math.radians(car.angle)),
            math.sin(math.radians(car.angle))
        ])

        car.vel += forward * throttle * ACCEL

        speed = np.linalg.norm(car.vel)
        if speed > MAX_SPEED:
            car.vel = car.vel / speed * MAX_SPEED

    def physics(self):
        self.car1.pos += self.car1.vel
        self.car2.pos += self.car2.vel
        self.ball.pos += self.ball.vel
        self.ball.vel *= BALL_DRAG

        for obj in [self.car1, self.car2, self.ball]:
            for i in [0, 1]:
                if obj.pos[i] < ARENA_MARGIN:
                    obj.pos[i] = ARENA_MARGIN
                    obj.vel[i] *= -0.7
                if obj.pos[i] > (SCREEN_W if i == 0 else SCREEN_H) - ARENA_MARGIN:
                    obj.pos[i] = (SCREEN_W if i == 0 else SCREEN_H) - ARENA_MARGIN
                    obj.vel[i] *= -0.7

    def step(self, action1, action2):
        self.apply_action(self.car1, action1)
        self.apply_action(self.car2, action2)
        self.physics()
        return self.get_state()


# ------------------------
# AI Action
# ------------------------

def act(model, state):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32)
        out = model(s).numpy()

    throttle = np.clip(out[0], -1, 1)
    turn = np.clip(out[1], -1, 1)
    boost = 1 if out[2] > 0 else 0
    drift = 1 if out[3] > 0 else 0

    return [throttle, turn, boost, drift]


# ------------------------
# Main Viewer
# ------------------------

pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

# Load model
model = Policy()
model.load_state_dict(torch.load("Roboshep_side/best_model.pth"))
model.eval()

env = Env()
state = env.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Same model plays both sides
    action1 = act(model, state)
    action2 = act(model, state[::-1].copy())

    state = env.step(action1, action2)

    # ---------------- RENDER ----------------
    screen.fill((30, 30, 40))

    # Arena
    pygame.draw.rect(
        screen,
        (50, 50, 60),
        (ARENA_MARGIN, ARENA_MARGIN,
         SCREEN_W - 2*ARENA_MARGIN,
         SCREEN_H - 2*ARENA_MARGIN),
        4
    )

    # Ball
    pygame.draw.circle(
        screen,
        (220, 220, 220),
        env.ball.pos.astype(int),
        BALL_RADIUS
    )

    # Cars
    pygame.draw.circle(
        screen,
        (80, 160, 255),
        env.car1.pos.astype(int),
        CAR_RADIUS
    )
    pygame.draw.circle(
        screen,
        (255, 100, 80),
        env.car2.pos.astype(int),
        CAR_RADIUS
    )

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
