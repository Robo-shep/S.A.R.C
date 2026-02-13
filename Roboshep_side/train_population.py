import pygame
import torch
import torch.nn as nn
import numpy as np
import random
import math
import copy

# ------------------------
# Basic Physics Constants
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

MATCH_STEPS = 900  # steps per match


# ------------------------
# Neural Network
# ------------------------

class Policy(nn.Module):
    def __init__(self, state_size=11, action_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# ------------------------
# Game Environment
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
        self.steps = 0
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

        # wall collisions
        for obj in [self.car1, self.car2, self.ball]:
            for i in [0, 1]:
                if obj.pos[i] < ARENA_MARGIN:
                    obj.pos[i] = ARENA_MARGIN
                    obj.vel[i] *= -0.7
                if obj.pos[i] > (SCREEN_W if i == 0 else SCREEN_H) - ARENA_MARGIN:
                    obj.pos[i] = (SCREEN_W if i == 0 else SCREEN_H) - ARENA_MARGIN
                    obj.vel[i] *= -0.7

        # ball-car collisions
        for car in [self.car1, self.car2]:
            diff = self.ball.pos - car.pos
            dist = np.linalg.norm(diff)
            if dist < CAR_RADIUS + BALL_RADIUS:
                normal = diff / (dist + 1e-6)
                self.ball.vel += normal * 2.5

    def step(self, action1, action2):
        self.apply_action(self.car1, action1)
        self.apply_action(self.car2, action2)

        self.physics()

        reward1 = 0
        reward2 = 0

        # goal check
        goal_top = SCREEN_H / 2 - GOAL_SIZE / 2
        goal_bot = SCREEN_H / 2 + GOAL_SIZE / 2

        if self.ball.pos[0] < ARENA_MARGIN:
            if goal_top < self.ball.pos[1] < goal_bot:
                reward1 -= 1
                reward2 += 1
                return True, reward1, reward2

        if self.ball.pos[0] > SCREEN_W - ARENA_MARGIN:
            if goal_top < self.ball.pos[1] < goal_bot:
                reward1 += 1
                reward2 -= 1
                return True, reward1, reward2

        self.steps += 1
        if self.steps > MATCH_STEPS:
            return True, reward1, reward2

        return False, reward1, reward2


# ------------------------
# Population Training
# ------------------------

POP_SIZE = 10
ELITE_COUNT = 3
GENERATIONS = 200
MUTATION_STRENGTH = 0.05


def mutate(model):
    new_model = copy.deepcopy(model)
    for param in new_model.parameters():
        param.data += torch.randn_like(param) * MUTATION_STRENGTH
    return new_model


def act(model, state):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32)
        out = model(s).numpy()

    # Convert outputs into actions
    throttle = np.clip(out[0], -1, 1)
    turn = np.clip(out[1], -1, 1)
    boost = 1 if out[2] > 0 else 0
    drift = 1 if out[3] > 0 else 0

    return [throttle, turn, boost, drift]


def play_match(modelA, modelB):
    env = Env()
    state = env.reset()

    scoreA = 0
    scoreB = 0

    done = False
    while not done:
        stateA = state
        stateB = state[::-1].copy()  # mirrored state

        actionA = act(modelA, stateA)
        actionB = act(modelB, stateB)

        done, rA, rB = env.step(actionA, actionB)
        scoreA += rA
        scoreB += rB
        state = env.get_state()

    return scoreA, scoreB


# ------------------------
# Main Training Loop
# ------------------------

population = [Policy() for _ in range(POP_SIZE)]
best_score_ever = -9999

for gen in range(GENERATIONS):
    scores = [0 for _ in range(POP_SIZE)]

    # Round-robin matches
    for i in range(POP_SIZE):
        for j in range(i + 1, POP_SIZE):
            sA, sB = play_match(population[i], population[j])
            scores[i] += sA
            scores[j] += sB

    # Rank population
    ranked = list(zip(population, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    best_model, best_score = ranked[0]

    print(f"Gen {gen} | Best Score: {best_score:.2f}")

    if best_score > best_score_ever:
        best_score_ever = best_score
        torch.save(best_model.state_dict(), "best_model.pth")
        print("Saved new best model")

    # Keep elites
    new_population = [copy.deepcopy(m[0]) for m in ranked[:ELITE_COUNT]]

    # Refill population with mutations
    while len(new_population) < POP_SIZE:
        parent = random.choice(new_population)
        child = mutate(parent)
        new_population.append(child)

    population = new_population

print("Training complete.")
