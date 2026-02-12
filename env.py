class SoccerEnv:
    def __init__(self):
        self.car = Car(...)
        self.ball = Ball(...)
        self.done = False

    def reset(self):
        self.car.pos.update(250, 500)
        self.car.vel.update(0, 0)
        self.ball.pos.update(600, 500)
        self.ball.vel.update(0, 0)
        self.done = False
        return self.get_state()

    def get_state(self):
        return [
            self.car.pos.x,
            self.car.pos.y,
            self.car.vel.x,
            self.car.vel.y,
            self.car.angle,
            self.ball.pos.x,
            self.ball.pos.y,
            self.ball.vel.x,
            self.ball.vel.y
        ]

    def step(self, action):
        """
        action: integer from 0–5
        """

        # decode action
        self.car.throttle = 0
        self.car.turn = 0
        self.car.boost_active = False

        if action == 1:
            self.car.throttle = 1
        elif action == 2:
            self.car.throttle = -1
        elif action == 3:
            self.car.turn = -1
        elif action == 4:
            self.car.turn = 1
        elif action == 5:
            self.car.boost_active = True

        # run physics step
        self.car.update_controls()
        self.car.pos += self.car.vel
        self.ball.pos += self.ball.vel

        resolve_car_ball(self.car, self.ball)

        # reward function
        reward = -0.01  # time penalty

        # reward for hitting ball
        if (self.car.pos - self.ball.pos).length() < 40:
            reward += 0.1

        # goal condition
        if self.ball.pos.x < 100:
            reward += 10
            self.done = True

        return self.get_state(), reward, self.done
