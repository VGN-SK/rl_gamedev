import game_env
import random

env = game_env.GameEnvironment(800, 600)

state = env.reset()

done = False
total_reward = 0.0
steps = 0

actions = [
    game_env.Action.Up,
    game_env.Action.Down,
    game_env.Action.Left,
    game_env.Action.Right
]

while not done:
    action = random.choice(actions)

    result = env.step(action)

    state = result.nextState
    total_reward += result.reward
    done = result.done
    steps += 1

print("Episode finished")
print("Steps:", steps)
print("Total reward:", total_reward)