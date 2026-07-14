import time
import torch

import game_env

from dqn import DQN

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "dqn_final_model.pt"

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

STEP_DELAY = 0.001
# ============================================================
# Action mapping
# ============================================================

ACTIONS = [
    game_env.Action.Up,
    game_env.Action.Down,
    game_env.Action.Left,
    game_env.Action.Right
]


# ============================================================
# Load trained DQN
# ============================================================

device = torch.device("cpu")

model = DQN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Trained DQN loaded successfully!")

# ============================================================
# Create environment
# ============================================================

env = game_env.GameEnvironment(
    WINDOW_WIDTH,
    WINDOW_HEIGHT
)

env.enable_rendering()

# ============================================================
# Run episodes
# ============================================================

episode = 1
no_of_success = 0

while env.is_window_open():

    state_obj = env.reset()

    state = [
        state_obj.dx,
        state_obj.dy
    ]

    done = False

    total_reward = 0.0
    steps = 0

    print(f"\nStarting episode {episode}")

    while not done and env.is_window_open():

        # Handle close button and other window events.
        env.handle_window_events()

        if not env.is_window_open():
            break

        # Convert current state to PyTorch tensor.
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=device
        )

        # Ask trained DQN for Q-values.
        with torch.no_grad():
            q_values = model(state_tensor)

        # Greedy action: epsilon = 0.
        action_index = q_values.argmax().item()

        # Execute action in C++ environment.
        result = env.step(
            ACTIONS[action_index]
        )

        # Update state.
        state = [
            result.nextState.dx,
            result.nextState.dy
        ]

        total_reward += result.reward
        steps += 1

        # Draw player and coin.
        env.render_window()

        # Slow it down enough for us to see.
        time.sleep(STEP_DELAY)

        done = result.done

    if not env.is_window_open():
        break

    success = total_reward > 0

    if success:
        no_of_success += 1

    print(
        f"Episode {episode} finished | "
        f"Success: {success} | "
        f"Steps: {steps} | "
        f"Reward: {total_reward:.3f}"
    )

    # Small pause before resetting.
    time.sleep(0.1)

    episode += 1

print(f"Total successes: {no_of_success}/{episode - 1}")