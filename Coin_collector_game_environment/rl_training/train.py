import os
import random
from collections import deque

import torch

import game_env

from dqn import DQN
from replay_buffer import ReplayBuffer


# ============================================================
# Hyperparameters
# ============================================================

NUM_EPISODES = 500

BATCH_SIZE = 64
REPLAY_CAPACITY = 100_000
MIN_REPLAY_SIZE = 1000

GAMMA = 0.99
LEARNING_RATE = 1e-3

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE_FREQUENCY = 1000

CHECKPOINT_FREQUENCY = 50
PRINT_FREQUENCY = 10

ROLLING_WINDOW_SIZE = 100

EVALUATION_FREQUENCY = 50
NUM_EVALUATION_EPISODES = 100

# Set to True only when you want to resume from a checkpoint.
RESUME_TRAINING = True


# ============================================================
# Paths
# ============================================================

if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working/rl_outputs"
else:
    OUTPUT_DIR = "./rl_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "dqn_checkpoint.pt"
)

FINAL_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "dqn_final_model.pt"
)

# ============================================================
# Device selection
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

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
# Epsilon-greedy action selection
# ============================================================

def select_action(
    state,
    online_network,
    epsilon,
    device
):
    # Exploration
    if random.random() < epsilon:
        return random.randint(0, 3)

    # Exploitation
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        q_values = online_network(state_tensor)

    return q_values.argmax().item()


# ============================================================
# One DQN training step
# ============================================================

def train_step(
    online_network,
    target_network,
    replay_buffer,
    optimizer,
    batch_size,
    gamma,
    min_replay_size,
    device
):
    if len(replay_buffer) < min_replay_size:
        return None

    batch = replay_buffer.sample(batch_size)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(
        states,
        dtype=torch.float32,
        device=device
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long,
        device=device
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32,
        device=device
    )

    next_states = torch.tensor(
        next_states,
        dtype=torch.float32,
        device=device
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Current Q-values from online network
    #
    # states:
    # [batch_size, 2]
    #
    # all_q_values:
    # [batch_size, 4]
    # --------------------------------------------------------

    all_q_values = online_network(states)

    # Select Q(s, a) only for actions actually taken.
    #
    # [batch_size, 4]
    #       ->
    # [batch_size]

    predicted_q_values = all_q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    # --------------------------------------------------------
    # Calculate Bellman targets using target network
    # --------------------------------------------------------

    with torch.no_grad():

        next_q_values = target_network(next_states)

        max_next_q_values = next_q_values.max(
            dim=1
        ).values

        target_q_values = (
            rewards
            + gamma
            * (1.0 - dones)
            * max_next_q_values
        )

    # Huber loss
    loss = torch.nn.functional.smooth_l1_loss(
        predicted_q_values,
        target_q_values
    )

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()


# ============================================================
# Greedy evaluation
# ============================================================

def evaluate_agent(
    env,
    online_network,
    num_episodes,
    device
):
    """
    Evaluate the learned policy with epsilon = 0.

    No exploration.
    No replay-buffer changes.
    No gradient updates.
    No epsilon decay.
    """

    successes = 0
    total_success_steps = 0
    total_rewards = 0.0

    # Remember whether network was in training mode.
    was_training = online_network.training

    online_network.eval()

    for _ in range(num_episodes):

        state_obj = env.reset()

        state = [
            state_obj.dx,
            state_obj.dy
        ]

        done = False
        episode_steps = 0
        episode_reward = 0.0

        while not done:

            action_index = select_action(
                state,
                online_network,
                epsilon=0.0,
                device=device
            )

            result = env.step(
                ACTIONS[action_index]
            )

            state = [
                result.nextState.dx,
                result.nextState.dy
            ]

            episode_steps += 1
            episode_reward += result.reward

            done = result.done

            if result.done and result.reward > 0:
                successes += 1
                total_success_steps += episode_steps

        total_rewards += episode_reward

    # Restore training mode if necessary.
    if was_training:
        online_network.train()

    success_rate = (
        100.0
        * successes
        / num_episodes
    )

    average_success_steps = (
        total_success_steps / successes
        if successes > 0
        else None
    )

    average_reward = (
        total_rewards / num_episodes
    )

    return (
        success_rate,
        average_success_steps,
        average_reward
    )


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path,
    episode,
    online_network,
    target_network,
    optimizer,
    epsilon,
    total_environment_steps,
    total_successes,
    replay_buffer,
    recent_successes,
    recent_rewards
):
    checkpoint = {
        "episode":
            episode,

        "online_network":
            online_network.state_dict(),

        "target_network":
            target_network.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "epsilon":
            epsilon,

        "total_environment_steps":
            total_environment_steps,

        "total_successes":
            total_successes,

        # Save replay experiences as an ordinary list.
        "replay_buffer":
            list(replay_buffer.buffer),

        "recent_successes":
            list(recent_successes),

        "recent_rewards":
            list(recent_rewards)
    }

    torch.save(checkpoint, path)

    print(
        f"\nCheckpoint saved at episode {episode}: "
        f"{path}\n"
    )


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(
    path,
    online_network,
    target_network,
    optimizer,
    replay_buffer,
    recent_successes,
    recent_rewards,
    device
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Restore neural networks and optimizer
    # --------------------------------------------------------

    online_network.load_state_dict(
        checkpoint["online_network"]
    )

    target_network.load_state_dict(
        checkpoint["target_network"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    # --------------------------------------------------------
    # Restore training counters
    # --------------------------------------------------------

    start_episode = (
        checkpoint["episode"] + 1
    )

    epsilon = checkpoint["epsilon"]

    total_environment_steps = checkpoint[
        "total_environment_steps"
    ]

    total_successes = checkpoint[
        "total_successes"
    ]

    # --------------------------------------------------------
    # Restore replay buffer if present
    #
    # Old checkpoints don't contain this field, so use .get().
    # --------------------------------------------------------

    saved_replay_buffer = checkpoint.get(
        "replay_buffer",
        []
    )

    replay_buffer.buffer.extend(
        saved_replay_buffer
    )

    # --------------------------------------------------------
    # Restore rolling statistics if present
    # --------------------------------------------------------

    saved_recent_successes = checkpoint.get(
        "recent_successes",
        []
    )

    saved_recent_rewards = checkpoint.get(
        "recent_rewards",
        []
    )

    recent_successes.extend(
        saved_recent_successes
    )

    recent_rewards.extend(
        saved_recent_rewards
    )

    print("\nCheckpoint loaded successfully.")

    print(
        f"Resuming from episode: "
        f"{start_episode}"
    )

    print(
        f"Current epsilon: "
        f"{epsilon:.4f}"
    )

    print(
        f"Total environment steps: "
        f"{total_environment_steps}"
    )

    print(
        f"Restored replay experiences: "
        f"{len(replay_buffer)}"
    )

    print(
        f"Restored rolling successes: "
        f"{len(recent_successes)}"
    )

    return (
        start_episode,
        epsilon,
        total_environment_steps,
        total_successes
    )


# ============================================================
# Main training loop
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Create C++ environment
    # --------------------------------------------------------

    env = game_env.GameEnvironment(800, 600)


    # --------------------------------------------------------
    # Create online and target networks
    # --------------------------------------------------------

    online_network = DQN().to(device)

    target_network = DQN().to(device)

    # Initially make them identical.
    target_network.load_state_dict(
        online_network.state_dict()
    )

    target_network.eval()


    # --------------------------------------------------------
    # Create optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # Create replay buffer
    # --------------------------------------------------------

    replay_buffer = ReplayBuffer(
        capacity=REPLAY_CAPACITY
    )


    # --------------------------------------------------------
    # Initial training state
    # --------------------------------------------------------

    start_episode = 1

    epsilon = EPSILON_START

    total_environment_steps = 0

    total_successes = 0


    # --------------------------------------------------------
    # Rolling statistics
    # --------------------------------------------------------

    recent_successes = deque(
        maxlen=ROLLING_WINDOW_SIZE
    )

    recent_rewards = deque(
        maxlen=ROLLING_WINDOW_SIZE
    )


    # --------------------------------------------------------
    # Resume from checkpoint if requested
    # --------------------------------------------------------

    if (
        RESUME_TRAINING
        and os.path.exists(CHECKPOINT_PATH)
    ):

        print(
            f"Found checkpoint at: "
            f"{CHECKPOINT_PATH}"
        )

        (
            start_episode,
            epsilon,
            total_environment_steps,
            total_successes

        ) = load_checkpoint(
            CHECKPOINT_PATH,
            online_network,
            target_network,
            optimizer,
            replay_buffer,
            recent_successes,
            recent_rewards,
            device
        )

    elif RESUME_TRAINING:

        print(
            "\nRESUME_TRAINING is True, "
            "but no checkpoint was found."
        )

        print(
            f"Expected checkpoint at: "
            f"{CHECKPOINT_PATH}"
        )

        print(
            "Starting fresh training instead.\n"
        )


    # ========================================================
    # Episode loop
    # ========================================================

    for episode in range(
        start_episode,
        NUM_EPISODES + 1
    ):

        state_obj = env.reset()

        state = [
            state_obj.dx,
            state_obj.dy
        ]

        done = False

        success = False

        episode_reward = 0.0

        episode_steps = 0

        # Store every training loss from this episode.
        episode_losses = []


        # ====================================================
        # Environment interaction loop
        # ====================================================

        while not done:

            # -----------------------------------------------
            # 1. Select epsilon-greedy action
            # -----------------------------------------------

            action_index = select_action(
                state,
                online_network,
                epsilon,
                device
            )

            cpp_action = ACTIONS[action_index]


            # -----------------------------------------------
            # 2. Execute action
            # -----------------------------------------------

            result = env.step(cpp_action)

            next_state = [
                result.nextState.dx,
                result.nextState.dy
            ]


            # -----------------------------------------------
            # 3. Store transition
            # -----------------------------------------------

            replay_buffer.push(
                state,
                action_index,
                result.reward,
                next_state,
                result.done
            )


            # -----------------------------------------------
            # 4. Train online network
            # -----------------------------------------------

            loss = train_step(
                online_network,
                target_network,
                replay_buffer,
                optimizer,
                BATCH_SIZE,
                GAMMA,
                MIN_REPLAY_SIZE,
                device
            )

            if loss is not None:
                episode_losses.append(loss)


            # -----------------------------------------------
            # 5. Update statistics
            # -----------------------------------------------

            episode_reward += result.reward

            episode_steps += 1

            total_environment_steps += 1


            if result.done and result.reward > 0:
                success = True


            # -----------------------------------------------
            # 6. Update target network periodically
            # -----------------------------------------------

            if (
                total_environment_steps
                % TARGET_UPDATE_FREQUENCY
                == 0
            ):
                target_network.load_state_dict(
                    online_network.state_dict()
                )


            # -----------------------------------------------
            # 7. Advance state
            # -----------------------------------------------

            state = next_state

            done = result.done


        # ====================================================
        # Episode finished
        # ====================================================

        if success:
            total_successes += 1


        recent_successes.append(
            1 if success else 0
        )

        recent_rewards.append(
            episode_reward
        )


        # ----------------------------------------------------
        # Average loss across all training batches this episode
        # ----------------------------------------------------

        average_episode_loss = (
            sum(episode_losses)
            / len(episode_losses)
            if episode_losses
            else None
        )


        # ----------------------------------------------------
        # Decay epsilon
        # ----------------------------------------------------

        epsilon = max(
            EPSILON_END,
            epsilon * EPSILON_DECAY
        )


        # ----------------------------------------------------
        # Calculate statistics
        # ----------------------------------------------------

        overall_success_rate = (
            100.0
            * total_successes
            / episode
        )

        recent_success_rate = (
            100.0
            * sum(recent_successes)
            / len(recent_successes)
        )

        recent_average_reward = (
            sum(recent_rewards)
            / len(recent_rewards)
        )


        # ----------------------------------------------------
        # Print training progress
        # ----------------------------------------------------

        if episode % PRINT_FREQUENCY == 0:

            if average_episode_loss is None:
                loss_text = "N/A"
            else:
                loss_text = (
                    f"{average_episode_loss:.6f}"
                )

            success_text = (
                "Yes" if success else "No"
            )

            print(
                f"Episode: {episode:5d} | "
                f"Success: {success_text:3s} | "
                f"Reward: {episode_reward:8.3f} | "
                f"Steps: {episode_steps:4d} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Overall Success: "
                f"{overall_success_rate:6.2f}% | "
                f"Recent Success: "
                f"{recent_success_rate:6.2f}% | "
                f"Recent Avg Reward: "
                f"{recent_average_reward:8.3f} | "
                f"Avg Loss: {loss_text}"
            )


        # ----------------------------------------------------
        # Greedy evaluation
        # ----------------------------------------------------

        if episode % EVALUATION_FREQUENCY == 0:

            (
                eval_success_rate,
                eval_avg_steps,
                eval_avg_reward

            ) = evaluate_agent(
                env,
                online_network,
                NUM_EVALUATION_EPISODES,
                device
            )

            if eval_avg_steps is None:
                eval_steps_text = "N/A"
            else:
                eval_steps_text = (
                    f"{eval_avg_steps:.2f}"
                )

            print(
                f"\n{'=' * 70}"
            )

            print(
                f"GREEDY EVALUATION AT EPISODE {episode}"
            )

            print(
                f"Evaluation episodes: "
                f"{NUM_EVALUATION_EPISODES}"
            )

            print(
                f"Success rate: "
                f"{eval_success_rate:.2f}%"
            )

            print(
                f"Average successful steps: "
                f"{eval_steps_text}"
            )

            print(
                f"Average reward: "
                f"{eval_avg_reward:.3f}"
            )

            print(
                f"{'=' * 70}\n"
            )


        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        if episode % CHECKPOINT_FREQUENCY == 0:

            save_checkpoint(
                CHECKPOINT_PATH,
                episode,
                online_network,
                target_network,
                optimizer,
                epsilon,
                total_environment_steps,
                total_successes,
                replay_buffer,
                recent_successes,
                recent_rewards
            )


    # ========================================================
    # Training complete
    # ========================================================

    torch.save(
        online_network.state_dict(),
        FINAL_MODEL_PATH
    )

    print("\nTraining complete.")

    print(
        f"Final model saved to: "
        f"{FINAL_MODEL_PATH}"
    )
