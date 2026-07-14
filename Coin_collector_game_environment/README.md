# C++ Game Environment with Deep Q-Learning (DQN)

A complete reinforcement learning project that combines a custom **C++/SFML game environment** with a **Deep Q-Network (DQN) implemented in Python using PyTorch**.

**The created Environment is also available in Kaggle at https://www.kaggle.com/datasets/vigneshwarsk/c-rl-coin-collection-environment which you can include in a notebook and move to working directory to train.**
**I have trained this model on 500 episodes only which gave an accuracy of about 88%**

The project demonstrates the complete pipeline:

```text
C++ Game
    ↓
RL-Compatible Environment
    ↓
pybind11 C++ ↔ Python Bridge
    ↓
Python DQN Agent
    ↓
Experience Replay + Target Network
    ↓
Training Locally or on Kaggle
    ↓
Saved PyTorch Model
    ↓
Visualize the Trained Agent in SFML
```

The initial environment is intentionally simple: a player must navigate through a 2D continuous game world and collect a randomly positioned coin.

Despite the simplicity of the game, the project establishes a reusable architecture for progressively more complex reinforcement learning environments involving obstacles, enemies, multiple objectives, larger state spaces, and more advanced RL algorithms.

---

## 1. Project Goals

The main goals of this project are to:

* Learn C++ game development using SFML.
* Understand the structure of a real-time game loop.
* Convert a normal playable game into an RL-compatible environment.
* Understand states, actions, rewards, and terminal conditions.
* Connect a C++ environment to Python using pybind11.
* Implement Deep Q-Learning from scratch using PyTorch.
* Use experience replay and target networks.
* Train the agent locally or using Kaggle.
* Save and restore trained models and training checkpoints.
* Visually watch the trained DQN control the C++ game.

The final architecture is:

```text
                        ┌───────────────────────┐
                        │     C++ Game Code     │
                        │  Player, Coin, SFML   │
                        └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │    GameEnvironment    │
                        │                       │
                        │ reset()               │
                        │ step(action)          │
                        │ getState()            │
                        │ reward                │
                        │ termination           │
                        └───────────┬───────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
          ┌───────────────────┐            ┌───────────────────┐
          │   Normal C++ Use  │            │    pybind11       │
          │                   │            │   Python Bridge   │
          │ main_game.cpp     │            └─────────┬─────────┘
          │ main_rl.cpp       │                      │
          └───────────────────┘                      ▼
                                          ┌───────────────────┐
                                          │   Python DQN      │
                                          │                   │
                                          │ Online Network    │
                                          │ Target Network    │
                                          │ Replay Buffer     │
                                          │ Bellman Learning  │
                                          └─────────┬─────────┘
                                                    │
                                                    ▼
                                          ┌───────────────────┐
                                          │   Trained Model   │
                                          │                   │
                                          │ .pt file          │
                                          └─────────┬─────────┘
                                                    │
                                                    ▼
                                          ┌───────────────────┐
                                          │  watch_agent.py   │
                                          │                   │
                                          │ Visual SFML Agent │
                                          └───────────────────┘
```

---

## 2. The Dual Purpose of the Project

A central design decision in this project is that the same game code serves two different purposes:

1. A normal human-playable C++ game.
2. A reinforcement learning environment controlled by an AI agent.

The core objects such as `Player`, `Coin`, and `GameEnvironment` are shared.

This prevents us from creating one implementation for human gameplay and an entirely separate implementation for reinforcement learning.

### Human gameplay

For human gameplay:

```text
Keyboard Input
      ↓
GameEnvironment
      ↓
Player Movement
      ↓
Collision Detection
      ↓
SFML Rendering
```

The human player controls the character using keyboard input.

### Reinforcement learning

For RL:

```text
DQN
 ↓
Action Index
 ↓
GameEnvironment::step(action)
 ↓
Next State + Reward + Done
 ↓
Replay Buffer
 ↓
DQN Training
```

The AI does not press physical keyboard keys. Instead, it sends abstract actions such as:

```text
Up
Down
Left
Right
```

directly to the environment.

This separation is important. The game logic should not depend entirely on keyboard input because an RL agent needs to control the same player programmatically.

---

## 3. Project Structure

The project is organized approximately as follows:

```text
first_game/
│
├── include/
│   ├── Player.hpp
│   ├── Coin.hpp
│   └── GameEnvironment.hpp
│
├── src/
│   ├── Player.cpp
│   ├── Coin.cpp
│   ├── GameEnvironment.cpp
│   └── bindings.cpp
│
├── rl_training/
│   ├── dqn.py
│   ├── replay_buffer.py
│   ├── train.py
│   ├── watch_agent.py
│   ├── game_env.cpython-312-x86_64-linux-gnu.so
│   └── dqn_final_model.pt
│
├── main_game.cpp
├── main_rl.cpp
└── README.md
```

The exact Python version in the `.so` filename may differ. For example:

```text
game_env.cpython-310-x86_64-linux-gnu.so
game_env.cpython-311-x86_64-linux-gnu.so
game_env.cpython-312-x86_64-linux-gnu.so
```

The extension is generated according to the active Python installation.

---

## 4. Purpose of Each File

### `include/Player.hpp`

Declares the `Player` class.

The player is the controllable game object. It is responsible for information such as:

* Position.
* Shape.
* Movement speed.
* Movement operations.
* Keeping the player within environment boundaries.
* Drawing itself when rendering is enabled.

Conceptually:

```cpp
class Player
{
private:
    sf::RectangleShape shape;
    float speed;

public:
    void move(float dx, float dy);
    void keepInsideBounds(float width, float height);
    void draw(sf::RenderWindow& window);
};
```

The exact implementation may differ depending on the current version of the project.

---

### `src/Player.cpp`

Contains the implementation of the methods declared in `Player.hpp`.

For example:

```cpp
void Player::move(float dx, float dy)
{
    shape.move(dx, dy);
}
```

The important design idea is that `Player` knows **how to move**, but it should not necessarily decide **why it moves**.

The movement command could originate from:

```text
Keyboard
Random Agent
DQN Agent
Future RL Algorithm
```

This separation makes the player reusable.

---

### `include/Coin.hpp`

Declares the `Coin` class.

The coin represents the objective that the agent must collect.

Its responsibilities include:

* Storing its position.
* Storing its SFML shape.
* Repositioning itself.
* Drawing itself when rendering is enabled.

The coin is usually randomly repositioned during:

```cpp
env.reset();
```

so that the agent must learn a general navigation policy rather than memorize one fixed target location.

---

### `src/Coin.cpp`

Contains the implementation of the `Coin` class.

This includes logic such as:

* Creating the coin shape.
* Setting its position.
* Randomizing its position.
* Rendering it.

---

### `include/GameEnvironment.hpp`

Declares the central `GameEnvironment` class.

This is the most important class in the project.

It acts as both:

```text
A normal game controller
        +
An RL-compatible environment
```

It owns or manages:

```text
Player
Coin
Window dimensions
Current episode step
Maximum episode steps
Optional SFML rendering window
```

The RL interface consists primarily of:

```cpp
State reset();

State getState() const;

StepResult step(Action action);
```

The environment also defines the state, actions, and step results.

For example:

```cpp
struct State
{
    float dx;
    float dy;
};
```

The action space is:

```cpp
enum class Action
{
    Up,
    Down,
    Left,
    Right
};
```

The result of one environment interaction is:

```cpp
struct StepResult
{
    State nextState;
    float reward;
    bool done;
};
```

Thus, one action produces:

```text
Action
  ↓
Environment Transition
  ↓
Next State
Reward
Done
```

This corresponds to the standard RL transition:

```text
(state, action, reward, next_state, done)
```

or mathematically:

[
(s,a,r,s',d)
]

---

### `src/GameEnvironment.cpp`

Implements the game and RL environment logic.

Its major responsibilities include:

* Resetting episodes.
* Reading the current state.
* Applying actions.
* Moving the player.
* Checking collisions.
* Calculating rewards.
* Detecting episode termination.
* Keeping the player inside the game boundaries.
* Optionally rendering the environment.

The central function is:

```cpp
StepResult GameEnvironment::step(Action action);
```

Conceptually:

```text
Receive action
      ↓
Move player
      ↓
Keep player inside boundaries
      ↓
Increment step counter
      ↓
Check player-coin collision
      ↓
Calculate reward
      ↓
Check termination
      ↓
Calculate next state
      ↓
Return StepResult
```

---

### `main_game.cpp`

This is the normal human-playable version of the game.

Its purpose is to test and play the environment manually.

The control flow is approximately:

```text
Create SFML Window
        ↓
Create GameEnvironment
        ↓
Game Loop
        ↓
Handle Window Events
        ↓
Read Keyboard Input
        ↓
Move Player
        ↓
Update Game
        ↓
Render Player and Coin
        ↓
Repeat
```

This executable is useful for:

* Testing the game manually.
* Verifying player movement.
* Verifying boundaries.
* Testing collision detection.
* Checking visual appearance.
* Debugging game logic before involving reinforcement learning.

A fundamental principle used in this project is:

> First verify that the game works manually. Then expose it as an RL environment.

---

### `main_rl.cpp`

This is the C++-only test program for the RL interface.

Its purpose is **not DQN training**.

Instead, it tests whether the environment behaves correctly when controlled programmatically.

For example, it can:

```text
Reset environment
      ↓
Select random action
      ↓
Call env.step(action)
      ↓
Read next state
      ↓
Read reward
      ↓
Read done
      ↓
Repeat
```

This creates a random-walk agent.

The purpose of `main_rl.cpp` is to verify that:

* `reset()` works correctly.
* `step(action)` works correctly.
* All four actions work.
* Rewards are correct.
* State transitions are correct.
* Episodes terminate correctly.
* The environment can run without keyboard input.

It also provides a random-policy baseline.

In our experiment, the random agent achieved approximately:

```text
3.6% success rate
```

This became the baseline against which the DQN was compared.

---

### `src/bindings.cpp`

Creates the pybind11 bridge between C++ and Python.

Without this file, the PyTorch DQN written in Python cannot directly interact with the C++ `GameEnvironment`.

The bridge exposes:

```text
C++ Action enum
C++ State struct
C++ StepResult struct
C++ GameEnvironment class
```

to Python.

A representative binding is:

```cpp
#include <pybind11/pybind11.h>

#include "GameEnvironment.hpp"

namespace py = pybind11;

PYBIND11_MODULE(game_env, m)
{
    py::enum_<Action>(m, "Action")
        .value("Up", Action::Up)
        .value("Down", Action::Down)
        .value("Left", Action::Left)
        .value("Right", Action::Right);

    py::class_<State>(m, "State")
        .def_readonly("dx", &State::dx)
        .def_readonly("dy", &State::dy);

    py::class_<StepResult>(m, "StepResult")
        .def_readonly(
            "nextState",
            &StepResult::nextState
        )
        .def_readonly(
            "reward",
            &StepResult::reward
        )
        .def_readonly(
            "done",
            &StepResult::done
        );

    py::class_<GameEnvironment>(
        m,
        "GameEnvironment"
    )
        .def(py::init<int, int>())
        .def("reset", &GameEnvironment::reset)
        .def("getState", &GameEnvironment::getState)
        .def("step", &GameEnvironment::step)
        .def(
            "enable_rendering",
            &GameEnvironment::enableRendering
        )
        .def(
            "render",
            &GameEnvironment::renderWindow
        )
        .def(
            "is_window_open",
            &GameEnvironment::isWindowOpen
        )
        .def(
            "handle_window_events",
            &GameEnvironment::handleWindowEvents
        );
}
```

After compilation, Python can use:

```python
import game_env

env = game_env.GameEnvironment(800, 600)

state = env.reset()

result = env.step(game_env.Action.Right)
```

Even though Python syntax is being used, the actual environment logic is executed in compiled C++.

---

### `rl_training/dqn.py`

Defines the Deep Q-Network.

The state is:

```text
[dx, dy]
```

so the input dimension is:

```text
2
```

The action space is:

```text
Up
Down
Left
Right
```

so the output dimension is:

```text
4
```

The architecture used is approximately:

```text
2 inputs
   ↓
64 neurons + ReLU
   ↓
64 neurons + ReLU
   ↓
4 Q-values
```

For one state, the network outputs:

```text
Q(Up)
Q(Down)
Q(Left)
Q(Right)
```

The greedy action is:

```python
action_index = q_values.argmax().item()
```

---

### `rl_training/replay_buffer.py`

Implements experience replay.

Every interaction generates a transition:

```text
(state, action, reward, next_state, done)
```

These transitions are stored in the replay buffer.

During training, a random mini-batch is sampled:

```python
batch = replay_buffer.sample(batch_size)
```

For our experiment:

```text
Batch size = 64
```

Therefore, 64 past transitions are used in one DQN update.

Experience replay helps:

* Reduce correlation between consecutive samples.
* Reuse past experiences.
* Improve training stability.
* Allow successful experiences to influence multiple future updates.

---

### `rl_training/train.py`

Contains the complete DQN training loop.

Its responsibilities include:

* Creating the C++ environment.
* Creating the online DQN.
* Creating the target DQN.
* Creating the optimizer.
* Creating the replay buffer.
* Epsilon-greedy action selection.
* Collecting experiences.
* Sampling mini-batches.
* Calculating Bellman targets.
* Computing the Huber loss.
* Running backpropagation.
* Updating the target network.
* Evaluating the greedy policy.
* Saving checkpoints.
* Saving the final model.

The core interaction is:

```text
Observe state
      ↓
Choose epsilon-greedy action
      ↓
env.step(action)
      ↓
Receive:
    next_state
    reward
    done
      ↓
Store transition
      ↓
Sample replay batch
      ↓
Train DQN
      ↓
Repeat
```

---

### `rl_training/watch_agent.py`

Loads the trained model and visually watches the agent play.

This script does **not train** the network.

It performs inference only.

The flow is:

```text
Load dqn_final_model.pt
        ↓
Create C++ GameEnvironment
        ↓
Enable SFML rendering
        ↓
Read current [dx, dy]
        ↓
Pass state through DQN
        ↓
Obtain four Q-values
        ↓
Select argmax action
        ↓
Call C++ env.step(action)
        ↓
Render updated game
        ↓
Repeat
```

This closes the complete loop:

```text
Train Agent
    ↓
Save Model
    ↓
Load Model
    ↓
Use Model to Control C++ Game
    ↓
Watch Agent Play
```

---

### `dqn_final_model.pt`

Contains the trained parameters of the final online DQN.

It is used for:

* Inference.
* Evaluation.
* Watching the trained agent.
* Deploying the learned policy.

It can be loaded with:

```python
model.load_state_dict(
    torch.load(
        "dqn_final_model.pt",
        map_location=device,
        weights_only=True
    )
)
```

---

### `dqn_checkpoint.pt`

When present, this is different from the final model.

A checkpoint can contain:

```text
Online-network parameters
Target-network parameters
Optimizer state
Current episode
Current epsilon
Environment-step count
Total successes
Replay buffer
Recent statistics
```

Use:

```text
dqn_final_model.pt
```

for inference and watching the agent.

Use:

```text
dqn_checkpoint.pt
```

for resuming training.

---

## 5. Reinforcement Learning Formulation

The game is converted into an RL problem by defining:

```text
State
Action
Reward
Transition
Termination
```

### State space

The state is:

[
s =
\begin{bmatrix}
dx \
dy
\end{bmatrix}
]

where `dx` and `dy` represent the relative position of the coin with respect to the player.

This is much better than directly using:

```text
player_x
player_y
coin_x
coin_y
```

for this simple game because the important information is the displacement from the player to the target.

For example:

```text
dx > 0 → Coin is to one horizontal side
dx < 0 → Coin is to the opposite horizontal side

dy > 0 → Coin is to one vertical side
dy < 0 → Coin is to the opposite vertical side
```

The exact interpretation of positive and negative `dy` depends on the SFML coordinate system and the implementation.

### Action space

The action space is discrete:

[
\mathcal{A}
===========

{
\text{Up},
\text{Down},
\text{Left},
\text{Right}
}.
]

The Python mapping is:

```python
ACTIONS = [
    game_env.Action.Up,
    game_env.Action.Down,
    game_env.Action.Left,
    game_env.Action.Right
]
```

Therefore:

```text
0 → Up
1 → Down
2 → Left
3 → Right
```

### Reward

The experiment used a small negative step reward and a large positive success reward.

Conceptually:

```text
Normal step:
    reward = -0.01

Coin collected:
    reward = +10
```

Therefore, the agent is encouraged to:

1. Collect the coin.
2. Reach it in fewer steps.

For example, a successful episode with 168 total steps produced approximately:

```text
Reward = 8.330
```

because:

[
10-167(0.01)=8.33.
]

### Episode termination

An episode terminates when:

* The player collects the coin, or
* The maximum number of allowed steps is reached.

The maximum used in the initial environment was:

```text
500 steps
```

---

## 6. Requirements

The project was developed on Ubuntu using:

* C++17.
* SFML.
* Python 3.
* pybind11.
* PyTorch.

Install the C++ dependencies:

```bash
sudo apt update
sudo apt install build-essential libsfml-dev python3-dev python3-pip
```

Install pybind11:

```bash
python3 -m pip install pybind11
```

Install PyTorch according to your system configuration.

For a basic CPU installation, depending on your Python environment:

```bash
python3 -m pip install torch
```

Verify pybind11:

```bash
python3 -m pybind11 --includes
```

Verify SFML:

```bash
pkg-config --modversion sfml-graphics
```

Verify PyTorch:

```bash
python3 -c "import torch; print(torch.__version__)"
```

---

## 7. Compiling the Human-Playable Game

From the project root:

```bash
g++ -std=c++17 \
    main_game.cpp \
    src/GameEnvironment.cpp \
    src/Player.cpp \
    src/Coin.cpp \
    -Iinclude \
    -o game \
    -lsfml-graphics \
    -lsfml-window \
    -lsfml-system
```

Run it:

```bash
./game
```

This version is intended for human interaction and manual testing.

---

## 8. Compiling the C++ Random-Walk RL Test

Compile:

```bash
g++ -std=c++17 \
    main_rl.cpp \
    src/GameEnvironment.cpp \
    src/Player.cpp \
    src/Coin.cpp \
    -Iinclude \
    -o rl_test \
    -lsfml-graphics \
    -lsfml-window \
    -lsfml-system
```

Run:

```bash
./rl_test
```

This tests the RL environment using programmatically selected actions without involving PyTorch.

Its purpose is to verify the environment before adding the complexity of neural-network training.

---

## 9. Compiling the C++ Environment for Python

From the project root:

```bash
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    -Iinclude \
    src/bindings.cpp \
    src/GameEnvironment.cpp \
    src/Player.cpp \
    src/Coin.cpp \
    -o rl_training/game_env$(python3-config --extension-suffix) \
    -lsfml-graphics \
    -lsfml-window \
    -lsfml-system
```

This creates something similar to:

```text
rl_training/game_env.cpython-312-x86_64-linux-gnu.so
```

Test it:

```bash
cd rl_training
python3 -c "import game_env; print('game_env imported successfully')"
```

Expected output:

```text
game_env imported successfully
```

---

## 10. Testing the C++ Environment from Python

From `rl_training/`, open Python:

```bash
python3
```

Then:

```python
import game_env

env = game_env.GameEnvironment(800, 600)

state = env.reset()

print(state.dx)
print(state.dy)

result = env.step(game_env.Action.Right)

print(result.nextState.dx)
print(result.nextState.dy)
print(result.reward)
print(result.done)
```

If this works, the complete bridge is functional:

```text
Python
  ↓
pybind11
  ↓
C++ GameEnvironment
  ↓
State + Reward + Done
  ↓
Python
```

---

## 11. Training the DQN Locally

From the project root:

```bash
cd rl_training
python3 train.py
```

The training script automatically:

1. Creates the C++ environment.
2. Creates the online network.
3. Creates the target network.
4. Creates the replay buffer.
5. Begins epsilon-greedy exploration.
6. Stores transitions.
7. Samples mini-batches.
8. Calculates Bellman targets.
9. Updates the online network.
10. Periodically updates the target network.
11. Evaluates the greedy policy.
12. Saves checkpoints and the final model.

Example training output:

```text
Episode:   100
Reward:    8.330
Steps:       168
Epsilon:  0.6058
Overall Success: 28.00%
Recent Avg Reward: -1.328
```

---

## 12. DQN Training Details

The main hyperparameters used were approximately:

```python
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

EVALUATION_FREQUENCY = 50
NUM_EVALUATION_EPISODES = 100
```

### Online network

The online network is trained through gradient descent.

It predicts:

[
Q_{\text{online}}(s,a).
]

### Target network

The target network provides relatively stable Bellman targets:

[
y
=

r+
\gamma(1-d)
\max_{a'}
Q_{\text{target}}(s',a').
]

Periodically:

[
\theta_{\text{target}}
\leftarrow
\theta_{\text{online}}.
]

### Mini-batch training

For a batch of 64 states:

```text
Input:
    [64, 2]
```

The online network outputs:

```text
[64, 4]
```

because every state gets four Q-values.

Only the Q-value corresponding to the action actually taken is selected:

```text
[64, 4]
    ↓ gather()
[64]
```

The target network processes the next states:

```text
[64, 2]
    ↓
[64, 4]
    ↓ max over actions
[64]
```

Thus, the loss compares:

```text
Predicted Q(s, a): [64]
Bellman targets:   [64]
```

---

## 13. Running on Kaggle

Training was moved to Kaggle because longer local runs caused significant laptop heating.

The uploaded project contained:

```text
first_game/
├── include/
├── src/
└── rl_training/
```

The C++ source code should be uploaded rather than relying only on a locally compiled `.so`, because compiled Python extensions depend on:

* Python version.
* Operating system.
* CPU architecture.
* Compiler ABI.
* Installed libraries.

Therefore, the robust workflow is:

```text
Upload source code
       ↓
Copy to /kaggle/working
       ↓
Install dependencies
       ↓
Compile C++ extension on Kaggle
       ↓
Test environment
       ↓
Train DQN
```

---

## 14. Kaggle Setup

### Copy the project to the writable working directory

Kaggle inputs are effectively read-only.

Use:

```python
import os
import shutil

SOURCE = (
    "/kaggle/input/datasets/vigneshwarsk/"
    "c-rl-coin-collection-environment/"
    "first_game (Copy)"
)

PROJECT = "/kaggle/working/first_game"

shutil.copytree(
    SOURCE,
    PROJECT,
    dirs_exist_ok=True
)

print("Project copied successfully!")
print(os.listdir(PROJECT))
```

Adjust the source path if the Kaggle dataset name changes.

### Install pybind11

```python
!python3 -m pip install -q pybind11
```

### Install SFML

```python
!apt-get update -qq
!apt-get install -y libsfml-dev
```

### Compile the Python extension

On Kaggle, `python3-config` may not be available.

The error encountered was:

```text
/bin/bash: line 1: python3-config: command not found
```

Instead, use Python's `sysconfig` module:

```python
!c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    -I"/kaggle/working/first_game/include" \
    "/kaggle/working/first_game/src/bindings.cpp" \
    "/kaggle/working/first_game/src/GameEnvironment.cpp" \
    "/kaggle/working/first_game/src/Player.cpp" \
    "/kaggle/working/first_game/src/Coin.cpp" \
    -o "/kaggle/working/first_game/rl_training/game_env$(python3 -c \
'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')" \
    -lsfml-graphics \
    -lsfml-window \
    -lsfml-system
```

### Import the module

```python
import sys

TRAINING_DIR = (
    "/kaggle/working/"
    "first_game/rl_training"
)

if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)

import game_env

print("Module imported successfully!")
```

### Test the environment

```python
env = game_env.GameEnvironment(800, 600)

state = env.reset()

print("Initial state:")
print("dx =", state.dx)
print("dy =", state.dy)

result = env.step(game_env.Action.Right)

print("Next state:")
print("dx =", result.nextState.dx)
print("dy =", result.nextState.dy)
print("Reward:", result.reward)
print("Done:", result.done)
```

### Start training

```python
%cd /kaggle/working/first_game/rl_training
!python3 -u train.py
```

The `-u` option uses unbuffered output, making training progress appear immediately in the notebook.

---

## 15. Experimental Results

The random-agent baseline achieved approximately:

```text
3.6% success
```

An early laptop run reached approximately:

```text
36% success by episode 80
```

One Kaggle run reached:

```text
8% success by episode 100
```

Another reached:

```text
28% success by episode 100
```

The final 500-episode experiment achieved approximately:

```text
Recent training success:      93%
Overall training success:     71%
Final greedy evaluation:      88%
```

The progression demonstrates the stochastic nature of reinforcement learning.

Different runs can produce different early results because of:

* Random network initialization.
* Random starting positions.
* Random coin positions.
* Random exploratory actions.
* Random replay-buffer sampling.
* Different timing of early successful experiences.

The key comparison is:

```text
Random baseline:         ~3.6%
Trained greedy policy:     88%
```

---

## 16. Greedy Evaluation

Training performance alone does not accurately measure the learned policy because training uses epsilon-greedy exploration.

For example, at episode 100:

```text
epsilon ≈ 0.6058
```

This means approximately 60.58% of action selections are exploratory.

Therefore, separate greedy evaluation is performed with:

```text
epsilon = 0
```

Every action is then:

```python
action = q_values.argmax().item()
```

This directly tests the policy learned by the network.

---

## 17. Checkpoints and Model Files

### Final model

```text
dqn_final_model.pt
```

Contains only the trained online-network parameters.

Use it for:

* Watching the agent.
* Inference.
* Evaluation.

### Checkpoint

```text
dqn_checkpoint.pt
```

Can contain:

```text
Episode number
Online network
Target network
Optimizer state
Epsilon
Environment-step count
Success statistics
Replay buffer
Recent reward history
```

Use it for resuming training.

A useful rule is:

```text
Want to play/evaluate?  → dqn_final_model.pt

Want to continue training? → dqn_checkpoint.pt
```

---

## 18. Watching the Trained Agent Play

Place the trained model inside:

```text
rl_training/dqn_final_model.pt
```

Make sure the C++ Python extension has been compiled.

Then:

```bash
cd rl_training
python3 watch_agent.py
```

The script:

1. Loads the trained DQN.
2. Creates the C++ environment.
3. Enables SFML rendering.
4. Resets the environment.
5. Gets the current `[dx, dy]` state.
6. Passes it through the neural network.
7. Selects the action with the largest Q-value.
8. Calls the C++ environment's `step()` method.
9. Renders the new frame.
10. Repeats until the episode ends.

The runtime pipeline is:

```text
Player + Coin Positions
          ↓
       [dx, dy]
          ↓
    PyTorch DQN
          ↓
┌───────────────────────────┐
│ Q(Up)                     │
│ Q(Down)                   │
│ Q(Left)                   │
│ Q(Right)                  │
└───────────────────────────┘
          ↓
        argmax
          ↓
    Selected Action
          ↓
 C++ env.step(action)
          ↓
    Player Movement
          ↓
     SFML Rendering
          ↓
        Repeat
```

---

## 19. Optional Rendering Design

The environment is designed to remain headless during training.

This is important because rendering every frame during hundreds or thousands of episodes would waste computational resources.

The environment therefore stores an optional SFML window:

```cpp
std::unique_ptr<sf::RenderWindow> window;
```

Normally:

```text
window == nullptr
```

During training:

```text
No window
No rendering
Maximum simulation speed
```

When watching the agent:

```python
env.enable_rendering()
```

creates the SFML window.

The relevant methods are:

```cpp
void enableRendering();

void renderWindow();

bool isWindowOpen() const;

void handleWindowEvents();
```

These are exposed to Python as:

```python
env.enable_rendering()
env.render()
env.is_window_open()
env.handle_window_events()
```

---

## 20. Important Rendering Note

During development, an overloaded rendering design caused a pybind11 compilation ambiguity:

```cpp
void render(sf::RenderWindow& window);
void render();
```

Binding this:

```cpp
.def("render", &GameEnvironment::render)
```

was ambiguous because C++ could not determine which overload was intended.

The cleaner solution was to use a uniquely named C++ method:

```cpp
void renderWindow();
```

and expose it to Python as:

```cpp
.def("render", &GameEnvironment::renderWindow)
```

Thus:

```python
env.render()
```

calls:

```cpp
GameEnvironment::renderWindow()
```

The Python and C++ function names do not need to be identical.

---

## 21. Rebuilding After C++ Changes

Whenever any of these files change:

```text
GameEnvironment.hpp
GameEnvironment.cpp
Player.hpp
Player.cpp
Coin.hpp
Coin.cpp
bindings.cpp
```

the `.so` Python extension must be recompiled.

First remove the old extension:

```bash
rm -f rl_training/game_env*.so
```

Then compile again:

```bash
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    -Iinclude \
    src/bindings.cpp \
    src/GameEnvironment.cpp \
    src/Player.cpp \
    src/Coin.cpp \
    -o rl_training/game_env$(python3-config --extension-suffix) \
    -lsfml-graphics \
    -lsfml-window \
    -lsfml-system
```

Then verify that the module imports:

```bash
cd rl_training
python3 -c "import game_env; print('Import successful')"
```

---

## 22. Common Errors

### `ModuleNotFoundError: No module named 'game_env'`

Possible causes:

* The `.so` file has not been compiled.
* Python is running from the wrong directory.
* The compiled extension is incompatible with the active Python version.

Check:

```bash
ls rl_training/game_env*.so
```

---

### `python3-config: command not found`

This occurred on Kaggle.

Use:

```bash
python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'
```

instead of:

```bash
python3-config --extension-suffix
```

---

### `undefined symbol`

For example:

```text
undefined symbol:
_ZN15GameEnvironment6renderERN2sf12RenderWindowE
```

This means the compiled shared library references a C++ function whose matching implementation was not linked.

Check for:

* A function declared in a header but never defined.
* A mismatched function signature.
* A source file missing from the compilation command.
* An outdated `.so` that needs recompilation.

After fixing the source:

```bash
rm -f rl_training/game_env*.so
```

and rebuild.

---

### `unresolved overloaded function type`

This can occur with pybind11 when binding an overloaded C++ function:

```cpp
.def("render", &GameEnvironment::render)
```

If several `render()` overloads exist, pybind11 cannot automatically determine which one is intended.

Use unique method names where possible.

For this project:

```cpp
void renderWindow();
```

was chosen for the Python visualization interface.

---

## 23. Recommended Workflow for Future Development

When adding a new feature, use this order:

```text
1. Modify the C++ game objects
            ↓
2. Test manually with main_game.cpp
            ↓
3. Update GameEnvironment if necessary
            ↓
4. Test programmatically with main_rl.cpp
            ↓
5. Update pybind11 bindings if the Python interface changed
            ↓
6. Recompile game_env.so
            ↓
7. Test from Python
            ↓
8. Train or evaluate the RL agent
            ↓
9. Watch the trained policy visually
```

This layered debugging strategy is extremely useful.

If something fails, it helps identify whether the problem lies in:

```text
Game logic?
RL environment?
C++/Python binding?
Python training loop?
Neural network?
Saved model?
Rendering?
```

---

## 24. Future Improvements

This project can be extended gradually.

Possible next steps include:

* Save the best model based on greedy evaluation score.
* Add multiple coins.
* Add static obstacles.
* Add collision penalties.
* Add enemies.
* Add health.
* Add multiple objectives.
* Increase the complexity of the state space.
* Use ray-based observations.
* Use image observations.
* Use CNN-based DQN.
* Implement Double DQN.
* Implement Dueling DQN.
* Add Prioritized Experience Replay.
* Move toward continuous action spaces.
* Experiment with PPO, SAC, or TD3 for suitable environments.

A sensible progression is:

```text
Current Project:
Player + One Coin
        ↓
Multiple Coins
        ↓
Static Obstacles
        ↓
Moving Enemies
        ↓
Partial Observability
        ↓
Image-Based State
        ↓
CNN + DQN
        ↓
More Advanced RL Algorithms
```

---

## 25. Key Lessons

This project demonstrates several important principles.

### Separate action execution from input source

The player should be able to receive movement commands from:

```text
Keyboard
Random policy
DQN
Future agents
```

The game logic should not be tightly coupled only to keyboard input.

### Keep training headless

Rendering is useful for debugging and demonstration, but unnecessary during high-speed training.

### Test the environment before testing the DQN

Use `main_rl.cpp` to verify:

```text
reset()
step()
state
reward
done
```

before adding neural-network complexity.

### Establish a random baseline

The random policy achieved approximately:

```text
3.6%
```

This made the trained policy's:

```text
88%
```

performance meaningful.

### Training success is different from greedy evaluation

During training, the agent deliberately takes random exploratory actions.

Therefore:

```text
Training success ≠ Pure policy performance
```

Always evaluate separately with:

```text
epsilon = 0
```

### RL is stochastic

Different runs may learn at very different speeds, especially when rewards are sparse.

### Save checkpoints

Cloud environments such as Kaggle should not be treated as permanent local storage.

Save and download important checkpoints and final models.

### Save the best model in future experiments

DQN evaluation can fluctuate significantly.

A future training loop should save:

```text
best_model.pt
```

whenever the greedy evaluation score exceeds the previous best score.

---

## 26. Final Results

The complete first experiment achieved:

```text
Environment:
    Custom C++/SFML coin-collection game

State:
    [dx, dy]

Actions:
    Up, Down, Left, Right

Algorithm:
    Deep Q-Network

Network:
    2 → 64 → 64 → 4

Replay Buffer:
    100,000 capacity

Batch Size:
    64

Discount Factor:
    0.99

Training:
    500 episodes

Random Baseline:
    ~3.6% success

Final Greedy Evaluation:
    88% success

Recent Training Success:
    93%
```

The complete pipeline is now:

```text
C++ Game Development
        ↓
RL Environment Design
        ↓
State / Action / Reward Definition
        ↓
Random Policy Testing
        ↓
pybind11 C++ ↔ Python Integration
        ↓
PyTorch DQN
        ↓
Experience Replay
        ↓
Target Network
        ↓
Bellman Learning
        ↓
Kaggle Cloud Training
        ↓
500 Episodes
        ↓
88% Greedy Success
        ↓
Saved .pt Model
        ↓
SFML Visualization of the Trained Agent
```

---

## 27. Quick Start Summary

For a fresh local setup:

```bash
# Install dependencies
sudo apt update
sudo apt install build-essential libsfml-dev python3-dev python3-pip

python3 -m pip install pybind11 torch
```

Compile the human game:

```bash
g++ -std=c++17 \
    main_game.cpp \
    src/GameEnvironment.cpp \
    src/Player.cpp \
    src/Coin.cpp \
    -Iinclude \
    -o game \
    -lsfml-graphics \
    -lsfml-window \
    -lsfml-system
```

Run it:

```bash
./game
```

Compile the random RL test:

```bash
g++ -std=c++17 \
    main_rl.cpp \
    src/GameEnvironment.cpp \
    src/Player.cpp \
    src/Coin.cpp \
    -Iinclude \
    -o rl_test \
    -lsfml-graphics \
    -lsfml-window \
    -lsfml-system
```

Run it:

```bash
./rl_test
```

Compile the C++ Python extension:

```bash
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    -Iinclude \
    src/bindings.cpp \
    src/GameEnvironment.cpp \
    src/Player.cpp \
    src/Coin.cpp \
    -o rl_training/game_env$(python3-config --extension-suffix) \
    -lsfml-graphics \
    -lsfml-window \
    -lsfml-system
```

Train:

```bash
cd rl_training
python3 train.py
```

Watch the trained agent:

```bash
python3 watch_agent.py
```

---

## 28. Conclusion

This project began as a simple C++ game with a player and a coin.

It evolved into a complete reinforcement learning system in which:

* C++ handles the game environment.
* SFML handles graphics.
* `GameEnvironment` provides a standard RL interface.
* `main_game.cpp` allows human gameplay and manual testing.
* `main_rl.cpp` tests the environment with a random policy.
* pybind11 connects C++ to Python.
* PyTorch implements the DQN.
* Experience replay stores past transitions.
* A target network stabilizes Bellman learning.
* Kaggle provides cloud-based training.
* The trained `.pt` model is brought back to the local machine.
* `watch_agent.py` allows the learned policy to visually control the original SFML game.

The final architecture is reusable:

```text
One C++ Game
     │
     ├── Human Player
     │
     ├── Random Agent
     │
     └── Deep RL Agent
```

That is the central design achievement of the project: **the game itself remains reusable, while the source of control can be changed independently.**

The same architecture can now serve as the foundation for more challenging custom reinforcement learning games.

