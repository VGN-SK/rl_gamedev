#include <iostream>
#include <random>

#include "../include/GameEnvironment.hpp"

int main()
{
    GameEnvironment env(800, 600);

    std::random_device rd;
    std::mt19937 generator(rd());

    std::uniform_int_distribution<int> actionDistribution(0, 3);

    const int numberOfEpisodes = 1000;

    int totalSuccesses = 0;

    for (int episode = 1; episode <= numberOfEpisodes; episode++)
    {
        State state = env.reset();

        bool done = false;
        float totalReward = 0.0f;
        int steps = 0;

        while (!done)
        {
            int actionIndex = actionDistribution(generator);

            Action action = static_cast<Action>(actionIndex);

            StepResult result = env.step(action);

            totalReward += result.reward;
            steps++;

            done = result.done;

            state = result.nextState;
        }

        if (totalReward > 0.0f)
        {
            totalSuccesses++;
        }

        std::cout
            << "Episode " << episode
            << " | Steps: " << steps
            << " | Reward: " << totalReward
            << " | Successes: " << totalSuccesses
            << '\n';
    }

    std::cout << "\nTraining test complete.\n";

    std::cout
        << "Success rate: "
        << (100.0f * totalSuccesses / numberOfEpisodes)
        << "%\n";

    return 0;
}