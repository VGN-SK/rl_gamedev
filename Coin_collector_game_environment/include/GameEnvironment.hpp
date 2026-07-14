#pragma once

#include "Player.hpp"
#include "Coin.hpp"

#include <random>

#include <memory> //for graphics window rendering without affecting RL training by using unique_ptr
#include <SFML/Graphics.hpp>

enum class Action
{
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3
};

struct State
{
    float dx;
    float dy;
};

struct StepResult
{
    State nextState;
    float reward;
    bool done;
};

class GameEnvironment
{
private:
    Player player;
    Coin coin;

    int windowWidth;
    int windowHeight;

    int currentStep;
    int maxSteps;

    std::mt19937 generator;

    std::unique_ptr<sf::RenderWindow> window;

public:
    GameEnvironment(int width, int height);

    State reset();
    State getState() const;
    StepResult step(Action action);

    void enableRendering();
    void renderWindow();
    bool isWindowOpen() const;
    void handleWindowEvents();
};