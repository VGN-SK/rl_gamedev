#include "../include/GameEnvironment.hpp"

GameEnvironment::GameEnvironment(int width, int height)
    : windowWidth(width),
      windowHeight(height),
      currentStep(0),
      maxSteps(500),
      generator(std::random_device{}()),
      window(nullptr)
{
}

State GameEnvironment::getState() const
{
    sf::Vector2f playerPosition = player.getPosition();
    sf::Vector2f coinPosition = coin.getPosition();

    State state;

    state.dx =
        (coinPosition.x - playerPosition.x)
        / static_cast<float>(windowWidth);

    state.dy =
        (coinPosition.y - playerPosition.y)
        / static_cast<float>(windowHeight);

    return state;
}

State GameEnvironment::reset()
{
    currentStep = 0;

    float playerWidth = player.getBounds().width;
    float playerHeight = player.getBounds().height;

    std::uniform_real_distribution<float> randomPlayerX(
        0.0f,
        static_cast<float>(windowWidth) - playerWidth
    );

    std::uniform_real_distribution<float> randomPlayerY(
        0.0f,
        static_cast<float>(windowHeight) - playerHeight
    );

    player.setPosition(
        randomPlayerX(generator),
        randomPlayerY(generator)
    );

    do{
        coin.respawn(windowWidth, windowHeight);
    }while (player.getBounds().intersects(coin.getBounds()));

    return getState();
}

StepResult GameEnvironment::step(Action action)
{
    constexpr float FIXED_DT = 1.0f / 60.0f;

    sf::Vector2f direction(0.0f, 0.0f);

    switch (action)
    {
        case Action::Up:
            direction.y = -1.0f;
            break;

        case Action::Down:
            direction.y = 1.0f;
            break;

        case Action::Left:
            direction.x = -1.0f;
            break;

        case Action::Right:
            direction.x = 1.0f;
            break;
    }

    player.move(direction, FIXED_DT);

    player.keepInsideWindow(static_cast<float>(windowWidth),static_cast<float>(windowHeight));

    currentStep++;

    float reward = -0.01f;
    bool done = false;

    if (player.getBounds().intersects(coin.getBounds()))
    {
        reward = 10.0f;
        done = true;
    }

    if (currentStep >= maxSteps)
    {
        done = true;
    }

    return {
        getState(),
        reward,
        done
    };
}

void GameEnvironment::enableRendering()
{
    if (!window)
    {
        window = std::make_unique<sf::RenderWindow>(
            sf::VideoMode(windowWidth, windowHeight),
            "DQN Agent Playing Coin Collector"
        );

        window->setFramerateLimit(60);
    }
}

void GameEnvironment::renderWindow()
{
    if (!window)
    {
        return;
    }

    window->clear();

    player.draw(*window);
    coin.draw(*window);

    window->display();
}

bool GameEnvironment::isWindowOpen() const
{
    return window && window->isOpen();
}

void GameEnvironment::handleWindowEvents()
{
    if (!window)
    {
        return;
    }

    sf::Event event;

    while (window->pollEvent(event))
    {
        if (event.type == sf::Event::Closed)
        {
            window->close();
        }
    }
}
