#ifndef GAME_HPP
#define GAME_HPP

#include <SFML/Graphics.hpp>

#include "Player.hpp"
#include "Coin.hpp"

class Game
{
private:
    sf::RenderWindow window;

    Player player;
    Coin coin;

    sf::Clock clock;

    int score;

public:
    Game();

    void run();

private:
    void processEvents();
    void update(float dt);
    void render();
};

#endif