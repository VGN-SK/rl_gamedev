#include "../include/Game.hpp"
#include <iostream>

Game::Game()
    : window(sf::VideoMode(800, 600),"Coin Collector"),
      score(0)
{
    window.setFramerateLimit(60);
}

void Game::run()
{
    while (window.isOpen())
    {
        float dt = clock.restart().asSeconds();

        processEvents();

        update(dt);

        render();
    }
}

void Game::processEvents()
{
    sf::Event event;

    while (window.pollEvent(event))
    {
        if (event.type == sf::Event::Closed)
        {
            window.close();
        }
    }
}

void Game::update(float dt)
{
    player.handleInput(dt);
    player.keepInsideWindow(window);

    if (player.getBounds().intersects(coin.getBounds()))
    {
        score++;
        coin.respawn(window);
        std::cout << "Score: " << score << '\n';
    }
}

void Game::render()
{
    window.clear();

    coin.draw(window);
    player.draw(window);

    window.display();
}