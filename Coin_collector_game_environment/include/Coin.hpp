#ifndef COIN_HPP
#define COIN_HPP

#include <SFML/Graphics.hpp>
#include <random>

class Coin
{
private:
    sf::CircleShape shape;

    std::mt19937 generator;

public:
    Coin();

    void respawn(const sf::RenderWindow& window);

    void respawn(unsigned int width, unsigned int height);

    void draw(sf::RenderWindow& window);

    sf::FloatRect getBounds() const;
    sf::Vector2f getPosition() const;
};

#endif