#include "../include/Coin.hpp"

Coin::Coin()
    : generator(std::random_device{}())
{
    shape.setRadius(15.0f);
    shape.setFillColor(sf::Color::Yellow);
    shape.setPosition(500.0f, 300.0f);
}

void Coin::respawn(const sf::RenderWindow& window)
{
    float diameter = shape.getRadius() * 2.0f;

    float maxX = window.getSize().x - diameter;
    float maxY = window.getSize().y - diameter;

    std::uniform_real_distribution<float> randomX(0.0f,maxX);

    std::uniform_real_distribution<float> randomY(0.0f, maxY);

    shape.setPosition(randomX(generator),randomY(generator));
}

void Coin::respawn(unsigned int width, unsigned int height)
{
    float diameter = shape.getRadius() * 2.0f;

    float maxX = static_cast<float>(width) - diameter;
    float maxY = static_cast<float>(height) - diameter;

    std::uniform_real_distribution<float> randomX(0.0f, maxX);
    std::uniform_real_distribution<float> randomY(0.0f, maxY);

    shape.setPosition(randomX(generator),randomY(generator));
}

void Coin::draw(sf::RenderWindow& window)
{
    window.draw(shape);
}

sf::FloatRect Coin::getBounds() const
{
    return shape.getGlobalBounds();
}

sf::Vector2f Coin::getPosition() const
{
    return shape.getPosition();
}