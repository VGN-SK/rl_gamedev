#include "../include/Player.hpp"
#include <cmath>

Player::Player()
{
    shape.setSize(sf::Vector2f(50.0f, 50.0f));
    shape.setFillColor(sf::Color::Green);
    shape.setPosition(100.0f, 100.0f);

    speed = 300.0f;
}

void Player::move(sf::Vector2f direction, float dt)
{
    if (direction.x != 0.0f || direction.y != 0.0f)
    {
        float length = std::sqrt(
            direction.x * direction.x +
            direction.y * direction.y
        );

        direction.x /= length;
        direction.y /= length;
    }

    shape.move(direction * speed * dt);
}

void Player::handleInput(float dt)
{
    sf::Vector2f direction(0.0f, 0.0f);

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
    {
        direction.y -= 1.0f;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
    {
        direction.y += 1.0f;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
    {
        direction.x -= 1.0f;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
    {
        direction.x += 1.0f;
    }

    move(direction, dt);
}

void Player::keepInsideWindow(const sf::RenderWindow& window)
{
    sf::Vector2f position = shape.getPosition();
    sf::Vector2f playerSize = shape.getSize();
    sf::Vector2u windowSize = window.getSize();

    if (position.x < 0.0f)
    {
        position.x = 0.0f;
    }

    if (position.y < 0.0f)
    {
        position.y = 0.0f;
    }

    if (position.x + playerSize.x > windowSize.x)
    {
        position.x = windowSize.x - playerSize.x;
    }

    if (position.y + playerSize.y > windowSize.y)
    {
        position.y = windowSize.y - playerSize.y;
    }

    shape.setPosition(position);
}

void Player::keepInsideWindow(float windowWidth, float windowHeight)
{
    sf::Vector2f position = shape.getPosition();
    sf::Vector2f playerSize = shape.getSize();

    if (position.x < 0.0f){
        position.x = 0.0f;
    }

    if (position.y < 0.0f){
        position.y = 0.0f;
    }

    if (position.x + playerSize.x > windowWidth){
        position.x = windowWidth - playerSize.x;
    }

    if (position.y + playerSize.y > windowHeight){
        position.y = windowHeight - playerSize.y;
    }

    shape.setPosition(position);
}

void Player::draw(sf::RenderWindow& window)
{
    window.draw(shape);
}

sf::FloatRect Player::getBounds() const
{
    return shape.getGlobalBounds();
}

sf::Vector2f Player::getPosition() const
{
    return shape.getPosition();
}

void Player::setPosition(float x, float y)
{
    shape.setPosition(x, y);
}