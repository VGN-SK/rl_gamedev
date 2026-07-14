#ifndef PLAYER_HPP
#define PLAYER_HPP

#include <SFML/Graphics.hpp>

class Player
{
private:
    sf::RectangleShape shape;
    float speed;

public:
    Player();

    void handleInput(float dt);
    void keepInsideWindow(const sf::RenderWindow& window);

    void draw(sf::RenderWindow& window);

    void move(sf::Vector2f direction, float dt);
    void setPosition(float x, float y);
    void keepInsideWindow(float windowWidth, float windowHeight);

    sf::FloatRect getBounds() const;
    sf::Vector2f getPosition() const;
};

#endif