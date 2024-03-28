# Program that learns to play a simple game of chance

## Description

The game is a variant of the card game “Blackjack”. Two players alternately roll dice, and keep track of their total across turns. They are each trying to reach a sum that lies in a specified target, between a fixed low value and high value. If a player reaches a score in the target range, they immediately win. If they exceed the high value, they immediately lose.

The players can choose the number of dice to roll on each turn, between 1 and a fixed maximum. The game thus has four parameters:

- NSides, The number of sides of the die. The die is numbered 1 to NSides and all outcomes are equally likely. lowest winning value.
- UTarget, the highest winning value.
- NDice, the maximum number of dice a player may roll.

## Example

For instance, with NSides = 6, LTarget = 15, UTarget = 17, NDice = 2, the following are two possible games. (The players are not necessarily playing well in the games below, just legally.)

Game 1:
Player A rolls 2 dice, which come up 5 and 6. A total: 11.
Player B rolls 2 dice, which come up 3 and 4. B total: 7.
Player A rolls 2 dice, which come up 5 and 5. A total: 21. A loses.

Game 2:
Player A rolls 2 dice, which come up 3 and 4. A total: 7. Player B rolls 2 dice, which come up 5 and 6. B total: 11. Player A rolls 2 dice, which come up 3 and 1. A total: 11. Player B rolls 1 die, which comes up 4. B total: 15. B wins.

Game 3:
Player A rolls 2 dice, which come up 3 and 4. A total: 7. Player B rolls 2 dice, which come up 1 and 4. B total: 5. Player A rolls 2 dice, which come up 2 and 5. A total: 14. Player B rolls 2 dice, which come up 3 and 5. B total: 13. Player A rolls 1 die, which comes up 1. A total: 15. A wins.

## How to run

This program is tested with python 3.12

``` bash
python main.py <n_sides> <l_target> <u_target> <n_dice> <m> <n_games>
```

- n_sides: The number of sides of the die.
- l_target: The lowest winning value.
- u_target: The highest winning value.
- n_dice: The maximum number of dice a player may roll.
- m: The exploit/explore parameter. Recommended value is 100.
- n_games: The number of games to train the model. Recommended value is at least 10000.

This program output two LTarget × LTarget arrays which are the correct number of dice to roll in state and the probability of winning if you roll the correct number of dice.

Note: Probability of 0.5 is an initialized state. It has no meaning after the training because the state is unattainable.
