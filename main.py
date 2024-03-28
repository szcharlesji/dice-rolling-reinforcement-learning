import random
import numpy as np
import sys


def roll_dice(NSides, NDice):
    """
    Rolling multiple dice with a given number of sides.

    Args:
    NSides (int): The number of sides on each dice.
    NDice (int): The number of dice to roll.

    Returns:
    int: The sum of the values rolled on all the dice.
    """
    return sum([random.randint(1, NSides) for _ in range(NDice)])


def choose(score, opponent_score, n_dice, m, win_count, lose_count):
    """
    Choose the number of dice to roll based on the given parameters.

    Args:
        score (int): The current player's score.
        opponent_score (int): The opponent's score.
        n_dice (int): The number of dice available to roll.
        m (float): Exploit/Explore parameter.
        win_count (list): A list representing the number of wins for each dice combination.
        lose_count (list): A list representing the number of losses for each dice combination.

    Returns:
        int: The optimal number of dice to roll based on probability.

    """
    f = []  # Winning probability for each dice
    probabilities = []  # Probability of choosing each dice
    total_records = 0  # Total number of records

    for i in range(n_dice):
        f.append(
            win_count[score][opponent_score][i]
            / (
                win_count[score][opponent_score][i]
                + lose_count[score][opponent_score][i]
            )  # Calculate the winning probability for each dice
        )
        total_records += (
            win_count[score][opponent_score][i] + lose_count[score][opponent_score][i]
        )  # Update the total number of records

    # Find the highest probability and index
    max_f = max(f)
    max_index = f.index(max(f))

    s = 0  # Sum of all probabilities except the max
    for item in f:
        if item != max_f:
            s += item

    max_prob = (total_records * f[max_index] + m) / (
        total_records * f[max_index] + m * n_dice
    )  # Calculate the probability of choosing the max dice

    for i in range(
        n_dice
    ):  # Calculate the probability of choosing each dice other than the max
        if i == max_index:
            probabilities.append(max_prob)
        else:
            p = (
                (1 - max_prob)
                * (total_records * f[i] + m)
                / (total_records * s + m * (n_dice - 1))
            )
            probabilities.append(p)

    # softmax to ensure the probabilities sum to 1
    probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))

    # Choose the based on the probabilities
    outcome = np.arange(1, n_dice + 1)

    return np.random.choice(outcome, p=probabilities)


def game(n_sides, l_target, u_target, n_dice, m, n_games):
    """
    Simulates a game and updates the win and lose counts based on the game results.

    Parameters:
    - n_sides (int): The number of sides on each dice.
    - l_target (int): The lower target score for winning the game.
    - u_target (int): The upper target score for winning the game.
    - n_dice (int): The number of dice to roll in each turn.
    - m (int): The maximum number of moves to consider for each turn.
    - n_games (int): The number of games to simulate.

    Returns:
    - win_counts (ndarray): An array representing the win counts for each possible score combination and move.
    - lose_counts (ndarray): An array representing the lose counts for each possible score combination and move.
    """

    win_counts = np.ones((l_target, l_target, n_dice), dtype=int)
    lose_counts = np.ones((l_target, l_target, n_dice), dtype=int)

    for _ in range(n_games):
        scores = [[0, 0]]
        moves = []
        turn = 0
        score_a = 0
        score_b = 0

        while (score_a < l_target) and (score_b < l_target):
            if turn % 2 == 0:
                move = choose(score_a, score_b, n_dice, m, win_counts, lose_counts)
                score_a += roll_dice(n_sides, move)
            else:
                move = choose(score_b, score_a, n_dice, m, win_counts, lose_counts)
                score_b += roll_dice(n_sides, move)

            scores.append([score_a, score_b])
            moves.append(move)
            turn += 1

            # Update the win and lose counts
            if l_target <= score_a <= u_target or score_b > u_target:  # When A wins
                for j, score in enumerate(scores):
                    if score[0] < l_target and score[1] < l_target:
                        if j % 2 == 0:
                            win_counts[score[0]][score[1]][moves[j] - 1] += 1
                        else:
                            lose_counts[score[1]][score[0]][moves[j] - 1] += 1

            elif l_target <= score_b <= u_target or score_a > u_target:  # When B wins
                for j, score in enumerate(scores):
                    if score[0] < l_target and score[1] < l_target:
                        if j % 2 == 0:
                            lose_counts[score[0]][score[1]][moves[j] - 1] += 1
                        else:
                            win_counts[score[1]][score[0]][moves[j] - 1] += 1

    return win_counts, lose_counts


def main(*args, **kwargs):
    # system arguments
    if len(sys.argv) != 7:
        print(
            "Usage: python main.py <n_sides> <l_target> <u_target> <n_dice> <m> <n_games>"
        )
        sys.exit(1)

    n_sides = int(sys.argv[1])
    l_target = int(sys.argv[2])
    u_target = int(sys.argv[3])
    n_dice = int(sys.argv[4])
    m = int(sys.argv[5])
    n_games = int(sys.argv[6])

    win_counts, lose_counts = game(n_sides, l_target, u_target, n_dice, m, n_games)

    # output information
    print("n_sides = ", n_sides)
    print("l_target = ", l_target)
    print("u_target = ", u_target)
    print("n_dice = ", n_dice)
    print("m = ", m)
    print("n_games = ", n_games)
    print()

    # output decision matrix
    print("Play = ")
    for x in range(l_target):
        for y in range(l_target):
            print(f"{np.argmax(win_counts[x][y] + lose_counts[x][y]) + 1}\t", end="")
        print()
    print()

    # output probability matrix
    print("Probability = ")
    for x in range(l_target):
        for y in range(l_target):
            print(
                f"{win_counts[x][y][np.argmax(win_counts[x][y] + lose_counts[x][y])] / (win_counts[x][y][np.argmax(win_counts[x][y] + lose_counts[x][y])] + lose_counts[x][y][np.argmax(win_counts[x][y] + lose_counts[x][y])]):.5f}\t",
                end="",
            )
        print()


if __name__ == "__main__":
    main()
