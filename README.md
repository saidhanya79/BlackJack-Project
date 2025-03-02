The Blackjack.py script implements a Blackjack game with AI-assisted decision-making. Here's a structured description of the code:

Overview
This script simulates a game of Blackjack, including deck management, player actions, and a reinforcement learning model to predict the best moves.

Key Components
Deck Management (Deck class)

Creates a standard deck of 52 cards, replicated 6 times.
Implements shuffling and card-drawing mechanics.
Ensures reshuffling when the deck is low on cards.
Player Management (Player class)

Tracks player information: hand, score, and bet.
Implements basic Blackjack actions such as Hit, Stand, Double Down, Split, and Surrender.
Manages card reception.
Dealer Logic (Dealer class, likely present)

Handles dealer-specific rules, including automatic hitting until a threshold.
AI Decision Making (Reinforcement Learning)

Uses TensorFlow and NumPy to build a model for predicting the best move.
Employs experience replay with a deque to store past game states.
The model likely trains based on game simulations.
Game Flow and Execution

Handles rounds of Blackjack, player decisions, and result evaluation.
Implements reinforcement learning to improve strategy over time.
