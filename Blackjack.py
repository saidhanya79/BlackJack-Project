#This code consits of many modules for the mentioned balckjack game
import random
import numpy as np
import tensorflow as tf
from collections import deque

# Deck and Shuffling
class Deck:
    def __init__(self):
        self.cards = self.create_deck() * 6
        self.shuffle()

    def create_deck(self):
        values = list(range(2, 11)) + ['J', 'Q', 'K', 'A']
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        return [(str(value), suit) for value in values for suit in suits]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        if len(self.cards) < 20:
            self.cards = self.create_deck() * 6
            self.shuffle()
        return self.cards.pop()

# Player Class
class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.score = 0
        self.bet = 0
        self.actions = ['Hit', 'Stand', 'Double Down', 'Split', 'Surrender']

    def receive_card(self, card):
        self.hand.append(card)
        self.score = self.calculate_score()

    def calculate_score(self):
        values = [card[0] for card in self.hand]
        score = 0
        aces = values.count('A')

        for value in values:
            if value in ['J', 'Q', 'K']:
                score += 10
            elif value == 'A':
                score += 11
            else:
                score += int(value)

        while score > 21 and aces:
            score -= 10
            aces -= 1

        return score

# Dealer Class
class Dealer(Player):
    def __init__(self):
        super().__init__("Dealer")

    def play(self, deck):
        while self.score < 17:
            self.receive_card(deck.draw_card())

# Markov Decision Process (MDP)
class MDP:
    def __init__(self, actions):
        # States are the possible scores in Blackjack (4 to 21)
        self.states = list(range(4, 22))
        self.actions = actions
        # Initialize policy with improved Blackjack strategy
        self.policy = {}
        for state in self.states:
            if state >= 17:
                self.policy[state] = "Stand"
            elif state <= 11:
                self.policy[state] = "Hit"
            elif state == 16:
                self.policy[state] = "Surrender"  # Only if dealer shows 9, 10, or Ace
            elif state == 15:
                self.policy[state] = "Surrender"  # Only if dealer shows 10
            else:
                self.policy[state] = "Hit"

    def recommend_action(self, state, player_hand):
        # Only recommend Split if the two cards are identical
        if len(player_hand) == 2 and player_hand[0][0] == player_hand[1][0]:
            return "Split"
        # Otherwise, use the policy
        if state not in self.policy:
            return "Stand"
        return self.policy[state]

# Deep Q-Network (DQN)
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def recommend_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]))
        return np.argmax(q_values[0])

# Real-Time Decision Maker
class DecisionMaker:
    def __init__(self, mdp, dqn):
        self.mdp = mdp
        self.dqn = dqn

    def recommend(self, state, player_hand):
        return self.mdp.recommend_action(state, player_hand)

# Main Game Loop
def main():
    deck = Deck()
    players = [Player("Player 1"), Player("Player 2"), Player("Player 3")]
    dealer = Dealer()

    for player in players:
        player.receive_card(deck.draw_card())
        player.receive_card(deck.draw_card())

    dealer.receive_card(deck.draw_card())
    dealer.receive_card(deck.draw_card())

    for player in players:
        print(f"\n{player.name}'s Hand: {player.hand} - Score: {player.score}")
        print(f"Dealer's Hand: [{dealer.hand[0]}, ('?', '?')]")

        # Real-Time Decision Recommendation
        mdp = MDP(player.actions)
        dqn = DQN(state_size=1, action_size=len(player.actions))
        decision_maker = DecisionMaker(mdp, dqn)
        recommended_action = decision_maker.recommend(player.score, player.hand)

        print(f"Recommended Action: {recommended_action}")

        while player.score < 21:
            action = input(f"{player.name}, Choose an action (Hit, Stand, Double Down, Split, Surrender): ")
            
            if action == "Hit":
                player.receive_card(deck.draw_card())
                print(f"\n{player.name}'s Hand: {player.hand} - Score: {player.score}")
            
            elif action == "Stand":
                print(f"{player.name} Stands at {player.score}.")
                break
            
            elif action == "Double Down":
                player.receive_card(deck.draw_card())
                print(f"\n{player.name}'s Hand: {player.hand} - Score: {player.score}")
                print(f"{player.name} Doubles Down. Final Score: {player.score}")
                break
            
            elif action == "Split":
                if len(player.hand) == 2 and player.hand[0][0] == player.hand[1][0]:
                    print(f"{player.name} Splits the hand.")
                else:
                    print("Split not allowed. Choose another action.")
            
            elif action == "Surrender":
                print(f"{player.name} Surrenders. You lose half your bet.")
                player.score = 0
                break

    dealer.play(deck)
    print(f"\nDealer's Hand: {dealer.hand} - Score: {dealer.score}")

    print("\n=== Result ===")
    for player in players:
        if player.score > 21:
            print(f"{player.name} Busts. Dealer Wins.")
        elif dealer.score > 21:
            print(f"{player.name} Wins! Dealer Busts.")
        elif player.score > dealer.score:
            print(f"{player.name} Wins.")
        elif player.score < dealer.score:
            print(f"{player.name} Loses.")
        else:
            print(f"{player.name} Push (Tie).")

if __name__ == "__main__":
    main()
