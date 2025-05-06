# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
# https://books.google.ca/books?id=_ATpBwAAQBAJ&lpg=PA201&ots=rinZM8jQ6s&dq=hoeffding%20bound%20gives%20probability%20%22greater%20than%201%22&pg=PA201#v=onepage&q&f=false
from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 100000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
  def __init__(self, p):
    self.p = p
    self.p_estimate = 0.
    self.N = 0  # number of times this bandit was selected

  def pull(self):
    return np.random.random() < self.p

  def update(self, x):
    self.N += 1
    self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N


def ucb(mean, total_plays, plays_for_bandit):
  if plays_for_bandit == 0:
    return float('inf')  # ensure each bandit is selected at least once
  return mean + np.sqrt(2 * np.log(total_plays) / plays_for_bandit)


def run_experiment():
  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
  rewards = np.empty(NUM_TRIALS)
  total_plays = 0

  # Initialization: play each bandit once
  for j in range(len(bandits)):
    x = bandits[j].pull()
    total_plays += 1
    bandits[j].update(x)

   # The RL Agent
  for i in range(len(bandits), NUM_TRIALS):
    ucb_values = [ucb(b.p_estimate, total_plays, b.N) for b in bandits]
    j = np.argmax(ucb_values) # select arm with highest UCB
    x = bandits[j].pull() # take action and observe reward
    total_plays += 1 
    bandits[j].update(x) # update belief about this arm
    rewards[i] = x # record reward

  cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

  # Plot moving average reward (log scale)
  plt.figure(figsize=(10, 6))
  plt.plot(cumulative_average, label='UCB')
  plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES), label='Optimal', linestyle='--')
  plt.xscale('log')
  plt.xlabel("Trial")
  plt.ylabel("Cumulative Average Reward")
  plt.title("UCB Performance (Log Scale)")
  plt.legend()
  plt.grid(True)
  plt.show()

  # Plot (linear scale)
  plt.figure(figsize=(10, 6))
  plt.plot(cumulative_average, label='UCB')
  plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES), label='Optimal', linestyle='--')
  plt.xlabel("Trial")
  plt.ylabel("Cumulative Average Reward")
  plt.title("UCB Performance (Linear Scale)")
  plt.legend()
  plt.grid(True)
  plt.show()

  # Print summary
  for i, b in enumerate(bandits):
    print(f"Bandit {i}: estimated p = {b.p_estimate:.4f}, true p = {b.p}, pulled = {b.N} times")

  print("Total reward earned:", rewards.sum())
  print("Overall win rate:", rewards.sum() / NUM_TRIALS)
  print("Number of times each bandit was selected:", [b.N for b in bandits])

  return cumulative_average


if __name__ == '__main__':
  run_experiment()
