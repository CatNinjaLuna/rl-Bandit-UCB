import numpy as np
import matplotlib.pyplot as plt

# True probabilities for each arm (unknown to the agent)
TRUE_PROBABILITIES = [0.2, 0.5, 0.75]
NUM_TRIALS = 10000
NUM_ARMS = len(TRUE_PROBABILITIES)

class BayesianBandit:
    def __init__(self, true_p):
        self.true_p = true_p
        self.alpha = 1  # prior: number of successes
        self.beta = 1   # prior: number of failures

    def pull(self):
        return np.random.random() < self.true_p

    def sample(self):
        return np.random.beta(self.alpha, self.beta)

    def update(self, reward):
        if reward == 1:
            self.alpha += 1
        else:
            self.beta += 1

def run_experiment():
    bandits = [BayesianBandit(p) for p in TRUE_PROBABILITIES]
    rewards = np.empty(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        # Sample from each bandit's belief distribution
        sampled_probs = [b.sample() for b in bandits]
        # Choose the bandit with the highest sampled belief
        j = np.argmax(sampled_probs)
        # Pull the chosen bandit
        reward = bandits[j].pull()
        # Update the belief
        bandits[j].update(reward)
        rewards[i] = reward

    # Plot cumulative average reward
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_average, label="Thompson Sampling")
    plt.plot(np.ones(NUM_TRIALS) * np.max(TRUE_PROBABILITIES), label="Best Possible", linestyle='--')
    plt.xlabel("Trial")
    plt.ylabel("Average Reward")
    plt.title("Bayesian Bandits via Thompson Sampling")
    plt.legend()
    plt.grid(True)
    plt.show()

    for i, b in enumerate(bandits):
        print(f"Bandit {i}: α={b.alpha}, β={b.beta}, Estimated p = {b.alpha / (b.alpha + b.beta):.4f}, True p = {b.true_p}")

if __name__ == '__main__':
    run_experiment()
