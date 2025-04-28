import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


class Bandit(object):
    def __init__(self, arms):
        self.arms = arms
        self.means = np.random.normal(0, 1, arms)
        self.qvalues = [0] * arms
        self.armcounts = [0] * arms

    def pullArm(self, arm):
        self.armcounts[arm] += 1
        return np.random.normal(self.means[arm], 1, 1)[0]

    def updateArm(self, arm, reward):
        q_t = self.qvalues[arm]
        self.qvalues[arm] = q_t + (1 / self.armcounts[arm]) * (reward - q_t)


def eps_greedy(epsilon):
    def strategy(bandit, t=0):
        return np.random.randint(0, bandit.arms) if np.random.random() < epsilon else np.argmax(bandit.qvalues)
    return strategy


def softmax(tau):
    def strategy(bandit, t=0):
        scaled_vals = np.multiply(bandit.qvalues, tau)
        shifted_vals = scaled_vals - np.max(scaled_vals)
        exp_vals = np.exp(shifted_vals)
        total = np.sum(exp_vals)
        if total == 0 or not np.isfinite(total):
            return np.random.randint(len(bandit.qvalues))
        softmax_probs = exp_vals / total
        return np.random.choice(len(bandit.qvalues), p=softmax_probs)
    return strategy


def ucb(c):
    def strategy(bandit, t):
        for i in range(bandit.arms):
            if bandit.armcounts[i] == 0:
                return i
        ucb_vals = [
            q + c * math.sqrt(math.log(t) / n)
            for q, n in zip(bandit.qvalues, bandit.armcounts)
        ]
        return np.argmax(ucb_vals)
    return strategy


def simulate_bandit(strategy, arms=10, timesteps=1000, episodes=200):
    rewards = [0] * timesteps
    for e in range(episodes):
        np.random.seed(e)
        bandit = Bandit(arms=arms)
        for t in range(timesteps):
            arm = strategy(bandit=bandit, t=t + 1)
            reward = bandit.pullArm(arm)
            bandit.updateArm(arm=arm, reward=reward)
            rewards[t] += (1 / (e + 1)) * (reward - rewards[t])
    return rewards


if __name__ == "__main__":
    timesteps = 1000
    episodes = 2000
    arms = 10

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.rcParams["figure.figsize"] = (15, 10)

    # replace the list with [] if you don't wnat to execute the strategy 
    
    # UCB
    for c in [2]:
        strategy = ucb(c)
        rewards = simulate_bandit(strategy=strategy, arms=arms, timesteps=timesteps, episodes=episodes)
        plt.plot(range(timesteps), rewards, label='$c$ = ' + str(c) + (' (Greedy)' if c == 0 else ''))

    # Softmax
    for tau in [5]:
        strategy = softmax(tau)
        rewards = simulate_bandit(strategy=strategy, arms=arms, timesteps=timesteps, episodes=episodes)
        label = '$\\tau$ = ' + str(tau)
        if tau == 0:
            label += ' (Uniform)'
        elif tau == 100:
            label += ' (Greedy)'
        plt.plot(range(timesteps), rewards, label=label)

    # Epsilon-Greedy
    for eps in [0, 0.1]:
        strategy = eps_greedy(eps)
        rewards = simulate_bandit(strategy=strategy, arms=arms, timesteps=timesteps, episodes=episodes)
        label = '$\\epsilon$ = ' + str(eps) + (' (Greedy)' if eps == 0 else '')
        plt.plot(range(timesteps), rewards, label=label)

    plt.legend(loc='lower right')
    plt.title("Multi-Armed Bandit Strategy Comparison")
    plt.grid(True)
    plt.show()
