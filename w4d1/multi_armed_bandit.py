#%%
import os
from typing import Optional, Union
import gym
import gym.envs.registration
import gym.spaces
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm_notebook

MAIN = __name__ == "__main__"
max_episode_steps = 1000
stationary = True
num_arms = 10
IS_CI = os.getenv("IS_CI")
N_RUNS = 200 if not IS_CI else 5
# %%
ObsType = int
ActType = int

class MultiArmedBandit(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    num_arms: int
    stationary: bool
    arm_reward_means: np.ndarray
    arm_star: int

    def __init__(self, num_arms=10, stationary=True):
        super().__init__()
        self.num_arms = num_arms
        self.stationary = stationary
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(num_arms)
        self.reset()

    def step(self, arm: ActType) -> tuple[ObsType, float, bool, dict]:
        '''
        Note: some documentation references a new style which has 
        (termination, truncation) bools in place of the done bool.
        '''
        assert self.action_space.contains(arm)
        if not self.stationary:
            q_drift = self.np_random.normal(loc=0.0, scale=0.01, size=self.num_arms)
            self.arm_reward_means += q_drift
            self.best_arm = int(np.argmax(self.arm_reward_means))
        reward = self.np_random.normal(loc=self.arm_reward_means[arm], scale=1.0)
        obs = 0
        done = False
        info = dict(best_arm=self.best_arm)
        return (obs, reward, done, info)

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        super().reset(seed=seed)
        if self.stationary:
            self.arm_reward_means = self.np_random.normal(
                loc=0.0, scale=1.0, size=self.num_arms
            )
        else:
            self.arm_reward_means = np.zeros(shape=[self.num_arms])
        self.best_arm = int(np.argmax(self.arm_reward_means))
        if return_info:
            return (0, dict())
        else:
            return 0

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"
        bandit_samples = []
        for arm in range(self.action_space.n):
            bandit_samples += [
                np.random.normal(loc=self.arm_reward_means[arm], scale=1.0, size=1000)
            ]
        plt.violinplot(bandit_samples, showmeans=True)
        plt.xlabel("Bandit Arm")
        plt.ylabel("Reward Distribution")
        plt.show()

# %%
gym.envs.registration.register(
    id="ArmedBanditTestbed-v0",
    entry_point=MultiArmedBandit,
    max_episode_steps=max_episode_steps,
    nondeterministic=True,
    reward_threshold=1.0,
    kwargs={"num_arms": num_arms, "stationary": stationary},
)
if MAIN:
    env = gym.make("ArmedBanditTestbed-v0")
    print("Our env inside its wrappers looks like: ", env)
# %%
class Agent:
    '''Base class for agents in a multi-armed bandit environment (you do not need to 
    add any implementation here)'''

    rng: np.random.Generator

    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.reset(seed)

    def get_action(self) -> ActType:
        raise NotImplementedError()

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        pass

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

def run_episode(env: gym.Env, agent: Agent, seed: int):
    (rewards, was_best) = ([], [])
    env.reset(seed=seed)
    agent.reset(seed=seed)
    done = False
    while not done:
        arm = agent.get_action()
        (obs, reward, done, info) = env.step(arm)
        agent.observe(arm, reward, info)
        rewards.append(reward)
        was_best.append(1 if arm == info["best_arm"] else 0)
    rewards = np.array(rewards, dtype=float)
    was_best = np.array(was_best, dtype=int)
    return (rewards, was_best)

def test_agent(env: gym.Env, agent: Agent, n_runs=200, base_seed=1):
    all_rewards = []
    all_was_bests = []
    base_rng = np.random.default_rng(base_seed)
    for n in tqdm_notebook(range(n_runs)):
        seed = base_rng.integers(low=0, high=10_000, size=1).item()
        (rewards, corrects) = run_episode(env, agent, seed)
        all_rewards.append(rewards)
        all_was_bests.append(corrects)
    return (np.array(all_rewards), np.array(all_was_bests))

#%% [markdown]
#### RandomAgent
#%%
class RandomAgent(Agent):
    def get_action(self) -> ActType:
        return self.rng.integers(0, self.num_arms)

if MAIN:
    rand_agent = RandomAgent(num_arms=10, seed=0)
    test_results = test_agent(env, rand_agent, n_runs=200, base_seed=2)
# %%
rand_results, best_results = test_results
# %%
rand_results.mean(), best_results.astype(np.float64).mean()
# %%
# %%
np.testing.assert_allclose(rand_results.mean(), 0, rtol=0, atol=.02)
# %%
np.testing.assert_allclose(best_results.mean(), 0.1, rtol=0, atol=.01)
 
 #%% [markdown]
 #### Reward averaging
# %%
def plot_rewards(all_rewards: np.ndarray):
    (n_runs, n_steps) = all_rewards.shape
    (fig, ax) = plt.subplots(figsize=(15, 5))
    ax.plot(all_rewards.mean(axis=0), label="Mean over all runs")
    quantiles = np.quantile(all_rewards, [0.05, 0.95], axis=0)
    ax.fill_between(range(n_steps), quantiles[0], quantiles[1], alpha=0.5)
    ax.set(xlabel="Step", ylabel="Reward")
    ax.axhline(0, color="red", linewidth=1)
    fig.legend()
    return fig
#%%
class RewardAveraging(Agent):

    def reset(self, seed: int):
        super().reset(seed)
        self.samples = np.zeros(num_arms)
        self.means = np.ones(num_arms, dtype=np.float64) * self.optimism

    def __init__(self, num_arms: int, seed: int, epsilon: float, optimism: float):
        self.epsilon = epsilon
        self.optimism = optimism
        super().__init__(num_arms, seed)
        
    def observe(self, action: ActType, reward: float, info: dict) -> None:
        self.samples[action] += 1
        self.means[action] += (reward - self.means[action]) / self.samples[action]

    def get_action(self) -> ActType:
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.num_arms)
        return self.means.argmax()

if MAIN:
    env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)
    regular_reward_averaging = RewardAveraging(num_arms, 0, epsilon=0.1, optimism=0)
    (all_rewards, all_corrects) = test_agent(env, regular_reward_averaging, n_runs=N_RUNS)
    print('regular_reward_averaging')
    print(f"Frequency of correct arm: {all_corrects.mean()}")
    print(f"Average reward: {all_rewards.mean()}")
    fig = plot_rewards(all_rewards)
    optimistic_reward_averaging = RewardAveraging(num_arms, 0, epsilon=0.1, optimism=5)
    (all_rewards, all_corrects) = test_agent(env, optimistic_reward_averaging, n_runs=N_RUNS)
    print('optimistic_reward_averaging')
    print(f"Frequency of correct arm: {all_corrects.mean()}")
    print(f"Average reward: {all_rewards.mean()}")
    plot_rewards(all_rewards)
#%% [markdown]
#### Snooping
# %%
class CheatyMcCheater(Agent):
    def __init__(self, num_arms: int, seed: int):
        super().__init__(num_arms, seed)

    def get_action(self):
        return self.best_arm

    def observe(self, action, reward, info):
        self.best_arm = info['best_arm']

    def reset(self, seed: int) -> None:
        self.best_arm = 0
        super().reset(seed)

if MAIN:
    cheater = CheatyMcCheater(num_arms, 0)
    (all_rewards, all_corrects) = test_agent(env, cheater, n_runs=N_RUNS)
    print(f"Frequency of correct arm: {all_corrects.mean()}")
    print(f"Average reward: {all_rewards.mean()}")
    plot_rewards(all_rewards)

#%%[markdown]
#### Upper Confidence Bound
# %%
class UCBActionSelection(Agent):
    def reset(self, seed: int):
        super().reset(seed)
        self.samples = np.zeros(num_arms)
        self.means = np.zeros(num_arms, dtype=np.float64)
        self.t = 1

    def __init__(
        self, num_arms: int, seed: int, c: float, eps: float = 1e-4
    ):
        self.c = c
        self.eps = eps
        super().__init__(num_arms, seed)
        
    def observe(self, action: ActType, reward: float, info: dict) -> None:
        self.t += 1
        self.samples[action] += 1
        self.means[action] += (reward - self.means[action]) / self.samples[action]

    def get_action(self) -> ActType:
        num = np.log(self.t) 
        den = self.samples + self.eps
        obj = (
            self.means + self.c * np.sqrt(num / den)
        )
        assert obj.shape == self.means.shape
        return obj.argmax()

if MAIN:
    env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)
    grad_bandit = UCBActionSelection(num_arms, 0, c=2.0)
    (all_rewards, all_corrects) = test_agent(env, grad_bandit, n_runs=N_RUNS)
    print(f"Frequency of correct arm: {all_corrects.mean()}")
    print(f"Average reward: {all_rewards.mean()}")
    plot_rewards(all_rewards)
#%%[markdown]
#### Gradient bandit
#%%
class GradientBandit(Agent):
    def reset(self, seed: int):
        super().reset(seed)
        self.baseline = 0
        self.preferences = np.zeros(num_arms, dtype=np.float64)
        self.probs = np.exp(self.preferences)
        self.probs /= self.probs.sum()
        self.t = 1

    def __init__(
        self, num_arms: int, seed: int, alpha: float
    ):
        self.alpha = alpha
        super().__init__(num_arms, seed)
        
    def observe(self, action: ActType, reward: float, info: dict) -> None:
        self.t += 1
        self.baseline += (reward - self.baseline) / self.t
        surplus = reward - self.baseline
        updates = -self.alpha * surplus * self.probs
        updates[action] = (
            self.alpha * surplus * (1 - self.probs[action])
        )
        self.preferences += updates
        np.testing.assert_allclose(
            self.preferences.sum(), 0, rtol=0, atol=.01, 
            err_msg=f'prefs should sum to 0: self.probs={self.probs}, preferences={self.preferences}, updates={updates}, reward={reward}, baseline={self.baseline}'
        )
        self.probs = np.exp(self.preferences)
        self.probs /= self.probs.sum()
        np.testing.assert_allclose(
            self.probs.sum(), 1, rtol=0, atol=.01, 
            err_msg=f'probs dont sum to 1: self.probs={self.probs}, preferences={self.preferences}'
        )

    def get_action(self) -> ActType:
        return self.rng.choice(np.arange(self.num_arms), p=self.probs)

if MAIN:
    env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)
    grad_bandit = GradientBandit(num_arms, 0, alpha=0.1)
    (all_rewards, all_corrects) = test_agent(env, grad_bandit, n_runs=20)
    print(f"Frequency of correct arm: {all_corrects.mean()}")
    print(f"Average reward: {all_rewards.mean()}")
    plot_rewards(all_rewards)
# %%
