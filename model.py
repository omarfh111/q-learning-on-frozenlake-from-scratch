"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # TODO: build a 2D float64 numpy array of zeros sized by states and actions.
    Q = np.zeros((num_states, num_actions), dtype=float)
    return Q

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    # TODO: index the row for `state` and return its maximum value
    best_next_value = np.max(q_table[state])
    return best_next_value

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    # TODO: return argmax over the action axis for this state's Q values
    action = int(np.argmax(q_table[state]))
    return action

# Step 4 - sample_random_action
def sample_random_action(action_space):
    # TODO: draw a uniformly random action from the given Gymnasium action space
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    # TODO: draw a uniform sample from rng and compare it to epsilon
    return rng.random() < epsilon

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # TODO: with prob epsilon explore via action_space, else pick a max-Q action (random among ties)
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)
    row = q_table[state]
    best_value = np.max(row)
    best_actions= np.where(row == best_value)[0]
    action = rng.choice(best_actions)
    return int(action)

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    # TODO: return max(min_epsilon, epsilon * decay_rate)
    return max(min_epsilon,epsilon * decay_rate)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    # TODO: compute r + gamma * max_a Q(next_state, a), zeroing the bootstrap when done.
    if done:
        return float(reward)
    target = reward + gamma * max_q_value(q_table, next_state)
    return float(target)

# Step 9 - td_error
def td_error(target, q_table, state, action):
    # TODO: return the TD error: target minus current Q(state, action)
    td_error = float(target - q_table[state,action])
    return td_error

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # TODO: apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    target = td_target(reward, gamma, q_table, next_state, done)
    error = td_error(target, q_table, state, action)

    q_table[state, action] += alpha * error

    return float(q_table[state, action])

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    # TODO: select epsilon-greedy action, step env, apply Q-learning update, return (next_state, reward, done)
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space , rng)               # epsilon-greedy

    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)                         # update Q-table

    return int(next_state), float(reward), bool(done)

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # TODO: reset env, then repeatedly call interaction_step until done or max_steps, returning total reward.
    state, info = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        next_state, reward, done = interaction_step(
            env, q_table, state, epsilon, alpha, gamma, rng
        )

        total_reward += reward
        state = next_state

        if done:
            break

    return float(total_reward)

# Step 13 - train_q_learning
import numpy as np

def train_q_learning(env, num_episodes, alpha=0.8, gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99, seed=0, max_steps=200):
    # TODO: train a Q-learning agent for num_episodes; return (q_table, returns)
    q_table = np.zeros(
        (env.observation_space.n, env.action_space.n),
        dtype=float
    )

    returns = []
    epsilon = epsilon_start

    rng = np.random.default_rng(seed)

    env.action_space.seed(seed)
    env.reset(seed=seed)

    for _ in range(num_episodes):
        episode_return = run_training_episode(
            env,
            q_table,
            epsilon,
            alpha,
            gamma,
            rng
        )

        returns.append(float(episode_return))

        epsilon = decay_epsilon(
            epsilon,
            epsilon_decay,
            epsilon_min
        )

    return q_table, returns

# Step 14 - extract_greedy_policy
def extract_greedy_policy(q_table):
    # TODO: return a 1D int64 array mapping each state to its best (argmax) action.
    return np.argmax(q_table, axis=1).astype(np.int64)

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

