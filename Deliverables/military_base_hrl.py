"""
Military Base Autonomous Logistics and Security Rover
Hierarchical Reinforcement Learning: SMDP Q-Learning

Project 15: Game-Level Skill Hierarchy
Unit 5: Real-World RL Homework Projects

Deliverable 2: Implementation
Algorithm: Semi-Markov Decision Process (SMDP) Q-Learning with Options Framework

Architecture:
    MilitaryBaseRoverEnv  -- Environment wrapper (MiniGrid DoorKey compound)
    SkillLibrary          -- Reusable low-level motor controllers (Options)
    SMDPManagerAgent      -- High-level mission sequencer (SMDP Q-learning)
    RandomManagerBaseline -- Baseline 1: Random skill selection
    FlatQLearningBaseline -- Baseline 2: Primitive flat Q-learning
"""

import os
import random
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments
from minigrid.core.actions import Actions


# ==============================================================================
# Config: change these to adjust experiment scale or environment
# ==============================================================================
ENV_ID = "MiniGrid-DoorKey-6x6-v0"
NUM_EPISODES = 300
MANAGER_MAX_OPTIONS_PER_EP = 8    # max options the manager can invoke per episode
SKILL_MAX_STEPS = 35              # max primitive steps a single skill may take
SMDP_ALPHA = 0.3                  # SMDP Q-learning step size
SMDP_GAMMA = 0.99                 # temporal discount base
EPSILON_START = 1.0               # initial exploration rate
EPSILON_MIN = 0.05                # floor exploration rate
EPSILON_DECAY_EPISODES = 150      # episodes to reach floor
FLAT_ALPHA = 0.2                  # flat agent step size
FLAT_MAX_STEPS_PER_EP = 100       # primitive step budget per episode
SMOOTHING_WINDOW = 20             # window for moving-average plots
TRAIN_SEED_OFFSET = 2000          # base seed for training episodes
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# 1. Environment Wrapper: Military Base Compound
# ==============================================================================
class MilitaryBaseRoverEnv:
    """
    Wraps MiniGrid DoorKey as a military base partitioned into two sectors:

    Sector 1 (Staging Area)
        Contains the encrypted clearance keycard the rover must retrieve.

    Security Checkpoint
        A reinforced blast gate. Remains locked until the rover carries
        the clearance keycard and toggles it.

    Sector 2 (Command Bunker)
        Contains the mission-critical mainframe terminal (the goal tile).

    The rover starts in Sector 1 and must retrieve the keycard, breach the
    checkpoint, and access the terminal in that order.
    """

    # Compass: direction index -> (dx, dy) displacement on grid
    DIR_TO_VEC = {
        0: (1, 0),   # East
        1: (0, 1),   # South
        2: (-1, 0),  # West
        3: (0, -1),  # North
    }

    def __init__(self, env_id: str = ENV_ID):
        self.env_id = env_id
        self.env = gym.make(env_id)

    def reset(self, seed: int = None):
        """Reset the compound to a fresh layout."""
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def get_poi_coordinates(self):
        """
        Scan the grid and return positions of all Points of Interest (POI).

        Returns:
            keycard_pos: (x, y) of the clearance keycard
            gate_pos:    (x, y) of the blast gate
            terminal_pos:(x, y) of the command terminal
        """
        grid = self.env.unwrapped.grid
        keycard_pos = gate_pos = terminal_pos = None

        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None:
                    if cell.type == "key":
                        keycard_pos = (x, y)
                    elif cell.type == "door":
                        gate_pos = (x, y)
                    elif cell.type == "goal":
                        terminal_pos = (x, y)

        return keycard_pos, gate_pos, terminal_pos

    def get_manager_state(self):
        """
        Compute the compact 3-bit tactical state used by the high-level manager.

        Bits:
            bit 2 (MSB): has_clearance_card  (1 if rover carries keycard)
            bit 1:       gate_is_open        (1 if blast gate is unlocked)
            bit 0 (LSB): rover_in_sector_2   (1 if rover east of gate column)

        State integer range: 0 to 7
        """
        unwrapped = self.env.unwrapped
        has_card = 1 if unwrapped.carrying is not None else 0

        _, gate_pos, _ = self.get_poi_coordinates()
        gate_cell = unwrapped.grid.get(*gate_pos)
        gate_is_open = 1 if (gate_cell is not None and gate_cell.is_open) else 0

        rover_x = int(unwrapped.agent_pos[0])
        rover_in_sector_2 = 1 if rover_x > gate_pos[0] else 0

        state_id = (has_card << 2) | (gate_is_open << 1) | rover_in_sector_2
        return state_id

    def step(self, action):
        """Execute a single primitive motor command."""
        return self.env.step(action)


# ==============================================================================
# 2. Modular Skill Library (Low-Level Options)
# ==============================================================================
class SkillLibrary:
    """
    Reusable certifiable motor controllers for the rover chassis.

    Each skill is an Option omega = (I_omega, pi_omega, beta_omega):
        I_omega   -- initiation set (precondition guard)
        pi_omega  -- BFS-driven navigation and interaction policy
        beta_omega -- termination condition (objective achieved or timeout)

    Adding a new skill: add a constant, extend SKILL_NAMES, and add a branch
    inside execute_skill. No changes needed in the manager.
    """

    SKILL_RETRIEVE_KEYCARD = 0
    SKILL_UNLOCK_GATE = 1
    SKILL_REACH_TERMINAL = 2

    SKILL_NAMES = [
        "Retrieve Keycard",
        "Unlock Blast Gate",
        "Access Terminal",
    ]

    @staticmethod
    def execute_skill(base_env: MilitaryBaseRoverEnv,
                      skill_id: int,
                      max_execution_steps: int = SKILL_MAX_STEPS):
        """
        Execute a named tactical skill until termination or timeout.

        Args:
            base_env:            The active MilitaryBaseRoverEnv instance.
            skill_id:            One of the SKILL_* constants.
            max_execution_steps: Hard step budget (prevents infinite loops).

        Returns:
            success (bool):  True if the skill objective was achieved.
            reward (float):  Cumulative primitive reward plus sub-goal bonus.
            duration (int):  Number of primitive steps consumed (tau).
            done (bool):     True if the episode terminated (rover at terminal).
        """
        keycard_pos, gate_pos, terminal_pos = base_env.get_poi_coordinates()
        unwrapped = base_env.env.unwrapped

        # --- Initiation set check and target configuration ---
        if skill_id == SkillLibrary.SKILL_RETRIEVE_KEYCARD:
            if unwrapped.carrying is not None:
                return False, -0.02, 1, False  # keycard already held
            target_pos = keycard_pos
            target_walkable = False
            interact_action = Actions.pickup
            bonus_reward = 0.1

        elif skill_id == SkillLibrary.SKILL_UNLOCK_GATE:
            gate_cell = unwrapped.grid.get(*gate_pos)
            if gate_cell.is_open:
                return True, 0.0, 1, False  # gate already open
            if unwrapped.carrying is None:
                return False, -0.05, 1, False  # no keycard to unlock with
            target_pos = gate_pos
            target_walkable = False
            interact_action = Actions.toggle
            bonus_reward = 0.2

        elif skill_id == SkillLibrary.SKILL_REACH_TERMINAL:
            target_pos = terminal_pos
            target_walkable = True
            interact_action = Actions.forward
            bonus_reward = 0.0  # environment reward already high (~0.97)

        else:
            return False, -0.05, 1, False  # unknown skill

        steps_spent = 0
        cumulative_reward = 0.0

        while steps_spent < max_execution_steps:
            agent_pos = tuple(int(c) for c in unwrapped.agent_pos)
            agent_dir = int(unwrapped.agent_dir)
            grid = unwrapped.grid

            # Termination: rover arrived at walkable target
            if target_walkable and agent_pos == target_pos:
                return True, cumulative_reward, steps_spent, True

            # Tactical pathfinder: BFS over free cells
            path = SkillLibrary._bfs(
                grid, agent_pos, target_pos, target_walkable
            )

            if path is None:
                # Target currently unreachable (gate locked, etc.)
                return False, -0.05, steps_spent + 1, False

            steps_spent += 1

            if len(path) == 0:
                # Rover is adjacent to target. Face it and interact.
                dx = target_pos[0] - agent_pos[0]
                dy = target_pos[1] - agent_pos[1]
                needed_dir = SkillLibrary._vec_to_dir(dx, dy)

                if agent_dir == needed_dir:
                    _, r, term, trunc, _ = base_env.step(interact_action)
                    cumulative_reward += r
                    if term or trunc:
                        # Mission terminal reached via forward step onto goal
                        return True, cumulative_reward, steps_spent, term

                    # Check skill-specific termination conditions
                    if skill_id == SkillLibrary.SKILL_RETRIEVE_KEYCARD:
                        if unwrapped.carrying is not None:
                            return True, cumulative_reward + bonus_reward, steps_spent, False
                    elif skill_id == SkillLibrary.SKILL_UNLOCK_GATE:
                        if unwrapped.grid.get(*gate_pos).is_open:
                            return True, cumulative_reward + bonus_reward, steps_spent, False

                    return False, cumulative_reward, steps_spent, False

                elif (agent_dir + 1) % 4 == needed_dir:
                    _, r, term, trunc, _ = base_env.step(Actions.right)
                else:
                    _, r, term, trunc, _ = base_env.step(Actions.left)

            else:
                # Navigate one step along the BFS path
                next_node = path[0]
                dx = next_node[0] - agent_pos[0]
                dy = next_node[1] - agent_pos[1]
                needed_dir = SkillLibrary._vec_to_dir(dx, dy)

                if agent_dir == needed_dir:
                    _, r, term, trunc, _ = base_env.step(Actions.forward)
                elif (agent_dir + 1) % 4 == needed_dir:
                    _, r, term, trunc, _ = base_env.step(Actions.right)
                else:
                    _, r, term, trunc, _ = base_env.step(Actions.left)

            cumulative_reward += r
            if term or trunc:
                return True, cumulative_reward, steps_spent, term

        # Hard timeout
        return False, cumulative_reward - 0.05, steps_spent, False

    # --- Internal helpers ---

    @staticmethod
    def _bfs(grid, start: tuple, target: tuple, target_walkable: bool):
        """
        BFS shortest path from start to a position adjacent to (or on) target.

        Returns list of (x,y) waypoints, or None if unreachable.
        An empty list means start is already at the target position/adjacency.
        """
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            curr, path = queue.popleft()

            if target_walkable and curr == target:
                return path
            if not target_walkable:
                dist = abs(curr[0] - target[0]) + abs(curr[1] - target[1])
                if dist == 1:
                    return path

            for dx, dy in MilitaryBaseRoverEnv.DIR_TO_VEC.values():
                nxt = (curr[0] + dx, curr[1] + dy)
                if 0 <= nxt[0] < grid.width and 0 <= nxt[1] < grid.height:
                    cell = grid.get(*nxt)
                    is_free = (
                        cell is None
                        or (cell.type == "door" and cell.is_open)
                        or cell.type == "goal"
                        or (nxt == target and target_walkable)
                    )
                    if is_free and nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, path + [nxt]))

        return None  # unreachable

    @staticmethod
    def _vec_to_dir(dx: int, dy: int) -> int:
        """Convert a displacement vector to a compass direction index."""
        for d, vec in MilitaryBaseRoverEnv.DIR_TO_VEC.items():
            if vec == (dx, dy):
                return d
        raise ValueError(f"Invalid displacement ({dx}, {dy})")


# ==============================================================================
# 3. High-Level Mission Manager: SMDP Q-Learning
# ==============================================================================
class SMDPManagerAgent:
    """
    Tactical mission controller using SMDP Q-learning.

    The manager selects a skill omega at abstract state S, waits tau steps
    for the skill to terminate, then updates using the SMDP Bellman equation:

        Q(S, omega) <- Q(S, omega) + alpha * [
            R_cumulative + gamma^tau * max_omega' Q(S', omega') - Q(S, omega)
        ]

    The gamma^tau factor correctly discounts rewards proportional to how long
    the rover spent executing that skill.
    """

    def __init__(self,
                 num_states: int = 8,
                 num_options: int = 3,
                 alpha: float = SMDP_ALPHA,
                 gamma: float = SMDP_GAMMA):
        self.num_states = num_states
        self.num_options = num_options
        self.alpha = alpha
        self.gamma = gamma
        # Q(state, option) table, initialized to zero
        self.q_table = np.zeros((num_states, num_options))

    def select_skill(self, state: int, epsilon: float) -> int:
        """Epsilon-greedy option selection."""
        if random.random() < epsilon:
            return random.randint(0, self.num_options - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, option: int, reward: float,
               tau: int, next_state: int, done: bool):
        """SMDP Q-learning update with multi-step discount gamma^tau."""
        discount = 0.0 if done else (self.gamma ** tau)
        target = reward + discount * np.max(self.q_table[next_state])
        td_error = target - self.q_table[state, option]
        self.q_table[state, option] += self.alpha * td_error

    def greedy_policy_summary(self) -> str:
        """Return a human-readable summary of the learned greedy policy."""
        lines = ["Learned greedy policy (State -> Best Skill):"]
        for s in range(self.num_states):
            has_card = (s >> 2) & 1
            gate_open = (s >> 1) & 1
            in_s2 = s & 1
            best = int(np.argmax(self.q_table[s]))
            q_vals = np.round(self.q_table[s], 3)
            lines.append(
                f"  S{s} (card={has_card} gate={gate_open} sec2={in_s2})"
                f" -> {SkillLibrary.SKILL_NAMES[best]:20s} Q={q_vals}"
            )
        return "\n".join(lines)


# ==============================================================================
# 4. Baseline Agents
# ==============================================================================
class RandomManagerBaseline:
    """
    Baseline 1: Uniform-random skill selection.

    Demonstrates that knowing the correct skill sequence (not just having
    skills available) is what the SMDP manager learns.
    """

    def __init__(self, num_options: int = 3):
        self.num_options = num_options

    def select_skill(self, state: int) -> int:
        return random.randint(0, self.num_options - 1)


class FlatQLearningBaseline:
    """
    Baseline 2: Tabular Q-learning over raw primitive motor actions.

    State: (x, y, direction, has_card, gate_open) -- 5-tuple
    Actions: 7 MiniGrid primitives

    Demonstrates that flat RL cannot solve multi-stage sparse-reward
    environments without temporal abstraction.
    """

    def __init__(self,
                 num_actions: int = 7,
                 alpha: float = FLAT_ALPHA,
                 gamma: float = SMDP_GAMMA):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: dict = {}

    def _state_key(self, base_env: MilitaryBaseRoverEnv) -> tuple:
        unwrapped = base_env.env.unwrapped
        pos = tuple(int(c) for c in unwrapped.agent_pos)
        direction = int(unwrapped.agent_dir)
        has_card = 1 if unwrapped.carrying is not None else 0
        _, gate_pos, _ = base_env.get_poi_coordinates()
        gate_open = 1 if unwrapped.grid.get(*gate_pos).is_open else 0
        return (pos[0], pos[1], direction, has_card, gate_open)

    def _ensure_state(self, key: tuple):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.num_actions)

    def select_action(self, state_key: tuple, epsilon: float) -> int:
        self._ensure_state(state_key)
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        return int(np.argmax(self.q_table[state_key]))

    def update(self, state_key: tuple, action: int, reward: float,
               next_key: tuple, done: bool):
        self._ensure_state(state_key)
        self._ensure_state(next_key)
        target = reward + (0.0 if done else self.gamma * np.max(self.q_table[next_key]))
        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error

    def get_state_key(self, base_env: MilitaryBaseRoverEnv) -> tuple:
        return self._state_key(base_env)


# ==============================================================================
# 5. Experiment Runner
# ==============================================================================
def _epsilon(episode: int) -> float:
    """Linear epsilon decay from EPSILON_START to EPSILON_MIN."""
    frac = min(1.0, episode / EPSILON_DECAY_EPISODES)
    return EPSILON_START + frac * (EPSILON_MIN - EPSILON_START)


def train_smdp_manager(base_env: MilitaryBaseRoverEnv,
                       num_episodes: int = NUM_EPISODES):
    """Train the hierarchical SMDP manager. Returns (agent, rewards, successes)."""
    agent = SMDPManagerAgent()
    rewards, successes = [], []

    for ep in range(num_episodes):
        base_env.reset(seed=TRAIN_SEED_OFFSET + ep)
        state = base_env.get_manager_state()
        ep_reward = 0.0
        mission_success = False
        done = False
        manager_step = 0
        eps = _epsilon(ep)

        while not done and manager_step < MANAGER_MAX_OPTIONS_PER_EP:
            manager_step += 1
            option = agent.select_skill(state, eps)
            success, r, tau, done = SkillLibrary.execute_skill(base_env, option)
            next_state = base_env.get_manager_state()
            agent.update(state, option, r, tau, next_state, done)
            state = next_state
            ep_reward += r
            if done and r > 0.5:
                mission_success = True

        rewards.append(ep_reward)
        successes.append(int(mission_success))

    return agent, rewards, successes


def evaluate_random_manager(base_env: MilitaryBaseRoverEnv,
                             num_episodes: int = NUM_EPISODES):
    """Evaluate random baseline. Returns (rewards, successes)."""
    baseline = RandomManagerBaseline()
    rewards, successes = [], []

    for ep in range(num_episodes):
        base_env.reset(seed=TRAIN_SEED_OFFSET + ep)
        ep_reward = 0.0
        mission_success = False
        done = False
        manager_step = 0

        while not done and manager_step < MANAGER_MAX_OPTIONS_PER_EP:
            manager_step += 1
            option = baseline.select_skill(base_env.get_manager_state())
            success, r, tau, done = SkillLibrary.execute_skill(base_env, option)
            ep_reward += r
            if done and r > 0.5:
                mission_success = True

        rewards.append(ep_reward)
        successes.append(int(mission_success))

    return rewards, successes


def train_flat_qlearning(base_env: MilitaryBaseRoverEnv,
                         num_episodes: int = NUM_EPISODES):
    """Train flat Q-learning baseline. Returns (agent, rewards, successes)."""
    agent = FlatQLearningBaseline()
    rewards, successes = [], []

    for ep in range(num_episodes):
        base_env.reset(seed=TRAIN_SEED_OFFSET + ep)
        state_key = agent.get_state_key(base_env)
        ep_reward = 0.0
        mission_success = False
        done = False
        step_count = 0
        eps = max(0.1, 1.0 - ep / num_episodes)

        while not done and step_count < FLAT_MAX_STEPS_PER_EP:
            step_count += 1
            action = agent.select_action(state_key, eps)
            _, r, term, trunc, _ = base_env.step(action)
            done = term or trunc
            next_key = agent.get_state_key(base_env)
            agent.update(state_key, action, r, next_key, done)
            state_key = next_key
            ep_reward += r
            if done and r > 0.5:
                mission_success = True

        rewards.append(ep_reward)
        successes.append(int(mission_success))

    return agent, rewards, successes


def moving_average(data: list, window: int) -> np.ndarray:
    """Compute centered moving average."""
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_results(smdp_r, smdp_s, rand_r, rand_s, flat_r, flat_s,
                 window: int = SMOOTHING_WINDOW,
                 save_path: str = None):
    """Generate and save the learning curve comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Military Base Rover: Hierarchical RL vs. Baselines",
        fontsize=14, fontweight="bold", y=1.02
    )

    colors = {
        "smdp": "#1f77b4",
        "rand": "#ff7f0e",
        "flat": "#d62728",
    }

    # Subplot 1: Episode Reward
    ax = axes[0]
    ax.plot(moving_average(smdp_r, window),
            label="Hierarchical SMDP Manager", color=colors["smdp"], lw=2.5)
    ax.plot(moving_average(rand_r, window),
            label="Random Manager Baseline", color=colors["rand"], lw=1.8, ls="--")
    ax.plot(moving_average(flat_r, window),
            label="Flat Q-Learning Baseline", color=colors["flat"], lw=1.8, ls=":")
    ax.set_title(f"Cumulative Episode Reward ({window}-ep avg)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(loc="lower right")
    ax.grid(True, ls="--", alpha=0.5)

    # Subplot 2: Mission Success Rate
    ax = axes[1]
    ax.plot(moving_average(smdp_s, window) * 100,
            label="Hierarchical SMDP Manager", color=colors["smdp"], lw=2.5)
    ax.plot(moving_average(rand_s, window) * 100,
            label="Random Manager Baseline", color=colors["rand"], lw=1.8, ls="--")
    ax.plot(moving_average(flat_s, window) * 100,
            label="Flat Q-Learning Baseline", color=colors["flat"], lw=1.8, ls=":")
    ax.set_title(f"Mission Success Rate % ({window}-ep avg)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower right")
    ax.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.close()


def run_experiment(num_episodes: int = NUM_EPISODES):
    """
    Full experimental pipeline:
      1. Train SMDP hierarchical manager
      2. Evaluate random manager baseline
      3. Train flat Q-learning baseline
      4. Plot and print results
    """
    print("=" * 60)
    print("Military Base Rover: HRL Experiment")
    print("=" * 60)

    base_env = MilitaryBaseRoverEnv()

    print(f"\n[1/3] Training SMDP Hierarchical Manager ({num_episodes} ep)...")
    smdp_agent, smdp_r, smdp_s = train_smdp_manager(base_env, num_episodes)

    print(f"[2/3] Evaluating Random Manager Baseline ({num_episodes} ep)...")
    rand_r, rand_s = evaluate_random_manager(base_env, num_episodes)

    print(f"[3/3] Training Flat Q-Learning Baseline ({num_episodes} ep)...")
    flat_agent, flat_r, flat_s = train_flat_qlearning(base_env, num_episodes)

    # --- Plot ---
    plot_path = os.path.join(OUTPUT_DIR, "learning_curves_comparison.png")
    plot_results(smdp_r, smdp_s, rand_r, rand_s, flat_r, flat_s,
                 save_path=plot_path)

    # --- Console summary ---
    last = 50
    print("\n" + "=" * 60)
    print(f"Performance Summary (final {last} episodes)")
    print("=" * 60)
    print(f"  SMDP Hierarchical Manager : {np.mean(smdp_s[-last:]) * 100:5.1f}% success")
    print(f"  Random Manager Baseline   : {np.mean(rand_s[-last:]) * 100:5.1f}% success")
    print(f"  Flat Q-Learning Baseline  : {np.mean(flat_s[-last:]) * 100:5.1f}% success")
    print()
    print(smdp_agent.greedy_policy_summary())

    return smdp_agent, smdp_r, smdp_s, rand_r, rand_s, flat_r, flat_s


if __name__ == "__main__":
    run_experiment()
