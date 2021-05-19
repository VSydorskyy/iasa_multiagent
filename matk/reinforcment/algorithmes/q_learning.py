from typing import Dict, Any, Optional
from copy import deepcopy
from tqdm import tqdm

import numpy as np


class QLearning(object):
    def __init__(
        self,
        environment: object,
        environment_config: Dict[str, Any],
        gamma: float,
        epsilon: float,
        lr: float,
        lr_shed: Optional[float] = None,
        epsilon_shed: Optional[float] = None,
    ):
        self.environment = environment(**environment_config)

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.lr_shed = lr_shed
        self.epsilon_shed = epsilon_shed

        self.q_table = {}
        for s in self.environment.states:
            self.q_table[s] = {
                k: 0 for k in self.environment.get_possible_actions(s)
            }

        self.hashed_runs = []
        self.q_maps = None

    def run_policy(self, max_iter=1_000, verbose=True):
        self.environment.agent.reset_coord()
        terminated = False
        it = 0
        stats = {"n_escalators_steps": 0, "reward": 0}
        all_fields = [
            self.environment.v_print(with_agent=True, return_field=True)
        ]
        while not terminated and it < max_iter:

            state = tuple(self.environment.agent.coord)
            actions = self.q_table[state]
            action = max(actions, key=actions.get)

            self.environment.agent.step(
                action, self.environment.get_possible_actions(state)
            )
            all_fields.append(
                self.environment.v_print(with_agent=True, return_field=True)
            )
            terminated = self.environment.get_endgame(
                self.environment.agent.coord
            )

            stats["n_escalators_steps"] += int(
                self.environment.get_escalator(self.environment.agent.coord)
            )
            stats["reward"] += self.environment.get_reward(
                self.environment.agent.coord
            )

            if terminated and verbose:
                print("End game")
            it += 1

        stats["success"] = self.environment.get_success(
            self.environment.agent.coord
        )
        stats["n_iters"] = it
        return all_fields, stats

    def run_algorithm(self, n_steps):
        lr = self.lr
        epsilon = self.epsilon
        self.hashed_runs = []
        all_stats = []
        for step in tqdm(range(n_steps)):
            self.environment.agent.reset_coord()
            terminated = False
            all_fields = [
                self.environment.v_print(with_agent=True, return_field=True)
            ]
            stats = {"n_escalators_steps": 0, "reward": 0}
            while not terminated:

                state = tuple(self.environment.agent.coord)
                state_actions = self.q_table[state]

                if np.random.binomial(n=2, p=epsilon):
                    # Explore
                    action = np.random.choice(list(state_actions.keys()))
                else:
                    # Exploit
                    action = max(state_actions, key=state_actions.get)

                new_state = self.environment.agent.get_coord_from_action(
                    action, state
                )
                reward = self.environment.get_reward(new_state)

                self.q_table[state][action] = (
                    lr
                    * (
                        reward
                        + self.gamma * max(self.q_table[new_state].values())
                    )
                    + (1 - lr) * self.q_table[state][action]
                )

                self.environment.agent.step(
                    action, self.environment.get_possible_actions(state)
                )
                all_fields.append(
                    self.environment.v_print(
                        with_agent=True, return_field=True
                    )
                )
                terminated = self.environment.get_endgame(
                    self.environment.agent.coord
                )

                stats["n_escalators_steps"] += int(
                    self.environment.get_escalator(
                        self.environment.agent.coord
                    )
                )
                stats["reward"] += self.environment.get_reward(
                    self.environment.agent.coord
                )

            stats["success"] = self.environment.get_success(
                self.environment.agent.coord
            )
            self.hashed_runs.append(all_fields)
            all_stats.append(stats)
            if self.lr_shed is not None:
                lr = lr * self.lr_shed
            if self.epsilon_shed is not None:
                epsilon = epsilon * self.epsilon_shed

        self.compute_q_maps()
        return all_stats

    def compute_q_maps(self):
        self.q_maps = {
            k: np.zeros((self.environment.size, self.environment.size))
            for k in ["left", "right", "up", "down"]
        }
        self.q_maps["action"] = np.empty(
            (self.environment.size, self.environment.size), dtype=np.str
        )

        for k, v in self.q_table.items():
            for k_2, v_2 in v.items():
                self.q_maps[k_2][k[0], k[1]] = v_2
            self.q_maps["action"][k[0], k[1]] = max(v, key=v.get)

        self.q_maps["max"] = np.stack(
            list([self.q_maps[k] for k in ["left", "right", "up", "down"]]),
            axis=0,
        ).max(0)

    def reset(self):
        self.q_table = {}
        for s in self.environment.states:
            self.q_table[s] = {
                k: 0 for k in self.environment.get_possible_actions(s)
            }

        self.hashed_runs = []
