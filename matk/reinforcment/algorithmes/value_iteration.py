from typing import Dict, Any
from copy import deepcopy

import numpy as np


class ValueItteration(object):
    def __init__(
        self,
        environment: object,
        environment_config: Dict[str, Any],
        gamma: float,
        epsilon: float,
    ):
        self.environment = environment(**environment_config)

        self.gamma = gamma
        self.epsilon = epsilon

        self.policy = {}
        for s in self.environment.states:
            self.policy[s] = np.random.choice(
                self.environment.get_possible_actions(s)
            )
        self.hashed_policies = [deepcopy(self.policy)]

        self.v = {}
        for s in self.environment.states:
            self.v[s] = self.environment.get_reward(s)

        self.value_maps = None

    def run_policy(self, max_iter=1_000, input_policy=None, verbose=True):
        policy = self.policy if input_policy is None else input_policy
        self.environment.agent.reset_coord()
        terminated = False
        it = 0
        stats = {"n_escalators_steps": 0, "reward": 0}
        all_fields = [
            self.environment.v_print(with_agent=True, return_field=True)
        ]
        while not terminated and it < max_iter:

            state = tuple(self.environment.agent.coord)
            action = policy[state]

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

    def run_algorithm(self, verbose=True):
        iteration = 0
        while True:
            biggest_change = 0
            for s in self.environment.states:
                if not self.environment.get_endgame(s):

                    possible_actions = self.environment.get_possible_actions(s)
                    if len(possible_actions) == 0:
                        continue

                    old_v = self.v[s]
                    new_v = 0

                    for a in possible_actions:
                        value = self.v[
                            self.environment.agent.get_coord_from_action(a, s)
                        ]

                        if len(possible_actions) == 1:
                            main_prob = 1.0
                            additional_prob = 0
                        else:
                            main_prob = self.environment.agent.main_step_prob
                            additional_prob = (
                                1 - self.environment.agent.main_step_prob
                            ) / (len(possible_actions) - 1)

                        additional_value = sum(
                            [
                                self.v[
                                    self.environment.agent.get_coord_from_action(
                                        a_r, s
                                    )
                                ]
                                for a_r in possible_actions
                                if a_r != a
                            ]
                        )

                        v = self.environment.get_reward(s) + (
                            self.gamma
                            * (
                                value * main_prob
                                + additional_prob * additional_value
                            )
                        )

                        if v > new_v:
                            new_v = v
                            self.policy[s] = a

                    self.v[s] = new_v
                    biggest_change = max(
                        biggest_change, np.abs(old_v - self.v[s])
                    )

            self.hashed_policies.append(deepcopy(self.policy))
            if biggest_change < self.epsilon:
                break
            iteration += 1

        self.compute_value_maps()
        if verbose:
            print(f"Converged in {iteration} iterations")

    def compute_value_maps(self):
        self.value_maps = {
            "value": np.zeros((self.environment.size, self.environment.size)),
            "action": np.empty(
                (self.environment.size, self.environment.size), dtype=np.str
            ),
        }

        for k, v in self.v.items():
            self.value_maps["value"][k[0], k[1]] = v
            self.value_maps["action"][k[0], k[1]] = self.policy[k]
