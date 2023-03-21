from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Dict as gym_dict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class BaseProtocol(ABC):
    @abstractmethod
    def __init__(self, num_regions: int, num_discrete_action_levels: int) -> None:
        self.num_regions: int = num_regions
        self.num_discrete_action_levels: int = num_discrete_action_levels

        self.stage_idx = 0
        self.done = False

        if not hasattr(self, "stages"):
            raise NotImplementedError("stages attribute has not been set")

        action_lengths = [len(s["action_space"]) for s in self.stages]
        self.action_offsets = {i: o for i, o in enumerate(np.cumsum(action_lengths))}
        self.num_stages = len(self.stages)

    def check_do_step(self, rice_actions: dict, protocol_actions: dict) -> Tuple[bool, dict]:
        if self.stage_idx == self.num_stages:
            self.stage_idx = 0
            return True, rice_actions

        stage_actions = self.split_actions(protocol_actions)
        self.stages[self.stage_idx]["function"](stage_actions)
        self.stage_idx += 1

        return False, rice_actions

    def split_actions(self, actions) -> Dict[int, np.ndarray]:
        end_idx = self.action_offsets[self.stage_idx]
        if self.stage_idx == 0:
            return {k: v[0:end_idx] for k, v in actions.items()}

        start_idx = self.action_offsets[self.stage_idx - 1]
        return {k: v[start_idx:end_idx] for k, v in actions.items()}

    def check_done_restart(self) -> bool:
        if self.stage_idx == self.num_stages:
            self.stage_idx = 0
            return True
        return False

    def get_partial_action_mask(self) -> Dict[int, Dict[str, list]]:
        return defaultdict(dict)

    def reset(self) -> None:
        pass

    def get_protocol_state(self) -> Dict[str, np.ndarray]:
        return {}

    def get_pub_priv_features(self) -> Tuple[list, list]:
        return [], []


class NoProtocol(BaseProtocol):
    def __init__(self, num_regions, num_discrete_action_levels) -> None:
        self.stages = []
        super().__init__(num_regions, num_discrete_action_levels)


class DirectSanction(BaseProtocol):
    """This protocol directly punishes agents based on their mitigation rate.
    There is no negotiation involved. Actions are modified such that states
    that do not mitigate max will be max sanctioned in tariff and import.
    Vice versa, states that do mitigate max will receive max bonus.
    """
    def __init__(self, num_regions, num_discrete_action_levels) -> None:
        self.stages = []
        super().__init__(num_regions, num_discrete_action_levels)

    def check_do_step(self, rice_actions, protocol_actions) -> Tuple[bool, dict]:
        to_punish = []
        for region_id, actions in rice_actions.items():
            if not actions[1] == 9:
                to_punish.append(region_id)

        to_punish_import_indices = [e + 3 for e in to_punish]
        to_punish_tariff_indices = [e + 30 for e in to_punish]

        rice_actions_modified = {}
        for region_id, actions in rice_actions.items():
            actions_modified = actions.copy()
            actions_modified[3:30] = 0
            actions_modified[to_punish_import_indices] = 9
            actions_modified[30:] = 9
            actions_modified[to_punish_tariff_indices] = 0
            rice_actions_modified[region_id] = actions_modified

        return True, rice_actions_modified

    def get_partial_action_mask(self):
        """
        Generate action masks.
        """
        action_mask_dict = defaultdict(dict)
        mask = [0] * (self.num_discrete_action_levels * self.num_regions)
        for region_id in range(self.num_regions):
            action_mask_dict[region_id]["tariff"] = mask
            action_mask_dict[region_id]["import"] = mask

        return action_mask_dict
    

class DirectProportionalSanction(BaseProtocol):
    """This protocol directly punishes agents based on their mitigation rate.
    There is no negotiation involved. Actions are modified such that states
    that do not mitigate max will be max sanctioned in tariff and import.
    Vice versa, states that do mitigate max will receive max bonus.
    """
    def __init__(self, num_regions, num_discrete_action_levels) -> None:
        self.stages = []
        super().__init__(num_regions, num_discrete_action_levels)

    def check_do_step(self, rice_actions, protocol_actions) -> Tuple[bool, dict]:
        import_actions = []
        tariff_actions = []
        for region_id, actions in rice_actions.items():
            mitigation_rate = int(actions[1])
            import_actions.append(mitigation_rate)
            tariff_actions.append(self.num_discrete_action_levels - mitigation_rate - 1)

        rice_actions_modified = {}
        for region_id, actions in rice_actions.items():
            actions_modified = actions.copy()
            actions_modified[3:30] = import_actions
            actions_modified[30:] = tariff_actions
            rice_actions_modified[region_id] = actions_modified

        return True, rice_actions_modified

    def get_partial_action_mask(self):
        """
        Generate action masks.
        """
        action_mask_dict = defaultdict(dict)
        mask = [0] * (self.num_discrete_action_levels * self.num_regions)
        for region_id in range(self.num_regions):
            action_mask_dict[region_id]["tariff"] = mask
            action_mask_dict[region_id]["import"] = mask

        return action_mask_dict


class BilateralNegotiatorWithOnlyTariff(BaseProtocol):

    """
    Updated Bi-lateral negotiation included as the original with the following added:

    If agent is in a bilateral agreement:
    Then they tariff those not in a bilateral agreement

    NOTE: mitigation here is not masked, only tariffs are masked


    Action masks:
    - tariff

    Attributes:
          rice: instance of RICE-N model
          stages (list(dict)): function steps that this negotiator class performs,
                               each contains a function and number of actions
                               used by the function
          num_negotiation_stages (int): number of stages in total

    """

    def __init__(self, rice):
        """
        Defines necessary parameters for communication
        with rice class

        Args:
            Rice: instance of RICE-N model
        """
        self.rice = rice
        self.stages = [
            {
                "function": self.proposal_step,
                "numberActions": (
                    [self.rice.num_discrete_action_levels] * 2 * self.rice.num_regions
                ),
            },
            {
                "function": self.evaluation_step,
                "numberActions": [2] * self.rice.num_regions,
            },
        ]
        self.num_negotiation_stages = len(self.stages)

    def proposal_step(self, actions=None):
        """
        Update Proposal States and Observations using proposal actions
        Update Stage to 1 - Evaluation
        """
        assert self.rice.negotiation_on
        assert self.rice.stage == 1

        assert isinstance(actions, dict)
        assert len(actions) == self.rice.num_regions

        action_offset_index = len(
            self.rice.savings_action_nvec
            + self.rice.mitigation_rate_action_nvec
            + self.rice.export_action_nvec
            + self.rice.import_actions_nvec
            + self.rice.tariff_actions_nvec
        )
        num_proposal_actions = len(self.stages[0]["numberActions"])

        m1_all_regions = [
            actions[region_id][
                action_offset_index : action_offset_index + num_proposal_actions : 2
            ]
            / self.rice.num_discrete_action_levels
            for region_id in range(self.rice.num_regions)
        ]

        m2_all_regions = [
            actions[region_id][
                action_offset_index + 1 : action_offset_index + num_proposal_actions : 2
            ]
            / self.rice.num_discrete_action_levels
            for region_id in range(self.rice.num_regions)
        ]

        self.rice.set_global_state(
            "promised_mitigation_rate", np.array(m1_all_regions), self.rice.timestep
        )
        self.rice.set_global_state(
            "requested_mitigation_rate", np.array(m2_all_regions), self.rice.timestep
        )

        obs = self.rice.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.rice.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info

    def evaluation_step(self, actions=None):
        """
        Update minimum mitigation rates
        """
        assert self.rice.negotiation_on

        assert self.rice.stage == 2

        assert isinstance(actions, dict)
        assert len(actions) == self.rice.num_regions

        action_offset_index = len(
            self.rice.savings_action_nvec
            + self.rice.mitigation_rate_action_nvec
            + self.rice.export_action_nvec
            + self.rice.import_actions_nvec
            + self.rice.tariff_actions_nvec
            + self.rice.proposal_actions_nvec
        )
        num_evaluation_actions = len(self.stages[1]["numberActions"])

        proposal_decisions = np.array(
            [
                actions[region_id][
                    action_offset_index : action_offset_index + num_evaluation_actions
                ]
                for region_id in range(self.rice.num_regions)
            ]
        )
        # Force set the evaluation for own proposal to reject
        for region_id in range(self.rice.num_regions):
            proposal_decisions[region_id, region_id] = 0

        self.rice.set_global_state(
            "proposal_decisions", proposal_decisions, self.rice.timestep
        )

        for region_id in range(self.rice.num_regions):
            outgoing_accepted_mitigation_rates = [
                self.rice.global_state["promised_mitigation_rate"]["value"][
                    self.rice.timestep, region_id, j
                ]
                * self.rice.global_state["proposal_decisions"]["value"][
                    self.rice.timestep, j, region_id
                ]
                for j in range(self.rice.num_regions)
            ]
            incoming_accepted_mitigation_rates = [
                self.rice.global_state["requested_mitigation_rate"]["value"][
                    self.rice.timestep, j, region_id
                ]
                * self.rice.global_state["proposal_decisions"]["value"][
                    self.rice.timestep, region_id, j
                ]
                for j in range(self.rice.num_regions)
            ]

            self.rice.global_state["minimum_mitigation_rate_all_regions"]["value"][
                self.rice.timestep, region_id
            ] = max(
                outgoing_accepted_mitigation_rates + incoming_accepted_mitigation_rates
            )

        obs = self.rice.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.rice.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info

    def check_bilateral_negotiation(self, region_id):

        """
        check if a country is in a bilateral negotiation or not
        """

        # proposal decision of :,region_id would be all it has accepted, any presence of 1 is an acceptance of another states proposal
        accepted = self.rice.global_state["proposal_decisions"]["value"][
            self.rice.timestep, region_id, :
        ]
        # proposal decision of :, region_id would be all it has proposed, any presence of 1 is an acceptance by another state
        accepted_by_other = self.rice.global_state["proposal_decisions"]["value"][
            self.rice.timestep, :, region_id
        ]
        return sum(accepted + accepted_by_other) > 1

    def generate_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.rice.num_regions)}
        for region_id in range(self.rice.num_regions):
            mask = self.rice.default_agent_action_mask.copy()
            if self.rice.negotiation_on:

                # tariff masks
                tariff_masks = []
                for other_region_id in range(self.rice.num_regions):

                    # if other region is self or self not in bilateral negotiation
                    if (other_region_id == region_id) or (
                        not self.check_bilateral_negotiation(region_id)
                    ):

                        # make no change to tariff policy
                        regional_tariff_mask = np.array(
                            [1 for _ in range(self.rice.num_discrete_action_levels)]
                        )

                    # if other region not in bilateral negotiation
                    elif not self.check_bilateral_negotiation(other_region_id):

                        # tariff the 1-mitigation rate
                        region_tariff_rate = int(
                            round(
                                self.rice.num_discrete_action_levels
                                - self.rice.global_state[
                                    "minimum_mitigation_rate_all_regions"
                                ]["value"][self.rice.timestep, other_region_id]
                            )
                        )

                        regional_tariff_mask = np.array(
                            [0 for _ in range(region_tariff_rate)]
                            + [
                                1
                                for _ in range(
                                    self.rice.num_discrete_action_levels
                                    - region_tariff_rate
                                )
                            ]
                        )

                    else:
                        # make no change to tariff policy
                        regional_tariff_mask = np.array(
                            [1 for _ in range(self.rice.num_discrete_action_levels)]
                        )
                    tariff_masks.append(regional_tariff_mask)

                mask_start_tariff = sum(
                    self.rice.savings_action_nvec
                    + self.rice.mitigation_rate_action_nvec
                    + self.rice.export_action_nvec
                    + self.rice.import_actions_nvec
                )

                mask_end_tariff = mask_start_tariff + sum(self.rice.tariff_actions_nvec)
                mask[mask_start_tariff:mask_end_tariff] = np.concatenate(tariff_masks)

            mask_dict[region_id] = mask

        return mask_dict


class BilateralNegotiatorWithTariff(BaseProtocol):

    """
    Updated Bi-lateral negotiation included as the original with the following added:

    If agent is in a bilateral agreement:
    Then they tariff those not in a bilateral agreement

    Action masks:
    - mitigation
    - tariff

    Attributes:
          rice: instance of RICE-N model
          stages (list(dict)): function steps that this negotiator class performs,
                               each contains a function and number of actions
                               used by the function
          num_negotiation_stages (int): number of stages in total

    """

    def __init__(self, rice):
        """
        Defines necessary parameters for communication
        with rice class

        Args:
            Rice: instance of RICE-N model
        """
        self.rice = rice
        self.stages = [
            {
                "function": self.proposal_step,
                "numberActions": (
                    [self.rice.num_discrete_action_levels] * 2 * self.rice.num_regions
                ),
            },
            {
                "function": self.evaluation_step,
                "numberActions": [2] * self.rice.num_regions,
            },
        ]
        self.num_negotiation_stages = len(self.stages)

    def proposal_step(self, actions=None):
        """
        Update Proposal States and Observations using proposal actions
        Update Stage to 1 - Evaluation
        """
        assert self.rice.negotiation_on
        assert self.rice.stage == 1

        assert isinstance(actions, dict)
        assert len(actions) == self.rice.num_regions

        action_offset_index = len(
            self.rice.savings_action_nvec
            + self.rice.mitigation_rate_action_nvec
            + self.rice.export_action_nvec
            + self.rice.import_actions_nvec
            + self.rice.tariff_actions_nvec
        )
        num_proposal_actions = len(self.stages[0]["numberActions"])

        m1_all_regions = [
            actions[region_id][
                action_offset_index : action_offset_index + num_proposal_actions : 2
            ]
            / self.rice.num_discrete_action_levels
            for region_id in range(self.rice.num_regions)
        ]

        m2_all_regions = [
            actions[region_id][
                action_offset_index + 1 : action_offset_index + num_proposal_actions : 2
            ]
            / self.rice.num_discrete_action_levels
            for region_id in range(self.rice.num_regions)
        ]

        self.rice.set_global_state(
            "promised_mitigation_rate", np.array(m1_all_regions), self.rice.timestep
        )
        self.rice.set_global_state(
            "requested_mitigation_rate", np.array(m2_all_regions), self.rice.timestep
        )

        obs = self.rice.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.rice.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info

    def evaluation_step(self, actions=None):
        """
        Update minimum mitigation rates
        """
        assert self.rice.negotiation_on

        assert self.rice.stage == 2

        assert isinstance(actions, dict)
        assert len(actions) == self.rice.num_regions

        action_offset_index = len(
            self.rice.savings_action_nvec
            + self.rice.mitigation_rate_action_nvec
            + self.rice.export_action_nvec
            + self.rice.import_actions_nvec
            + self.rice.tariff_actions_nvec
            + self.rice.proposal_actions_nvec
        )
        num_evaluation_actions = len(self.stages[1]["numberActions"])

        proposal_decisions = np.array(
            [
                actions[region_id][
                    action_offset_index : action_offset_index + num_evaluation_actions
                ]
                for region_id in range(self.rice.num_regions)
            ]
        )
        # Force set the evaluation for own proposal to reject
        for region_id in range(self.rice.num_regions):
            proposal_decisions[region_id, region_id] = 0

        self.rice.set_global_state(
            "proposal_decisions", proposal_decisions, self.rice.timestep
        )

        for region_id in range(self.rice.num_regions):
            outgoing_accepted_mitigation_rates = [
                self.rice.global_state["promised_mitigation_rate"]["value"][
                    self.rice.timestep, region_id, j
                ]
                * self.rice.global_state["proposal_decisions"]["value"][
                    self.rice.timestep, j, region_id
                ]
                for j in range(self.rice.num_regions)
            ]
            incoming_accepted_mitigation_rates = [
                self.rice.global_state["requested_mitigation_rate"]["value"][
                    self.rice.timestep, j, region_id
                ]
                * self.rice.global_state["proposal_decisions"]["value"][
                    self.rice.timestep, region_id, j
                ]
                for j in range(self.rice.num_regions)
            ]

            self.rice.global_state["minimum_mitigation_rate_all_regions"]["value"][
                self.rice.timestep, region_id
            ] = max(
                outgoing_accepted_mitigation_rates + incoming_accepted_mitigation_rates
            )

        obs = self.rice.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.rice.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info

    def check_bilateral_negotiation(self, region_id):

        """
        check if a country is in a bilateral negotiation or not
        """

        # proposal decision of :,region_id would be all it has accepted, any presence of 1 is an acceptance of another states proposal
        accepted = self.rice.global_state["proposal_decisions"]["value"][
            self.rice.timestep, region_id, :
        ]
        # proposal decision of :, region_id would be all it has proposed, any presence of 1 is an acceptance by another state
        accepted_by_other = self.rice.global_state["proposal_decisions"]["value"][
            self.rice.timestep, :, region_id
        ]
        return sum(accepted + accepted_by_other) > 1

    def generate_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.rice.num_regions)}
        for region_id in range(self.rice.num_regions):
            mask = self.rice.default_agent_action_mask.copy()
            if self.rice.negotiation_on:
                minimum_mitigation_rate = int(
                    round(
                        self.rice.global_state["minimum_mitigation_rate_all_regions"][
                            "value"
                        ][self.rice.timestep, region_id]
                        * self.rice.num_discrete_action_levels
                    )
                )
                mitigation_mask = np.array(
                    [0 for _ in range(minimum_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.rice.num_discrete_action_levels
                            - minimum_mitigation_rate
                        )
                    ]
                )
                mask_start = sum(self.rice.savings_action_nvec)
                mask_end = mask_start + sum(self.rice.mitigation_rate_action_nvec)
                mask[mask_start:mask_end] = mitigation_mask

                # tariff masks
                tariff_masks = []
                for other_region_id in range(self.rice.num_regions):

                    # if other region is self or self not in bilateral negotiation
                    if (other_region_id == region_id) or (
                        not self.check_bilateral_negotiation(region_id)
                    ):

                        # make no change to tariff policy
                        regional_tariff_mask = np.array(
                            [1 for _ in range(self.rice.num_discrete_action_levels)]
                        )

                    # if other region not in bilateral negotiation
                    elif not self.check_bilateral_negotiation(other_region_id):

                        # tariff the 1-mitigation rate
                        region_tariff_rate = int(
                            round(
                                self.rice.num_discrete_action_levels
                                - self.rice.global_state[
                                    "minimum_mitigation_rate_all_regions"
                                ]["value"][self.rice.timestep, other_region_id]
                            )
                        )

                        regional_tariff_mask = np.array(
                            [0 for _ in range(region_tariff_rate)]
                            + [
                                1
                                for _ in range(
                                    self.rice.num_discrete_action_levels
                                    - region_tariff_rate
                                )
                            ]
                        )

                    else:
                        # make no change to tariff policy
                        regional_tariff_mask = np.array(
                            [1 for _ in range(self.rice.num_discrete_action_levels)]
                        )
                    tariff_masks.append(regional_tariff_mask)

                mask_start_tariff = sum(
                    self.rice.savings_action_nvec
                    + self.rice.mitigation_rate_action_nvec
                    + self.rice.export_action_nvec
                    + self.rice.import_actions_nvec
                )

                mask_end_tariff = mask_start_tariff + sum(self.rice.tariff_actions_nvec)
                mask[mask_start_tariff:mask_end_tariff] = np.concatenate(tariff_masks)

            mask_dict[region_id] = mask

        return mask_dict

class BasicClubDiscreteDefectClusterProposals(BaseProtocol):

    """
    Basic Climate Club. Works as follows:
    Each country proposes 1 value mitigation level, the level they think others should aspire to.
    this value is either accepted or rejected by other countries. 
    Countries commit to the max that they accept. 
    They tariff those in lower mitigation clubs than themselves and give bonus to those in the same club as themselves.

    Action masks:
    - mitigation
    - tariffs, inverse tarrifs within club, normal tariffs 

    Attributes:
          rice: instance of RICE-N model
          stages (list(dict)): function steps that this negotiator class performs,
                               each contains a function and number of actions
                               used by the function
          num_negotiation_stages (int): number of stages in total

    """

    def __init__(self, rice):
        """
        Defines necessary parameters for communication
        with rice class

        Args:
            Rice: instance of RICE-N model
        """
        self.rice = rice
        self.stages = [
            {
                "function": self.proposal_step,
                "numberActions": (
                    [self.rice.num_discrete_action_levels]
                ),
            },
            {
                "function": self.evaluation_step,
                "numberActions": [2] * self.rice.num_regions,
            },
            {
                "function": self.defect_step,
                "numberActions": [2]
            }
        ]
        self.num_negotiation_stages = len(self.stages)

    def reset(self):
        """
        Add any negotiator specific values to global state
        """

        for key in [            
            "proposed_mitigation_rate",
            "defect_decisions"
            ]:
            self.rice.set_global_state(
                key = key,
                value = np.zeros(self.rice.num_regions),
                timestep = self.rice.timestep,
            )

    def cluster_proposals(self, x):
        sil = []
        kmax = len(set(x.squeeze().tolist()))#int(x.shape[0]/2)  # kmax determines the number of clusters
        ks = []
        if len(set(x.squeeze().tolist())) == 1:
            return x.squeeze().tolist()
        for k in range(2, kmax + 1):
            ks.append(k)
            kmeans = KMeans(n_clusters=k, init='k-means++',
                            n_init=10, max_iter=300, 
                            tol=1e-04, random_state=42
                        ).fit(x)  
            labels = (
                kmeans.labels_
            ) 
            sil.append(silhouette_score(x, labels, metric="euclidean"))
        k = ks[sil.index(max(sil))]

        kmeans = KMeans(n_clusters=k, init='k-means++',
                                n_init=2, max_iter=300, 
                                tol=1e-04, random_state=42
                            ).fit(x)
        labels = kmeans.labels_

        centroids = kmeans.cluster_centers_.squeeze()
        clustered_proposals = [centroids[label] for label in labels]


        return clustered_proposals

    

    def proposal_step(self, actions=None):
        """
        Update Proposal States and Observations using proposal actions
        Update Stage to 1 - Evaluation
        """
        assert self.rice.negotiation_on
        assert self.rice.stage == 1

        assert isinstance(actions, dict)
        assert len(actions) == self.rice.num_regions

        action_offset_index = len(
            self.rice.savings_action_nvec
            + self.rice.mitigation_rate_action_nvec
            + self.rice.export_action_nvec
            + self.rice.import_actions_nvec
            + self.rice.tariff_actions_nvec
        )

        #each country proposes a mitigation rate 0-num_discrete_action_levels (10)
        proposed_mitigation_rate = [
            actions[region_id][
                action_offset_index
            ]
            / self.rice.num_discrete_action_levels
            for region_id in range(self.rice.num_regions)
        ]

        clusted_proposals = self.cluster_proposals(np.array(proposed_mitigation_rate).reshape(-1,1))

        self.rice.set_global_state(
            "proposed_mitigation_rate", np.array(clusted_proposals), self.rice.timestep
        )
        

        obs = self.rice.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.rice.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info

    def evaluation_step(self, actions=None):
        """
        Update minimum mitigation rates
        """
        assert self.rice.negotiation_on

        assert self.rice.stage == 2

        assert isinstance(actions, dict)
        assert len(actions) == self.rice.num_regions

        action_offset_index = len(
            self.rice.savings_action_nvec
            + self.rice.mitigation_rate_action_nvec
            + self.rice.export_action_nvec
            + self.rice.import_actions_nvec
            + self.rice.tariff_actions_nvec
            + self.rice.proposal_actions_nvec
        )
        num_evaluation_actions = len(self.stages[1]["numberActions"])

        proposal_decisions = np.array(
            [
                actions[region_id][
                    action_offset_index : action_offset_index + num_evaluation_actions
                ]
                for region_id in range(self.rice.num_regions)
            ]
        )
        # Force set the evaluation for own proposal to accept
        for region_id in range(self.rice.num_regions):
            proposal_decisions[region_id, region_id] = 1

        #update global states
        self.rice.set_global_state("proposal_decisions", proposal_decisions, self.rice.timestep)

        proposed_mitigation_rates = np.array([self.rice.global_state["proposed_mitigation_rate"]["value"][
                self.rice.timestep, j
            ] for j in range(self.rice.num_regions)])
        
        for region_id in range(self.rice.num_regions):

            result_mitigation = proposed_mitigation_rates * proposal_decisions[region_id, :]
            self.rice.global_state["minimum_mitigation_rate_all_regions"]["value"][
                self.rice.timestep, region_id
            ] = max(
                result_mitigation
            )
        obs = self.rice.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.rice.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info

    def defect_step(self, actions = None):

        """
        Decide to reset mitigation rates to 0
        """

        assert self.rice.negotiation_on

        assert self.rice.stage == 3

        assert isinstance(actions, dict)
        assert len(actions) == self.rice.num_regions
        #offsets
        action_offset_index = len(
            self.rice.savings_action_nvec
            + self.rice.mitigation_rate_action_nvec
            + self.rice.export_action_nvec
            + self.rice.import_actions_nvec
            + self.rice.tariff_actions_nvec
            + self.rice.proposal_actions_nvec
            + self.stages[1]["numberActions"]
        )
        num_defect_actions = len(self.stages[2]["numberActions"])

        #extract decisions from action vector
        defect_decisions = np.array(
            [
                actions[region_id][
                    action_offset_index : action_offset_index + num_defect_actions
                ]
                for region_id in range(self.rice.num_regions)
            ]
        ).squeeze()

        #update global states
        self.rice.set_global_state("defect_decisions", defect_decisions, self.rice.timestep)



        
        for region_id in range(self.rice.num_regions):

            #if defect cancel mitigation mask
            if defect_decisions[region_id] == 1:
                self.rice.global_state["minimum_mitigation_rate_all_regions"]["value"][
                    self.rice.timestep, region_id
                    ] = 0
            else:
                pass


        obs = self.rice.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.rice.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info



    def generate_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.rice.num_regions)}
        for region_id in range(self.rice.num_regions):
            mask = self.rice.default_agent_action_mask.copy()
            if self.rice.negotiation_on:

                #mask mitigation as per own club
                minimum_mitigation_rate = int(
                    round(
                        self.rice.global_state["minimum_mitigation_rate_all_regions"][
                            "value"
                        ][self.rice.timestep, region_id]
                        * self.rice.num_discrete_action_levels
                    )
                )
                mitigation_mask = np.array(
                    [0 for _ in range(minimum_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.rice.num_discrete_action_levels
                            - minimum_mitigation_rate
                        )
                    ]
                )
                mask_start = sum(self.rice.savings_action_nvec)
                mask_end = mask_start + sum(self.rice.mitigation_rate_action_nvec)
                mask[mask_start:mask_end] = mitigation_mask

                # tariff masks
                tariff_masks = []
                for other_region_id in range(self.rice.num_regions):

                    other_region_mitigation_rate = int(round(self.rice.global_state["minimum_mitigation_rate_all_regions"]["value"][self.rice.timestep, other_region_id]\
                                                     * self.rice.num_discrete_action_levels))

                    # if other region is self or self not in bilateral negotiation
                    if (other_region_id == region_id):

                        # make no change to tariff policy
                        regional_tariff_mask = np.array(
                            [1 for _ in range(self.rice.num_discrete_action_levels)]
                        )

                    # if other region's mitigation rate less than yours
                    elif other_region_mitigation_rate < minimum_mitigation_rate:

                        # tariff the 1-mitigation rate
                        region_tariff_rate = int(
                            round(
                                self.rice.num_discrete_action_levels
                                - self.rice.global_state[
                                    "minimum_mitigation_rate_all_regions"
                                ]["value"][self.rice.timestep, other_region_id]
                            )
                        )

                        regional_tariff_mask = np.array(
                            [0 for _ in range(region_tariff_rate)]
                            + [
                                1
                                for _ in range(
                                    self.rice.num_discrete_action_levels
                                    - region_tariff_rate
                                )
                            ]
                        )

                    # if other regions mitigation >= your own bonus
                    else:
                        # set tarrif cap
                        region_tariff_rate = int(
                            round(
                                self.rice.num_discrete_action_levels
                                - self.rice.global_state[
                                    "minimum_mitigation_rate_all_regions"
                                ]["value"][self.rice.timestep, other_region_id]
                            )
                        )

                        regional_tariff_mask = np.array(
                            [1 for _ in range(region_tariff_rate)]
                            + [
                                0
                                for _ in range(
                                    self.rice.num_discrete_action_levels
                                    - region_tariff_rate
                                )
                            ]
                        )
                    tariff_masks.append(regional_tariff_mask)

                mask_start_tariff = sum(
                    self.rice.savings_action_nvec
                    + self.rice.mitigation_rate_action_nvec
                    + self.rice.export_action_nvec
                    + self.rice.import_actions_nvec
                )

                mask_end_tariff = mask_start_tariff + sum(self.rice.tariff_actions_nvec)
                mask[mask_start_tariff:mask_end_tariff] = np.concatenate(tariff_masks)

            mask_dict[region_id] = mask

        return mask_dict

class BasicClubClusterProposals(BaseProtocol):

    """
    Basic Climate Club. Works as follows:
    Each country proposes 1 value mitigation level, the level they think others should aspire to.
    this value is either accepted or rejected by other countries. 
    Countries commit to the max that they accept. 
    They tariff those in lower mitigation clubs than themselves and give bonus to those in the same club as themselves.

    Action masks:
    - mitigation
    - tariffs, inverse tarrifs within club, normal tariffs 

    Attributes:
          rice: instance of RICE-N model
          stages (list(dict)): function steps that this negotiator class performs,
                               each contains a function and number of actions
                               used by the function
          num_negotiation_stages (int): number of stages in total

    """

    def __init__(self, rice):
        """
        Defines necessary parameters for communication
        with rice class

        Args:
            Rice: instance of RICE-N model
        """
        self.rice = rice
        self.stages = [
            {
                "function": self.proposal_step,
                "numberActions": (
                    [self.rice.num_discrete_action_levels]
                ),
            },
            {
                "function": self.evaluation_step,
                "numberActions": [2] * self.rice.num_regions,
            }
        ]
        self.num_negotiation_stages = len(self.stages)

    def reset(self):
        """
        Add any negotiator specific values to global state
        """

        for key in [            
            "proposed_mitigation_rate"
            ]:
            self.rice.set_global_state(
                key = key,
                value = np.zeros(self.rice.num_regions),
                timestep = self.rice.timestep,
            )

    def cluster_proposals(self, x):
        sil = []
        kmax = len(set(x.squeeze().tolist()))#int(x.shape[0]/2)  # kmax determines the number of clusters
        ks = []
        if len(set(x.squeeze().tolist())) == 1:
            return x.squeeze().tolist()
        for k in range(2, kmax + 1):
            ks.append(k)
            kmeans = KMeans(n_clusters=k, init='k-means++',
                            n_init=10, max_iter=300, 
                            tol=1e-04, random_state=42
                        ).fit(x)  
            labels = (
                kmeans.labels_
            ) 
            sil.append(silhouette_score(x, labels, metric="euclidean"))
        k = ks[sil.index(max(sil))]

        kmeans = KMeans(n_clusters=k, init='k-means++',
                                n_init=2, max_iter=300, 
                                tol=1e-04, random_state=42
                            ).fit(x)
        labels = kmeans.labels_

        centroids = kmeans.cluster_centers_.squeeze()
        clustered_proposals = [centroids[label] for label in labels]


        return clustered_proposals

    

    def proposal_step(self, actions=None):
        """
        Update Proposal States and Observations using proposal actions
        Update Stage to 1 - Evaluation
        """
        assert self.rice.negotiation_on
        assert self.rice.stage == 1

        assert isinstance(actions, dict)
        assert len(actions) == self.rice.num_regions

        action_offset_index = len(
            self.rice.savings_action_nvec
            + self.rice.mitigation_rate_action_nvec
            + self.rice.export_action_nvec
            + self.rice.import_actions_nvec
            + self.rice.tariff_actions_nvec
        )

        #each country proposes a mitigation rate 0-num_discrete_action_levels (10)
        proposed_mitigation_rate = [
            actions[region_id][
                action_offset_index
            ]
            / self.rice.num_discrete_action_levels
            for region_id in range(self.rice.num_regions)
        ]

        clusted_proposals = self.cluster_proposals(np.array(proposed_mitigation_rate).reshape(-1,1))

        self.rice.set_global_state(
            "proposed_mitigation_rate", np.array(clusted_proposals), self.rice.timestep
        )
        

        obs = self.rice.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.rice.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info

    def evaluation_step(self, actions=None):
        """
        Update minimum mitigation rates
        """
        assert self.rice.negotiation_on

        assert self.rice.stage == 2

        assert isinstance(actions, dict)
        assert len(actions) == self.rice.num_regions

        action_offset_index = len(
            self.rice.savings_action_nvec
            + self.rice.mitigation_rate_action_nvec
            + self.rice.export_action_nvec
            + self.rice.import_actions_nvec
            + self.rice.tariff_actions_nvec
            + self.rice.proposal_actions_nvec
        )
        num_evaluation_actions = len(self.stages[1]["numberActions"])

        proposal_decisions = np.array(
            [
                actions[region_id][
                    action_offset_index : action_offset_index + num_evaluation_actions
                ]
                for region_id in range(self.rice.num_regions)
            ]
        )
        # Force set the evaluation for own proposal to accept
        for region_id in range(self.rice.num_regions):
            proposal_decisions[region_id, region_id] = 1

        #update global states
        self.rice.set_global_state("proposal_decisions", proposal_decisions, self.rice.timestep)

        proposed_mitigation_rates = np.array([self.rice.global_state["proposed_mitigation_rate"]["value"][
                self.rice.timestep, j
            ] for j in range(self.rice.num_regions)])
        
        for region_id in range(self.rice.num_regions):

            result_mitigation = proposed_mitigation_rates * proposal_decisions[region_id, :]
            self.rice.global_state["minimum_mitigation_rate_all_regions"]["value"][
                self.rice.timestep, region_id
            ] = max(
                result_mitigation
            )
        obs = self.rice.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.rice.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info




    def generate_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.rice.num_regions)}
        for region_id in range(self.rice.num_regions):
            mask = self.rice.default_agent_action_mask.copy()
            if self.rice.negotiation_on:

                #mask mitigation as per own club
                minimum_mitigation_rate = int(
                    round(
                        self.rice.global_state["minimum_mitigation_rate_all_regions"][
                            "value"
                        ][self.rice.timestep, region_id]
                        * self.rice.num_discrete_action_levels
                    )
                )
                mitigation_mask = np.array(
                    [0 for _ in range(minimum_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.rice.num_discrete_action_levels
                            - minimum_mitigation_rate
                        )
                    ]
                )
                mask_start = sum(self.rice.savings_action_nvec)
                mask_end = mask_start + sum(self.rice.mitigation_rate_action_nvec)
                mask[mask_start:mask_end] = mitigation_mask

                # tariff masks
                tariff_masks = []
                for other_region_id in range(self.rice.num_regions):

                    other_region_mitigation_rate = int(round(self.rice.global_state["minimum_mitigation_rate_all_regions"]["value"][self.rice.timestep, other_region_id]\
                                                     * self.rice.num_discrete_action_levels))

                    # if other region is self or self not in bilateral negotiation
                    if (other_region_id == region_id):

                        # make no change to tariff policy
                        regional_tariff_mask = np.array(
                            [1 for _ in range(self.rice.num_discrete_action_levels)]
                        )

                    # if other region's mitigation rate less than yours
                    elif other_region_mitigation_rate < minimum_mitigation_rate:

                        # tariff the 1-mitigation rate
                        region_tariff_rate = int(
                            round(
                                self.rice.num_discrete_action_levels
                                - self.rice.global_state[
                                    "minimum_mitigation_rate_all_regions"
                                ]["value"][self.rice.timestep, other_region_id]
                            )
                        )

                        regional_tariff_mask = np.array(
                            [0 for _ in range(region_tariff_rate)]
                            + [
                                1
                                for _ in range(
                                    self.rice.num_discrete_action_levels
                                    - region_tariff_rate
                                )
                            ]
                        )

                    # if other regions mitigation >= your own bonus
                    else:
                        # set tarrif cap
                        region_tariff_rate = int(
                            round(
                                self.rice.num_discrete_action_levels
                                - self.rice.global_state[
                                    "minimum_mitigation_rate_all_regions"
                                ]["value"][self.rice.timestep, other_region_id]
                            )
                        )

                        regional_tariff_mask = np.array(
                            [1 for _ in range(region_tariff_rate)]
                            + [
                                0
                                for _ in range(
                                    self.rice.num_discrete_action_levels
                                    - region_tariff_rate
                                )
                            ]
                        )
                    tariff_masks.append(regional_tariff_mask)

                mask_start_tariff = sum(
                    self.rice.savings_action_nvec
                    + self.rice.mitigation_rate_action_nvec
                    + self.rice.export_action_nvec
                    + self.rice.import_actions_nvec
                )

                mask_end_tariff = mask_start_tariff + sum(self.rice.tariff_actions_nvec)
                mask[mask_start_tariff:mask_end_tariff] = np.concatenate(tariff_masks)

            mask_dict[region_id] = mask

        return mask_dict
    
class BasicClubDiscreteDefectWithPunishmentAndFreeTrade(BaseProtocol):

    """
    Basic Climate Club. Works as follows:
    Each country proposes 1 value mitigation level, the level they think others should aspire to.
    this value is either accepted or rejected by other countries. 
    Countries commit to the max that they accept. 
    They tariff those in lower mitigation clubs than themselves and give bonus to those in the same club as themselves.

    Action masks:
    - mitigation
    - tariffs, inverse tarrifs within club, normal tariffs 

    Attributes:
          rice: instance of RICE-N model
          stages (list(dict)): function steps that this negotiator class performs,
                               each contains a function and number of actions
                               used by the function
          num_negotiation_stages (int): number of stages in total

    """

    def __init__(self, num_regions, num_discrete_action_levels):
        """
        Defines necessary parameters for communication
        with rice class

        Args:
            Rice: instance of RICE-N model
        """
        self.stages = [
            {
                "function": self.proposal_step,
                "action_space": [num_discrete_action_levels],
            },
            {
                "function": self.evaluation_step,
                "action_space": [2] * num_regions,
            },
            {
                "function": self.defect_step,
                "action_space": [2]
            }
        ]
        super().__init__(num_regions, num_discrete_action_levels)

    def reset(self):
        """
        Add any negotiator specific values to global state
        """
        self.minimum_mitigation_rate_all_regions = np.zeros(self.num_regions)
        self.proposed_mitigation_rate = np.zeros(self.num_regions)
        self.defect_decisions = np.zeros(self.num_regions)
        self.proposal_decisions = np.zeros((self.num_regions, self.num_regions))
        self.defectors = np.zeros(self.num_regions)


    def get_protocol_state(self):
        protocol_state = {
            "stage": np.array([self.stage_idx]) / self.num_stages,
            "proposed_mitigation_rate": self.proposed_mitigation_rate
            / self.num_discrete_action_levels,
            "minimum_mitigation_rate_all_regions": self.minimum_mitigation_rate_all_regions
            / self.num_discrete_action_levels,
            "defect_decisions": self.defect_decisions,
            "proposal_decisions": self.proposal_decisions,
            "received_proposal_decisions": self.proposal_decisions.T,
            "defectors":self.defectors
        }
        return protocol_state

    def get_pub_priv_features(self):
        public_features = ["stage", "proposed_mitigation_rate", "minimum_mitigation_rate_all_regions", "defect_decisions", "defectors"]
        private_features = [
            # "proposal_decisions",
            "received_proposal_decisions",
        ]
        return public_features, private_features

    def proposal_step(self, actions: dict):
        """
        Update Proposal States and Observations using proposal actions
        Update Stage to 1 - Evaluation
        """
        assert self.stage_idx == 0

        #each country proposes a mitigation rate 0-num_discrete_action_levels (10)
        self.proposed_mitigation_rate = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        ).squeeze()

    def evaluation_step(self, actions: dict):
        """
        Update minimum mitigation rates
        """
        assert self.stage_idx == 1

        self.proposal_decisions = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        )
        # Force set the evaluation for own proposal to accept
        for region_id in range(self.num_regions):
            self.proposal_decisions[region_id, region_id] = 1

        
        mmrar = (self.proposed_mitigation_rate * self.proposal_decisions).max(axis=1)
        self.minimum_mitigation_rate_all_regions = mmrar


    def defect_step(self, actions: dict):
        """
        Decide to reset mitigation rates to 0
        """
        assert self.stage_idx == 2

        #extract decisions from action vector
        self.defect_decisions = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        ).squeeze()

        #keep track of who has defected
        for region_id in range(self.num_regions):
            if actions[region_id] == 1:
                self.defectors[region_id] = 1
            else:
                pass

        self.minimum_mitigation_rate_all_regions = self.minimum_mitigation_rate_all_regions * (1 - self.defect_decisions)


    def get_partial_action_mask(self):
        """
        Generate action masks.
        """
        action_mask_dict = defaultdict(dict)
        for region_id in range(self.num_regions):
            minimum_mitigation_rate = int(self.minimum_mitigation_rate_all_regions[region_id])
            mitigation_mask = [0] * int(minimum_mitigation_rate) + [1] * int(
                self.num_discrete_action_levels - minimum_mitigation_rate
            )

            # tariff masks
            tariff_mask = []
            for other_region_id in range(self.num_regions):
                other_region_mitigation_rate = int(self.minimum_mitigation_rate_all_regions[other_region_id])

                # if other region is self 
                if (other_region_id == region_id):
                    # make no change to tariff policy
                    regional_tariff_mask = [1] * self.num_discrete_action_levels

                #if region has ever defected, max tariff
                elif self.defectors[other_region_id] == 1:
                    regional_tariff_mask = [0]*(self.num_discrete_action_levels - 1)+[1]

                # if other region's mitigation rate less than yours
                elif other_region_mitigation_rate < minimum_mitigation_rate:
                    # tariff the 1-mitigation rate
                    region_tariff_rate = self.num_discrete_action_levels - other_region_mitigation_rate
                    regional_tariff_mask = [0] * region_tariff_rate + [1] * other_region_mitigation_rate

                # if other regions mitigation >= your own bonus: free trade
                else:
                    # set tarrif cap
                    regional_tariff_mask = [1]+ [0]*(self.num_discrete_action_levels - 1)

                tariff_mask.extend(regional_tariff_mask)

            action_mask_dict[region_id]["tariff"] = tariff_mask
            action_mask_dict[region_id]["mitigation"] = mitigation_mask

        return action_mask_dict
    
class BasicClubDiscreteDefectWithPunishment(BaseProtocol):

    """
    Basic Climate Club. Works as follows:
    Each country proposes 1 value mitigation level, the level they think others should aspire to.
    this value is either accepted or rejected by other countries. 
    Countries commit to the max that they accept. 
    They tariff those in lower mitigation clubs than themselves and give bonus to those in the same club as themselves.

    Action masks:
    - mitigation
    - tariffs, inverse tarrifs within club, normal tariffs 

    Attributes:
          rice: instance of RICE-N model
          stages (list(dict)): function steps that this negotiator class performs,
                               each contains a function and number of actions
                               used by the function
          num_negotiation_stages (int): number of stages in total

    """

    def __init__(self, num_regions, num_discrete_action_levels):
        """
        Defines necessary parameters for communication
        with rice class

        Args:
            Rice: instance of RICE-N model
        """
        self.stages = [
            {
                "function": self.proposal_step,
                "action_space": [num_discrete_action_levels],
            },
            {
                "function": self.evaluation_step,
                "action_space": [2] * num_regions,
            },
            {
                "function": self.defect_step,
                "action_space": [2]
            }
        ]
        super().__init__(num_regions, num_discrete_action_levels)

    def reset(self):
        """
        Add any negotiator specific values to global state
        """
        self.minimum_mitigation_rate_all_regions = np.zeros(self.num_regions)
        self.proposed_mitigation_rate = np.zeros(self.num_regions)
        self.defect_decisions = np.zeros(self.num_regions)
        self.proposal_decisions = np.zeros((self.num_regions, self.num_regions))
        self.defectors = np.zeros(self.num_regions)


    def get_protocol_state(self):
        protocol_state = {
            "stage": np.array([self.stage_idx]) / self.num_stages,
            "proposed_mitigation_rate": self.proposed_mitigation_rate
            / self.num_discrete_action_levels,
            "minimum_mitigation_rate_all_regions": self.minimum_mitigation_rate_all_regions
            / self.num_discrete_action_levels,
            "defect_decisions": self.defect_decisions,
            "proposal_decisions": self.proposal_decisions,
            "received_proposal_decisions": self.proposal_decisions.T,
            "defectors":self.defectors
        }
        return protocol_state

    def get_pub_priv_features(self):
        public_features = ["stage", "proposed_mitigation_rate", "minimum_mitigation_rate_all_regions", "defect_decisions", "defectors"]
        private_features = [
            # "proposal_decisions",
            "received_proposal_decisions",
        ]
        return public_features, private_features

    def proposal_step(self, actions: dict):
        """
        Update Proposal States and Observations using proposal actions
        Update Stage to 1 - Evaluation
        """
        assert self.stage_idx == 0

        #each country proposes a mitigation rate 0-num_discrete_action_levels (10)
        self.proposed_mitigation_rate = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        ).squeeze()

    def evaluation_step(self, actions: dict):
        """
        Update minimum mitigation rates
        """
        assert self.stage_idx == 1

        self.proposal_decisions = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        )
        # Force set the evaluation for own proposal to accept
        for region_id in range(self.num_regions):
            self.proposal_decisions[region_id, region_id] = 1

        
        mmrar = (self.proposed_mitigation_rate * self.proposal_decisions).max(axis=1)
        self.minimum_mitigation_rate_all_regions = mmrar


    def defect_step(self, actions: dict):
        """
        Decide to reset mitigation rates to 0
        """
        assert self.stage_idx == 2

        #extract decisions from action vector
        self.defect_decisions = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        ).squeeze()

        #keep track of who has defected
        for region_id in range(self.num_regions):
            if actions[region_id] == 1:
                self.defectors[region_id] = 1
            else:
                pass

        self.minimum_mitigation_rate_all_regions = self.minimum_mitigation_rate_all_regions * (1 - self.defect_decisions)


    def get_partial_action_mask(self):
        """
        Generate action masks.
        """
        action_mask_dict = defaultdict(dict)
        for region_id in range(self.num_regions):
            minimum_mitigation_rate = int(self.minimum_mitigation_rate_all_regions[region_id])
            mitigation_mask = [0] * int(minimum_mitigation_rate) + [1] * int(
                self.num_discrete_action_levels - minimum_mitigation_rate
            )

            # tariff masks
            tariff_mask = []
            for other_region_id in range(self.num_regions):
                other_region_mitigation_rate = int(self.minimum_mitigation_rate_all_regions[other_region_id])

                # if other region is self 
                if (other_region_id == region_id):
                    # make no change to tariff policy
                    regional_tariff_mask = [1] * self.num_discrete_action_levels

                #if region has ever defected, max tariff!
                elif self.defectors[other_region_id] == 1:
                    regional_tariff_mask = [0]*(self.num_discrete_action_levels - 1)+[1]

                # if other region's mitigation rate less than yours
                elif other_region_mitigation_rate < minimum_mitigation_rate:
                    # tariff the 1-mitigation rate
                    region_tariff_rate = self.num_discrete_action_levels - other_region_mitigation_rate
                    regional_tariff_mask = [0] * region_tariff_rate + [1] * other_region_mitigation_rate

                # if other regions mitigation >= your own bonus
                else:
                    # set tarrif cap
                    region_tariff_rate = self.num_discrete_action_levels - other_region_mitigation_rate
                    regional_tariff_mask = [1] * region_tariff_rate + [0] * other_region_mitigation_rate

                tariff_mask.extend(regional_tariff_mask)

            action_mask_dict[region_id]["tariff"] = tariff_mask
            action_mask_dict[region_id]["mitigation"] = mitigation_mask

        return action_mask_dict

class BasicClubDiscreteDefect(BaseProtocol):

    """
    Basic Climate Club. Works as follows:
    Each country proposes 1 value mitigation level, the level they think others should aspire to.
    this value is either accepted or rejected by other countries. 
    Countries commit to the max that they accept. 
    They tariff those in lower mitigation clubs than themselves and give bonus to those in the same club as themselves.

    Action masks:
    - mitigation
    - tariffs, inverse tarrifs within club, normal tariffs 

    Attributes:
          rice: instance of RICE-N model
          stages (list(dict)): function steps that this negotiator class performs,
                               each contains a function and number of actions
                               used by the function
          num_negotiation_stages (int): number of stages in total

    """

    def __init__(self, num_regions, num_discrete_action_levels):
        """
        Defines necessary parameters for communication
        with rice class

        Args:
            Rice: instance of RICE-N model
        """
        self.stages = [
            {
                "function": self.proposal_step,
                "action_space": [num_discrete_action_levels],
            },
            {
                "function": self.evaluation_step,
                "action_space": [2] * num_regions,
            },
            {
                "function": self.defect_step,
                "action_space": [2]
            }
        ]
        super().__init__(num_regions, num_discrete_action_levels)

    def reset(self):
        """
        Add any negotiator specific values to global state
        """
        self.minimum_mitigation_rate_all_regions = np.zeros(self.num_regions)
        self.proposed_mitigation_rate = np.zeros(self.num_regions)
        self.defect_decisions = np.zeros(self.num_regions)
        self.proposal_decisions = np.zeros((self.num_regions, self.num_regions))


    def get_protocol_state(self):
        protocol_state = {
            "stage": np.array([self.stage_idx]) / self.num_stages,
            "proposed_mitigation_rate": self.proposed_mitigation_rate
            / self.num_discrete_action_levels,
            "minimum_mitigation_rate_all_regions": self.minimum_mitigation_rate_all_regions
            / self.num_discrete_action_levels,
            "defect_decisions": self.defect_decisions,
            "proposal_decisions": self.proposal_decisions,
            "received_proposal_decisions": self.proposal_decisions.T,
        }
        return protocol_state

    def get_pub_priv_features(self):
        public_features = ["stage", "proposed_mitigation_rate", "minimum_mitigation_rate_all_regions", "defect_decisions"]
        private_features = [
            # "proposal_decisions",
            "received_proposal_decisions",
        ]
        return public_features, private_features

    def proposal_step(self, actions: dict):
        """
        Update Proposal States and Observations using proposal actions
        Update Stage to 1 - Evaluation
        """
        assert self.stage_idx == 0

        #each country proposes a mitigation rate 0-num_discrete_action_levels (10)
        self.proposed_mitigation_rate = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        ).squeeze()

    def evaluation_step(self, actions: dict):
        """
        Update minimum mitigation rates
        """
        assert self.stage_idx == 1

        self.proposal_decisions = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        )
        # Force set the evaluation for own proposal to accept
        for region_id in range(self.num_regions):
            self.proposal_decisions[region_id, region_id] = 1

        
        mmrar = (self.proposed_mitigation_rate * self.proposal_decisions).max(axis=1)
        self.minimum_mitigation_rate_all_regions = mmrar


    def defect_step(self, actions: dict):
        """
        Decide to reset mitigation rates to 0
        """
        assert self.stage_idx == 2

        #extract decisions from action vector
        self.defect_decisions = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        ).squeeze()

        self.minimum_mitigation_rate_all_regions = self.minimum_mitigation_rate_all_regions * (1 - self.defect_decisions)


    def get_partial_action_mask(self):
        """
        Generate action masks.
        """
        action_mask_dict = defaultdict(dict)
        for region_id in range(self.num_regions):
            minimum_mitigation_rate = int(self.minimum_mitigation_rate_all_regions[region_id])
            mitigation_mask = [0] * int(minimum_mitigation_rate) + [1] * int(
                self.num_discrete_action_levels - minimum_mitigation_rate
            )

            # tariff masks
            tariff_mask = []
            for other_region_id in range(self.num_regions):
                other_region_mitigation_rate = int(self.minimum_mitigation_rate_all_regions[other_region_id])

                # if other region is self or self not in bilateral negotiation
                if (other_region_id == region_id):
                    # make no change to tariff policy
                    regional_tariff_mask = [1] * self.num_discrete_action_levels

                # if other region's mitigation rate less than yours
                elif other_region_mitigation_rate < minimum_mitigation_rate:
                    # tariff the 1-mitigation rate
                    region_tariff_rate = self.num_discrete_action_levels - other_region_mitigation_rate
                    regional_tariff_mask = [0] * region_tariff_rate + [1] * other_region_mitigation_rate

                # if other regions mitigation >= your own bonus
                else:
                    # set tarrif cap
                    region_tariff_rate = self.num_discrete_action_levels - other_region_mitigation_rate
                    regional_tariff_mask = [1] * region_tariff_rate + [0] * other_region_mitigation_rate

                tariff_mask.extend(regional_tariff_mask)

            action_mask_dict[region_id]["tariff"] = tariff_mask
            action_mask_dict[region_id]["mitigation"] = mitigation_mask

        return action_mask_dict

class BasicClub(BaseProtocol):

    """
    Basic Climate Club. Works as follows:
    Each country proposes 1 value mitigation level, the level they think others should aspire to.
    this value is either accepted or rejected by other countries. 
    Countries commit to the max that they accept. 
    They tariff those in lower mitigation clubs than themselves and give bonus to those in the same club as themselves.

    Action masks:
    - mitigation
    - tariffs, inverse tarrifs within club, normal tariffs 

    Attributes:
          rice: instance of RICE-N model
          stages (list(dict)): function steps that this negotiator class performs,
                               each contains a function and number of actions
                               used by the function
          num_negotiation_stages (int): number of stages in total

    """

    def __init__(self, num_regions, num_discrete_action_levels):
        """
        Defines necessary parameters for communication
        with rice class

        Args:
            Rice: instance of RICE-N model
        """
        self.stages = [
            {
                "function": self.proposal_step,
                "action_space": [num_discrete_action_levels],
            },
            {
                "function": self.evaluation_step,
                "action_space": [2] * num_regions,
            },
        ]
        super().__init__(num_regions, num_discrete_action_levels)

    def reset(self):
        """
        Add any negotiator specific values to global state
        """
        self.minimum_mitigation_rate_all_regions = np.zeros(self.num_regions)
        self.proposed_mitigation_rate = np.zeros(self.num_regions)
        self.proposal_decisions = np.zeros((self.num_regions, self.num_regions))


    def get_protocol_state(self):
        protocol_state = {
            "stage": np.array([self.stage_idx]) / self.num_stages,
            "proposed_mitigation_rate": self.proposed_mitigation_rate
            / self.num_discrete_action_levels,
            "minimum_mitigation_rate_all_regions": self.minimum_mitigation_rate_all_regions
            / self.num_discrete_action_levels,
            "proposal_decisions": self.proposal_decisions,
            "received_proposal_decisions": self.proposal_decisions.T,
        }
        return protocol_state

    def get_pub_priv_features(self):
        public_features = ["stage", "proposed_mitigation_rate", "minimum_mitigation_rate_all_regions"]
        private_features = [
            # "proposal_decisions",
            "received_proposal_decisions",
        ]
        return public_features, private_features

    def proposal_step(self, actions: dict):
        """
        Update Proposal States and Observations using proposal actions
        Update Stage to 1 - Evaluation
        """
        assert self.stage_idx == 0

        #each country proposes a mitigation rate 0-num_discrete_action_levels (10)
        self.proposed_mitigation_rate = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        ).squeeze()

    def evaluation_step(self, actions: dict):
        """
        Update minimum mitigation rates
        """
        assert self.stage_idx == 1

        self.proposal_decisions = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        )
        # Force set the evaluation for own proposal to accept
        for region_id in range(self.num_regions):
            self.proposal_decisions[region_id, region_id] = 1

        
        mmrar = (self.proposed_mitigation_rate * self.proposal_decisions).max(axis=1)
        self.minimum_mitigation_rate_all_regions = mmrar

    def get_partial_action_mask(self):
        """
        Generate action masks.
        """
        action_mask_dict = defaultdict(dict)
        for region_id in range(self.num_regions):
            minimum_mitigation_rate = int(self.minimum_mitigation_rate_all_regions[region_id])
            mitigation_mask = [0] * int(minimum_mitigation_rate) + [1] * int(
                self.num_discrete_action_levels - minimum_mitigation_rate
            )

            # tariff masks
            tariff_mask = []
            for other_region_id in range(self.num_regions):
                other_region_mitigation_rate = int(self.minimum_mitigation_rate_all_regions[other_region_id])

                # if other region is self or self not in bilateral negotiation
                if (other_region_id == region_id):
                    # make no change to tariff policy
                    regional_tariff_mask = [1] * self.num_discrete_action_levels

                # if other region's mitigation rate less than yours
                elif other_region_mitigation_rate < minimum_mitigation_rate:
                    # tariff the 1-mitigation rate
                    region_tariff_rate = self.num_discrete_action_levels - other_region_mitigation_rate
                    regional_tariff_mask = [0] * region_tariff_rate + [1] * other_region_mitigation_rate

                # if other regions mitigation >= your own bonus
                else:
                    # set tarrif cap
                    region_tariff_rate = self.num_discrete_action_levels - other_region_mitigation_rate
                    regional_tariff_mask = [1] * region_tariff_rate + [0] * other_region_mitigation_rate

                tariff_mask.extend(regional_tariff_mask)

            action_mask_dict[region_id]["tariff"] = tariff_mask
            action_mask_dict[region_id]["mitigation"] = mitigation_mask

        return action_mask_dict


class BilateralNegotiator(BaseProtocol):

    """
    Basic Bi-lateral negotiation included as the original
    example

    Attributes:
          rice: instance of RICE-N model
          stages (list(dict)): function steps that this negotiator class performs,
                               each contains a function and number of actions
                               used by the function
          num_negotiation_stages (int): number of stages in total

    """

    def __init__(self, num_regions, num_discrete_action_levels):
        self.stages = [
            {
                "function": self.proposal_step,
                "action_space": ([num_discrete_action_levels] * 2 * num_regions),
            },
            {
                "function": self.evaluation_step,
                "action_space": [2] * num_regions,
            },
        ]
        super().__init__(num_regions, num_discrete_action_levels)

    def reset(self):
        self.promised_mitigation_rate = np.zeros((self.num_regions, self.num_regions))
        self.requested_mitigation_rate = np.zeros((self.num_regions, self.num_regions))
        self.proposal_decisions = np.zeros((self.num_regions, self.num_regions))
        self.minimum_mitigation_rate_all_regions = np.zeros(self.num_regions)

    def get_protocol_state(self):
        protocol_state = {
            "stage": np.array([self.stage_idx]) / self.num_stages,
            "promised_mitigation_rate": self.promised_mitigation_rate
            / self.num_discrete_action_levels,
            "requested_mitigation_rate": self.requested_mitigation_rate
            / self.num_discrete_action_levels,
            "proposal_decisions": self.proposal_decisions,
            "received_promised_mitigation_rate": self.promised_mitigation_rate.T
            / self.num_discrete_action_levels,
            "received_requested_mitigation_rate": self.requested_mitigation_rate.T
            / self.num_discrete_action_levels,
            "received_proposal_decisions": self.proposal_decisions.T,
            "minimum_mitigation_rate_all_regions": self.minimum_mitigation_rate_all_regions
            / self.num_discrete_action_levels,
        }
        return protocol_state

    def get_pub_priv_features(self):
        public_features = ["stage", "minimum_mitigation_rate_all_regions"]
        private_features = [
            # "promised_mitigation_rate",
            # "requested_mitigation_rate",
            # "proposal_decisions",
            "received_promised_mitigation_rate",
            "received_requested_mitigation_rate",
            "received_proposal_decisions",
        ]
        return public_features, private_features

    def proposal_step(self, actions: dict) -> dict:
        """
        Update Proposal States and Observations using proposal actions
        Update Stage to 1 - Evaluation
        """
        assert self.stage_idx == 0

        self.promised_mitigation_rate = np.array(
            [actions[region_id][0::2] for region_id in range(self.num_regions)]
        )

        self.requested_mitigation_rate = np.array(
            [actions[region_id][1::2] for region_id in range(self.num_regions)]
        )

    def evaluation_step(self, actions):
        """
        Update minimum mitigation rates
        """
        assert self.stage_idx == 1

        self.proposal_decisions = np.array(
            [actions[region_id] for region_id in range(self.num_regions)]
        )

        # Force set the evaluation for own proposal to reject
        for region_id in range(self.num_regions):
            self.proposal_decisions[region_id, region_id] = 0

        # outgoing accepted mitigation rates
        oamr = self.promised_mitigation_rate * self.proposal_decisions.T
        # incoming accepted mitigation rates
        iamr = self.requested_mitigation_rate.T * self.proposal_decisions
        # minimum mitigation rate all regions
        mmrar = np.concatenate((oamr, iamr), axis=1).max(axis=1)

        self.minimum_mitigation_rate_all_regions = mmrar

    def get_partial_action_mask(self):
        """
        Generate action masks.
        """
        action_mask_dict = defaultdict(dict)
        for region_id in range(self.num_regions):
            minimum_mitigation_rate = self.minimum_mitigation_rate_all_regions[
                region_id
            ]
            mitigation_mask = [0] * int(minimum_mitigation_rate) + [1] * int(
                self.num_discrete_action_levels - minimum_mitigation_rate
            )
            action_mask_dict[region_id]["mitigation"] = mitigation_mask

        return action_mask_dict


class TorchLinaerMasking(TorchModelV2, nn.Module):
    """Model that handles simple discrete action masking.

    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        fc_dims,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, gym_dict)
            and "action_mask" in orig_space.spaces
            and "features" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.num_outputs = num_outputs

        prev_layer_size = orig_space["features"].shape[0]
        layers = []

        for fc_dim in fc_dims:
            layers.append(nn.Linear(prev_layer_size, fc_dim))
            layers.append(nn.ReLU())
            prev_layer_size = fc_dim


        self._hidden_layers = nn.Sequential(*layers)

        self._logits = nn.Linear(fc_dims[-1], self.num_outputs)

        self._value_branch = nn.Linear(fc_dims[-1], 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        self._features = self._hidden_layers(input_dict["obs"]["features"])
        logits = self._logits(self._features)

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=-3.4e38)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state
    
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)

ModelCatalog.register_custom_model("torch_linear_masking", TorchLinaerMasking)
