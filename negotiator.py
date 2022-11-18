import numpy as np

class Negotiator():
  
  def __init__(self, rice):
    self.rice = rice
    self.stages = [
      {
        "function":self.proposal_step,
        "numberActions":(
                [self.rice.num_discrete_action_levels] \
                      * 2 * self.rice.num_regions
            )
      },
      {
        "function":self.evaluation_step,
        "numberActions":[2] * self.rice.num_regions
      } 
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
        num_proposal_actions = len(self.stages[0]['numberActions'])

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
        num_evaluation_actions = len(self.stages[1]['numberActions'])

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

        self.rice.set_global_state("proposal_decisions", proposal_decisions, self.rice.timestep)

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

  def generate_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.rice.num_regions)}
        for region_id in range(self.rice.num_regions):
            mask = self.rice.default_agent_action_mask.copy()
            if self.rice.negotiation_on:
                minimum_mitigation_rate = int(round(
                    self.rice.global_state["minimum_mitigation_rate_all_regions"]["value"][
                        self.rice.timestep, region_id
                    ]
                    * self.rice.num_discrete_action_levels
                ))
                mitigation_mask = np.array(
                    [0 for _ in range(minimum_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.rice.num_discrete_action_levels - minimum_mitigation_rate
                        )
                    ]
                )
                mask_start = sum(self.rice.savings_action_nvec)
                mask_end = mask_start + sum(self.rice.mitigation_rate_action_nvec)
                mask[mask_start:mask_end] = mitigation_mask
            mask_dict[region_id] = mask

        return mask_dict