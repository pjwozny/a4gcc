import pytest
import numpy as np
from rice import Rice


@pytest.fixture
def rice() -> Rice:
    rice_class = Rice(protocol_name="NoProtocol")
    return rice_class


def test_smoke(rice):
    rice


def test_rice_reset(rice: Rice):
    rice.reset()


def test_rice_reset_observation(rice: Rice):
    obs = rice.reset()
    assert len(obs) == rice.num_regions
    assert "features" in obs[0]
    assert "action_mask" in obs[0]


def test_rice_step(rice: Rice):
    rice.reset()
    action = np.zeros(len(rice.actions_nvec), dtype=rice.float_dtype)
    actions = {i: action for i in range(rice.num_regions)}
    rice.step(actions)
