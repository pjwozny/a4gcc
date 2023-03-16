import numpy as np
import pytest

from negotiator import (
    NoProtocol,
    BaseProtocol,
    BasicClub,
    BasicClubDiscreteDefect,
    BilateralNegotiator,
    DirectSanction,
    DirectProportionalSanction,
)

PROTOCOLS = [
    NoProtocol,
    BilateralNegotiator,
    BasicClub,
    BasicClubDiscreteDefect,
    DirectSanction,
    DirectProportionalSanction,
]


@pytest.mark.parametrize("protocol_class", PROTOCOLS)
def test_smoke_protocol(protocol_class: BaseProtocol):
    protocol_class(27, 10)


@pytest.mark.parametrize("protocol_class", PROTOCOLS)
def test_protocol_reset_state(protocol_class: BaseProtocol):
    protocol: BaseProtocol = protocol_class(27, 10)
    protocol.reset()


@pytest.mark.parametrize("protocol_class", PROTOCOLS)
def test_protocol_state(protocol_class: BaseProtocol):
    protocol: BaseProtocol = protocol_class(27, 10)
    protocol.reset()
    protocol_state = protocol.get_protocol_state()
    assert isinstance(protocol_state, dict)

    for key, value in protocol_state.items():
        assert isinstance(key, str)
        assert isinstance(value, np.ndarray)


@pytest.mark.parametrize("protocol_class", PROTOCOLS)
def test_protocol_steps(protocol_class: BaseProtocol):
    protocol: BaseProtocol = protocol_class(27, 10)
    protocol.reset()

    action_length = sum(
        [num for stage in protocol.stages for num in stage["action_space"]]
    )
    action = {i: np.zeros(action_length) for i in range(protocol.num_regions)}

    for _ in range(protocol.num_stages):
        protocol.check_do_step({}, action)

    done, rice_actions = protocol.check_do_step({}, action)
    assert done
    assert isinstance(rice_actions, dict)
