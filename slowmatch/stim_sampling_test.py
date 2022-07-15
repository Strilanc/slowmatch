import stim
import pytest
from slowmatch.stim_sampling import count_logical_errors


@pytest.mark.parametrize(
    "d,noise,num_shots",
    [(3, 0.1, 100), (5, 0.2, 100)]
)
def test_stim_repetition_code(d, noise, num_shots):
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=d * 3,
        distance=d,
        before_round_data_depolarization=noise)
    count_logical_errors(circuit, num_shots)

