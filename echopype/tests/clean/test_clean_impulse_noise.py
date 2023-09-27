import pytest
import numpy as np
import echopype.clean


# Note: We've removed all the setup and utility functions since they are now in conftest.py
RYAN_PARAMS = {"thr": 10, "m": 5, "n": 1}
RYAN_ITERABLE_PARAMS = {"thr": 10, "m": 5, "n": (1, 2)}
WANG_PARAMS = {"thr": (-70, -40), "erode": [(3, 3)], "dilate": [(5, 5), (7, 7)], "median": [(7, 7)]}
DESIRED_CHANNEL = "GPT 120 kHz 00907203422d 1 ES120-7"
DESIRED_FREQUENCY = 120000


@pytest.mark.parametrize(
    "method,parameters,desired_channel,desired_frequency,expected_true_false_counts",
    [
        ("ryan", RYAN_PARAMS, DESIRED_CHANNEL, None, (2130885, 32419)),
        ("ryan_iterable", RYAN_ITERABLE_PARAMS, DESIRED_CHANNEL, None, (2124976, 38328)),
        ("wang", WANG_PARAMS, None, DESIRED_FREQUENCY, (635732, 1527572)),
    ],
)
def test_get_impulse_noise_mask(
    sv_dataset_jr230,  # Use the specific fixture for the JR230 file
    method,
    parameters,
    desired_channel,
    desired_frequency,
    expected_true_false_counts,
):
    source_Sv = sv_dataset_jr230
    mask = echopype.clean.get_impulse_noise_mask(
        source_Sv,
        parameters=parameters,
        desired_channel=desired_channel,
        desired_frequency=desired_frequency,
        method=method,
    )
    count_true = np.count_nonzero(mask)
    count_false = mask.size - count_true
    true_false_counts = (count_true, count_false)

    assert true_false_counts == expected_true_false_counts
