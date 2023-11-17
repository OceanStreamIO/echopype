from echopype.testing import TEST_DATA_FOLDER
from echopype.mask.shoal import _weill
from echopype.mask.shoal import WEILL_DEFAULT_PARAMETERS
from echopype.mask.api import get_shoal_mask, get_shoal_mask_multichannel, apply_mask
import numpy as np
import matplotlib.pyplot as plt
from echopype.visualize import create_echogram


def test_scratch(sv_dataset_jr161, ek60_Sv):
    # source_Sv = sv_dataset_jr161
    source_Sv = ek60_Sv
    method = "will"
    # desired_channel = "GPT  38 kHz 009072033fa5 1 ES38"
    desired_channel = source_Sv.coords["channel"][1]
    print(desired_channel)
    parameters = WEILL_DEFAULT_PARAMETERS
    mask = _weill(
        source_Sv,
        desired_channel=desired_channel,
        parameters=parameters,
    )
    # mask = get_shoal_mask_multichannel(source_Sv, parameters)
    # print(mask)
    # t1 = apply_mask(source_Sv, mask)
    # t2 = create_echogram(t1)
    # plt.show()
    # print(source_Sv)
    # print(mask)
