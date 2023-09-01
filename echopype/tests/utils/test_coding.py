import pytest
import numpy as np
import xarray as xr
import math
import dask

from echopype.utils.coding import _get_auto_chunk, set_netcdf_encodings

@pytest.mark.parametrize(
    "chunk",
    ["auto", "5MB", "10MB", "30MB", "70MB", "100MB", "default"],
)
def test__get_auto_chunk(chunk):
    random_data = 15 + 8 * np.random.randn(10, 1000, 1000)

    da = xr.DataArray(
        data=random_data,
        dims=["x", "y", "z"]
    )
    
    if chunk == "auto":
        dask_data = da.chunk('auto').data
    elif chunk == "default":
        dask_data = da.chunk(_get_auto_chunk(da)).data
    else:
        dask_data = da.chunk(_get_auto_chunk(da, chunk)).data
    
    chunk_byte_size = math.prod(dask_data.chunksize + (dask_data.itemsize,))
    
    if chunk in ["auto", "100MB", "default"]:
        assert chunk_byte_size == dask_data.nbytes, "Default chunk is not equal to data array size!"
    else:
        assert chunk_byte_size <= dask.utils.parse_bytes(chunk), "Calculated chunk exceeded max chunk!"
        
def test_set_netcdf_encodings():
    # create a test dataset
    ds = xr.Dataset(
        {
            "var1": xr.DataArray(np.random.rand(10), dims="dim1"),
            "var2": xr.DataArray(np.random.rand(10), dims="dim1", attrs={"attr1": "value1"}),
            "var3": xr.DataArray(["a", "b", "c"], dims="dim2"),
        },
        attrs={"global_attr": "global_value"},
    )

    # test with default compression settings
    encoding = set_netcdf_encodings(ds, {})
    assert isinstance(encoding, dict)
    assert len(encoding) == 3
    assert "var1" in encoding
    assert "var2" in encoding
    assert "var3" in encoding
    assert encoding["var1"]["zlib"] is True
    assert encoding["var1"]["complevel"] == 4
    assert encoding["var2"]["zlib"] is True
    assert encoding["var2"]["complevel"] == 4
    assert encoding["var3"]["zlib"] is False

    # test with custom compression settings
    compression_settings = {"zlib": True, "complevel": 5}
    encoding = set_netcdf_encodings(ds, compression_settings)
    assert isinstance(encoding, dict)
    assert len(encoding) == 3
    assert "var1" in encoding
    assert "var2" in encoding
    assert "var3" in encoding
    assert encoding["var1"]["zlib"] is True
    assert encoding["var1"]["complevel"] == 5
    assert encoding["var2"]["zlib"] is True
    assert encoding["var2"]["complevel"] == 5
    assert encoding["var3"]["zlib"] is False
