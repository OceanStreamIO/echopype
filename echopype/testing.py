import os
from pathlib import Path

# HERE = Path(__file__).parent.absolute()
# TEST_DATA_FOLDER = HERE / "test_data"
current_directory = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_FOLDER = os.path.join(current_directory, "..", "test_data")
TEST_DATA_FOLDER = Path(TEST_DATA_FOLDER)
