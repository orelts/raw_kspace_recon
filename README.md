# MRI Data Reconstruction Using Twixtools for Raw K-Space .dat Files

## Credits

This repository utilizes the following tools for Twix file handling:
- [Twixtools](https://github.com/pehses/twixtools): For reading Twix files.
- [mapVBVD](https://github.com/pehses/mapVBVD): For mapping and processing VBVD data.

## Overview

This repository provides a workflow for reading raw k-space data from Twix files and performing basic reconstruction. It supports both MATLAB and Python formats.

## Dependencies

- `numpy`
- `matplotlib`
- `twixtools` (for reading Twix files; see [Twixtools GitHub page](https://github.com/pehses/twixtools) for installation instructions)
- Other standard libraries (e.g., `os`, `argparse`)

## Installation

Ensure you have the necessary Python packages installed. You can install them using pip:

```bash
pip install numpy matplotlib
```

For `twixtools`, follow the installation instructions on the [Twixtools GitHub page](https://github.com/pehses/twixtools). Typically, this involves cloning the repository and installing it with `pip`:

```bash
git clone https://github.com/pehses/twixtools.git
cd twixtools
pip install .
```

## Usage

### Script Usage

The repository includes a script called workflow.py for processing .dat files. This script reads raw k-space data from Twix .dat files and performs reconstruction based on the specified dimensionality type.

Run the script from the command line with the following command:

```
python workflow.py path/to/your/file.dat --type 2D
```

Replace path/to/your/file.dat with the actual path to your .dat file and adjust --type to the dimensionality type. The available options are:

- `type` (DimensionType): The type of dimensionality. Options include:
  - `DimensionType.TWO_D` (default): For 2D data.
  - `DimensionType.THREE_D`: For 3D data.
  - `DimensionType.TWO_D_DYNAMIC`: For 2D dynamic data.
  - `DimensionType.THREE_D_DYNAMIC`: For 3D dynamic data.
  
### Function Overview

#### `read_raw_twix_data(dat_file, type)`

Reads and processes data from a Twix `.dat` file according to the specified dimensionality type.

**Parameters:**
- `dat_file` (str): Path to the `.dat` file.
- `type` (DimensionType): The type of dimensionality. Options include:
  - `DimensionType.TWO_D` (default): For 2D data.
  - `DimensionType.THREE_D`: For 3D data.
  - `DimensionType.TWO_D_DYNAMIC`: For 2D dynamic data.
  - `DimensionType.THREE_D_DYNAMIC`: For 3D dynamic data.

**Returns:**
- `image` (ndarray): The resulting image data in the spatial domain.
- `data` (ndarray): The k-space data.

### Example Usage

Here's how you can use `read_raw_twix_data` in a Jupyter notebook:

```python
# Import necessary functions
from utils import read_raw_twix_data, DimensionType
import matplotlib.pyplot as plt

# Define the path to your .dat file
dat_file_path = 'path/to/your/file.dat'

# Read and process 2D data
image_2d, data_2d = read_raw_twix_data(dat_file_path, type=DimensionType.TWO_D)

# Display the resulting image
plt.imshow(image_2d[0, :, :], cmap='gray')
plt.title('Reconstructed 2D Image')
plt.colorbar()
plt.show()
```

**Note:** Replace `'path/to/your/file.dat'` with the actual path to your `.dat` file. The example assumes you are working with 2D and 3D data; adjust the code as needed for other dimensionality types.

### MATLAB Example

For MATLAB users, refer to the workflow.m file included in this repository for examples on how to use mapVBVD to read and process Twix files.


## Contact

For any questions or issues, please contact [Orel Tsioni](oreltsioni@gmail.com).

