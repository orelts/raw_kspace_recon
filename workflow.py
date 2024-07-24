import argparse
from utils import read_raw_twix_data, plot_slider_view_2d, DimensionType  # Replace 'your_module' with the actual module name


# Argument parser setup
parser = argparse.ArgumentParser(description="Process .dat files for MRI data.")
parser.add_argument('--dat_file_path', type=str, help="Path to the .dat file.")
parser.add_argument('--type', type=DimensionType, choices=list(DimensionType), default=DimensionType.TWO_D,
                    help="Type of dimensionality (2D, 3D, 2D_DYNAMIC, 3D_DYNAMIC).")

# Parse the arguments
args = parser.parse_args()

# Convert the string type to DimensionType Enum
dimension_type = DimensionType(args.type)

# Read and process the data
image, data = read_raw_twix_data(args.dat_file_path, type=dimension_type)

# Print or process the image and data as needed
print(f"Image shape: {image.shape}")
print(f"Data shape: {data.shape}")


if args.type is DimensionType.TWO_D:
    plot_slider_view_2d(image)
else:
    #TODO: Implement
    print(f"Plotting type {type} not yet implemented")
    pass

