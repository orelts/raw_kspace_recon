import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import twixtools
from enum import Enum

class DimensionType(Enum):
    """
    Enum to specify the type of dimensionality for data processing.
    
    Attributes:
        TWO_D: 2D static data.
        THREE_D: 3D static data.
        TWO_D_DYNAMIC: 2D dynamic data.
        THREE_D_DYNAMIC: 3D dynamic data.
    """
    TWO_D = "2D"
    THREE_D = "3D"
    TWO_D_DYNAMIC = "2D_DYNAMIC"
    THREE_D_DYNAMIC = "3D_DYNAMIC"
    
    def __str__(self):
        return self.value
    
def ifftnd(kspace, axes=[-1]):
    """
    Perform an n-dimensional inverse Fast Fourier Transform (FFT) on the given k-space data.
    
    Parameters:
        kspace (ndarray): The input k-space data.
        axes (list of int): The axes over which to compute the inverse FFT.
        
    Returns:
        img (ndarray): The resulting image data in the spatial domain.
    """
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = ifftshift(ifftn(fftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img

def fftnd(img, axes=[-1]):
    """
    Perform an n-dimensional Fast Fourier Transform (FFT) on the given image data.
    
    Parameters:
        img (ndarray): The input image data.
        axes (list of int): The axes over which to compute the FFT.
        
    Returns:
        kspace (ndarray): The resulting k-space data.
    """
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace

def rms_comb(sig, axis=1):
    """
    Compute the root mean square (RMS) combination of the signal along the specified axis.
    
    Parameters:
        sig (ndarray): The input signal.
        axis (int): The axis along which to compute the RMS combination.
        
    Returns:
        rms (ndarray): The resulting RMS combined signal.
    """
    return np.sqrt(np.sum(abs(sig)**2, axis))

def find_dat_files(folder):
    """
    Find all .dat files in the specified folder and its subdirectories.
    
    Parameters:
        folder (str): The root folder to search.
        
    Returns:
        dat_files (list of str): A list of paths to the .dat files found.
    """
    dat_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.dat'):
                dat_files.append(os.path.join(root, file))
    return dat_files

def read_twix_2d_kspace(dat_file):
    """
    Read a 2D k-space data from a .dat file using TwixTools.
    
    Parameters:
        dat_file (str): The path to the .dat file.
        
    Returns:
        data (ndarray): The k-space data.
        image_dim (list of int): The indices of the 2D image data in the data array.
        coil_index (int): The index of the coil in the data array.
    """
    twix = twixtools.read_twix(dat_file, keep_syncdata_and_acqend=True)
    mapped = twixtools.map_twix(twix)
    im_data = mapped[-1]['image']

    print("Dimensions: " + str(im_data.non_singleton_dims))

    line_index = im_data.non_singleton_dims.index("Lin")
    col_index = im_data.non_singleton_dims.index("Col")
    coil_index = im_data.non_singleton_dims.index("Cha")

    image_dim = [line_index, col_index]

    im_data.flags['remove_os'] = True

    data = im_data[:].squeeze()
    
    return data, image_dim, coil_index

def coil_combine_kspace(data, image_dim=[2,3], channel_dim=1):
    """
    Combine k-space data from multiple coils using RMS combination.
    
    Parameters:
        data (ndarray): The k-space data.
        image_dim (list of int): The dimensions of the image data.
        channel_dim (int): The dimension corresponding to the coils.
        
    Returns:
        image (ndarray): The combined image data.
    """
    image = ifftnd(data, image_dim)
    image = rms_comb(image, axis=channel_dim)
    return image

def read_raw_twix_data(dat_file, type=DimensionType.TWO_D):
    """
    Read a .dat (raw twix kspace) given the expected dimensionality and coil combine the kspace into an image.

    Parameters:
        dat_file (str): The path to the .dat file.
        type (DimensionType): The type of dimensionality to process (2D, 3D, etc.).
        
    Returns:
        image (ndarray): The resulting image data in the spatial domain.
        data (ndarray): The k-space data.
    """
    if type == DimensionType.TWO_D:
        data, image_dim, channel_dim = read_twix_2d_kspace(dat_file)
        image = coil_combine_kspace(data, image_dim=image_dim, channel_dim=channel_dim)
        assert image.ndim == 3, "Image should be a 3D array with shape (slices, x, y)"

    else:
        # TODO: implement the other types
        raise NotImplementedError(f"{type} is not yet implemented.")


    return image, data

def plot_slider_view_2d(images):
    # Initialize figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)

    # Display initial slice
    current_slice = 0
    current_contrast = 1.0
    # Apply initial vmin and vmax based on images data
    vmin = np.min(images)
    vmax = np.max(images)
    img = ax.imshow(images[current_slice, :, :], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f"Slice {current_slice}")

    # Add slider
    ax_slider = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_slice = Slider(ax_slider, 'Slice', 0, images.shape[0]-1, valinit=current_slice, valstep=1)

    # Update function for slider
    def update(val):
        slice_index = int(slider_slice.val)
        img.set_data(abs(images[slice_index, :, :]))
        ax.set_title(f"Slice {slice_index}")
        fig.canvas.draw_idle()

    slider_slice.on_changed(update)

    # Add slider for contrast adjustment
    ax_contrast = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_contrast = Slider(ax_contrast, 'Contrast', 0, 1.0, valinit=current_contrast, valstep=0.1)

    # Update function for contrast slider
    def update_contrast(val):
        global current_contrast
        current_contrast = slider_contrast.val
        slice_index = int(slider_slice.val)
        img.set_clim(vmin, current_contrast* vmax)
        img.set_data(abs(images[slice_index, :, :]))
        fig.canvas.draw_idle()

    slider_contrast.on_changed(update_contrast)


    plt.show()