from bart import bart
import matplotlib.pyplot as plt
import numpy as np
import os
import twixtools
import sigpy
from matplotlib.widgets import Slider

def ifftnd(kspace, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = ifftshift(ifftn(fftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img

def fftnd(img, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace

def rms_comb(sig, axis=1):
    return np.sqrt(np.sum(abs(sig)**2, axis))

def find_dat_files(folder):
    dat_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.dat'):
                dat_files.append(os.path.join(root, file))
    return dat_files

# read all image data from file
def read_image_data(filename):
    out = list()
    for mdb in twixtools.read_twix(filename)[-1]['mdb']:
        if mdb.is_image_scan():
            out.append(mdb.data)
    return np.asarray(out)  # 3D numpy array [acquisition_counter, n_channel, n_column]


# read image data from list of mdbs and sort into 3d k-space (+ coil dim.)
def import_kspace(mdb_list):
    image_mdbs = []
    for mdb in mdb_list:
        if mdb.is_image_scan():
            image_mdbs.append(mdb)

    n_line = 1 + max([mdb.cLin for mdb in image_mdbs])
    n_part = 1 + max([mdb.cPar for mdb in image_mdbs])
    n_channel, n_column = image_mdbs[0].data.shape

    out = np.zeros([n_part, n_line, n_channel, n_column], dtype=np.complex64)
    for mdb in image_mdbs:
        # '+=' takes care of averaging, but careful in case of other counters (e.g. echoes)
        out[mdb.cPar, mdb.cLin] += mdb.data

    return out  # 4D numpy array [n_part, n_line, n_channel, n_column]


ksp_dir = "/home/orel/Downloads/kspace/"
ksp_files = find_dat_files(ksp_dir)
ksp_files

ksp_files = [f for f in ksp_files if "Adj" not in f and "Anatomy" not in f]
ksp_files

twix = twixtools.read_twix(ksp_files[3],  keep_syncdata_and_acqend=True)
# map the twix data to twix_array objects
mapped = twixtools.map_twix(twix)
im_data = mapped[-1]['image']

# make sure that we later squeeze the right dimensions:
line_index = im_data.non_singleton_dims.index("Lin")
col_index = im_data.non_singleton_dims.index("Col")
coil_index = im_data.non_singleton_dims.index("Cha")

# the twix_array object makes it easy to remove the 2x oversampling in read direction
im_data.flags['remove_os'] = True

# read the data (array-slicing is also supported)

print("data dimensions before squeeze: "  + str(im_data.shape))
data = im_data[:].squeeze()
print("data dimensions after squeeze: "  + str(data.shape))

image = ifftnd(data, [line_index, col_index])
image = rms_comb(image, axis=coil_index)

assert image.ndim == 3, "Image should be a 3D array with shape (slices, x, y)"

# Initialize figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

# Display initial slice
current_slice = 0
current_contrast = 1.0
# Apply initial vmin and vmax based on image data
vmin = np.min(image)
vmax = np.max(image)
img = ax.imshow(image[current_slice, :, :], cmap='gray', vmin=vmin, vmax=vmax)
ax.set_title(f"Slice {current_slice}")

# Add slider
ax_slider = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_slice = Slider(ax_slider, 'Slice', 0, image.shape[0]-1, valinit=current_slice, valstep=1)

# Update function for slider
def update(val):
    slice_index = int(slider_slice.val)
    img.set_data(abs(image[slice_index, :, :]))
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
    img.set_data(abs(image[slice_index, :, :]))
    fig.canvas.draw_idle()

slider_contrast.on_changed(update_contrast)


plt.show()