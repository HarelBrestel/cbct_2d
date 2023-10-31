import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

from skimage import exposure # for histogram equalization


path_local = 'C:/Users/User/PycharmProjects/pythonProject1/'
# load the DICOM files
files = []
files_name_list = glob.glob(path_local + '*.dcm', recursive=False)

for fname in files_name_list:
    files.append(pydicom.dcmread(fname))

print(f"file count: {len(files)}")


# skip files with no SliceLocation (eg scout views)
slices = []
skipcount = 0
for f in files:
    if hasattr(f, "InstanceNumber"):
        slices.append(f)
    else:
        skipcount = skipcount + 1

print(f"skipped, no SliceLocation: {skipcount}")

# ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.InstanceNumber)

# pixel aspects, assuming all slices are the same
ps = slices[0].PixelSpacing
ss = slices[0].SliceThickness
ax_aspect = ps[1]/ps[0]
sag_aspect = ps[1]/ss
cor_aspect = ss/ps[0]


# create 3D array
img_shape = list(slices[0].pixel_array.shape)
img_shape.append(len(slices))
img3d = np.zeros(img_shape)

# fill 3D array with the images from the files
for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[:, :, i] = img2d

# plot 3 orthogonal slices
a1 = plt.subplot(2, 2, 1)
plt.imshow(img3d[:, :, img_shape[2]//2])
a1.set_aspect(ax_aspect)

a2 = plt.subplot(2, 2, 2)
plt.imshow(img3d[:, img_shape[1]//2, :])
a2.set_aspect(sag_aspect)

a3 = plt.subplot(2, 2, 3)
plt.imshow(img3d[img_shape[0]//2, :, :].T)
a3.set_aspect(cor_aspect)

plt.show()




imgh = np.array(img2d)


print(imgh)

#print( imgh.shape ) # image width and height in pixels

# Use numpy.ndarray.min() and numpy.ndarray.max() to find the smallest and largest values in the array a1
#print( np.ndarray.min(imgh) )
#print( np.ndarray.max(imgh) )

# Use numpy.histogram() to calculate histogram data for the values in the ndarray a1
#hist,bins = np.histogram(a1.flatten(),256,[0,256])

#hist,bins = np.histogram(imgh, bins=256)
#print( "histogram of pixel values", hist )
#print( sum(hist) )

plt.figure() # create a new figure
plt.hist(imgh) # plot a histogram of the pixel values

plt.show()



a1_eq = exposure.equalize_hist(imgh)


#hist_eq,bins_eq = np.histogram(a1_eq, bins=256)
#print( "histogram equaliation of pixel values", hist_eq )
#print( sum( hist_eq ) )
#print( a1_eq )
#print( np.ndarray.min(a1_eq) )
#print( np.ndarray.max(a1_eq) )
#print( a1_eq.shape )


#########################################
#  Plotting the image

"""
Now, we plot the original image (unmodified) and its histogram equalization.
We will do this with the "gray" colormap and the "spectral" colormap.
"""

############
# grayscale

fig1  = plt.figure()
plt.imshow(imgh, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig1.suptitle("Original + Gray colormap", fontsize=12)

fig2 = plt.figure()
plt.imshow(a1_eq, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig2.suptitle("Histogram equalization + Gray colormap", fontsize=12)

plt.show()

imx = img3d[:, :, img_shape[2]//2]
ax_eq = exposure.equalize_hist(imx)

fig1  = plt.figure()
plt.imshow(imx, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig1.suptitle("Original + Gray colormap", fontsize=12)

fig2 = plt.figure()
plt.imshow(ax_eq, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig2.suptitle("Histogram equalization + Gray colormap", fontsize=12)

plt.show()


imy = img3d[:, img_shape[1]//2, :]
ay_eq = exposure.equalize_hist(imy)

fig1  = plt.figure()
plt.imshow(imy, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig1.suptitle("Original + Gray colormap", fontsize=12)

fig2 = plt.figure()
plt.imshow(ay_eq, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig2.suptitle("Histogram equalization + Gray colormap", fontsize=12)

plt.show()


imz = img3d[img_shape[0]//2, :, :]
az_eq = exposure.equalize_hist(imz)

fig1 = plt.figure()
plt.imshow(imz, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig1.suptitle("Original + Gray colormap", fontsize=12)

fig2 = plt.figure()
plt.imshow(az_eq, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig2.suptitle("Histogram equalization + Gray colormap", fontsize=12)

plt.show()
