import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
from skimage import exposure # for histogram equalization

from PIL import Image

path_local = 'C:/Users/User/PycharmProjects/pythonProject1/'
# load the DICOM files

def load_dicom_files_serias(path):
    files = []
    files_name_list = glob.glob(path + '*.dcm', recursive=False)

    for fname in files_name_list:
      files.append(pydicom.dcmread(fname))

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

    w_center = (slices[0])[0x0028, 0x1050].value
    w_width = (slices[0])[0x0028, 0x1051].value
    return dicom_grayscale_normaliztaion(img3d,w_center,w_width),img_shape


def plot_3_orthogonal_slices(img3d,img_shape,ax_aspect,sag_aspect,cor_aspect):
# plot 3 orthogonal slices
    #a1 = plt.subplot(2, 2, 1)
    #plt.imshow(img3d[:, :, img_shape[2]//2] , cmap = 'gray')
    #a1.set_aspect(ax_aspect)

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img3d[:, img_shape[1]//2, :] , cmap = 'gray')
    a2.set_aspect(sag_aspect)
    #plt.savefig('C:/Users/User/PycharmProjects/pythonProject1/new.jpg',)
    #plt.imsave('C:/Users/User/PycharmProjects/pythonProject1/')

    #a3 = plt.subplot(2, 2, 3)
    #plt.imshow(img3d[img_shape[0]//2, :, :].T , cmap = 'gray')
    #a3.set_aspect(cor_aspect)

    plt.show()



def histgrom_normalztion(img2d,img3d,img_shape): # in bulding
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


#Now, we plot the original image (unmodified) and its histogram equalization.
#We will do this with the "gray" colormap and the "spectral" colormap.


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

def dicom_grayscale_normaliztaion(img3d,w_center,w_width):
   #img3d[i][j][k] = (w_width-(w_width-i))*256

   mask_minimum = img3d > (w_center - w_width/2)
   mask_max = img3d < (w_center + w_width/2)
   max_values = (~mask_max) * 256
   nimg = (w_width - (w_center + w_width / 2 - img3d)) * 256/w_width

   nimg = nimg * mask_minimum
   nimg = nimg * mask_max
   nimg = nimg +max_values

   return nimg

def save_sagital(img3d,img_shape,ax_aspect,sag_aspect,cor_aspect):

    sagital = (np.rint(img3d[:, img_shape[1]//2, :])).astype(int)
    plt.imshow(sagital, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig('C:/Users/User/PycharmProjects/pythonProject1/new.jpg', transparent=True, bbox_inches='tight', pad_inches=0, dpi = 300)



    #im = Image.fromarray(A, 'L')

    #im.save("new2.png")

    #im.show()


def main():

    img3d , img_shape = load_dicom_files_serias(path_local)
    save_sagital(img3d,img_shape,1,1,1)

if __name__ == "__main__":
    main()