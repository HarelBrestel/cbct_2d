import os

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import glob

# load the DICOM files

def ct_to_sagital(path,id):
    files = []
    folder = os.listdir(path + '/' + id)
    #files_name_list = glob.glob(path + '/' + id + '/' + folder[0] + '/' + '*.dcm', recursive=False)

    files_name_list = glob.glob(path + '/' + id +  '/' + '*.dcm', recursive=False)

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

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.InstanceNumber)
    slices.pop()


    # pixel aspects, assuming all slices are the same
    ps = 0
    ss = 0
    for i in slices:
        if hasattr(i, "PixelSpacing"):
            if i.PixelSpacing != [0,0]:
                ps = i.PixelSpacing
                break
    for i in slices:
        if hasattr(i, "SliceThickness"):
            if i.SliceThickness != 0:
                ss = i.SliceThickness
                break
    print(id)
    print(ps, ss, "\n")
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]
# create 3D array
    img_shape = 0
    for i in slices:
        if len(i.pixel_array.shape) == 2:
            img_shape = list(i.pixel_array.shape)
            break

    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

# fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        if np.ndim(img2d) == 2:
            img3d[:, :, i] = img2d

    w_center = 0
    w_width = 0
    for i in slices:
        if i[0x0028, 0x1050].value != 128:
            w_center = i[0x0028, 0x1050].value
            break

    for i in slices:
        if i[0x0028, 0x1051].value != 255:
            w_width = i[0x0028, 0x1051].value
            break

    img3d_n = dicom_grayscale_normaliztaion(img3d, w_center, w_width)

    #sagital = (np.rint(img3d_n[:, img_shape[1]//2, :])).astype(int)
    #plt.imshow(sagital, cmap='gray', vmin=0, vmax=255)
    #plt.axis('off')
    #plt.savefig('C:/Users/User/PycharmProjects/pythonProject1/pictures/' + id + '.jpg', transparent=True, bbox_inches='tight', pad_inches=0, dpi = 300)
    try:
        os.makedirs('C:/Users/User/PycharmProjects/pythonProject1/pictures/' + id)
    except:
        pass

    sagital = (np.rint(img3d_n[:, : , :])).astype(int)
    for i in range(img_shape[1]):
        plt.imshow(sagital[:, i, :], cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
        plt.savefig('C:/Users/User/PycharmProjects/pythonProject1/pictures/' + id + '/' + str(i) + '.jpg', transparent=True, bbox_inches='tight', pad_inches=0, dpi = 300)


def dicom_grayscale_normaliztaion(img3d,w_center,w_width):

   mask_minimum = img3d > (w_center - w_width/2)
   mask_max = img3d < (w_center + w_width/2)
   max_values = (~mask_max) * 256
   nimg = (w_width - (w_center + w_width / 2 - img3d)) * 256/w_width

   nimg = nimg * mask_minimum
   nimg = nimg * mask_max
   nimg = nimg +max_values

   return nimg

def main():
    path_local = 'C:/Users/User/PycharmProjects/pythonProject1/ct files'


    #for i in os.listdir(path_local):
    #    ct_to_sagital(path_local,i)

    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '13527')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '18486')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '20963')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '21347')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '22467')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '22596')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '22637')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '24065')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '24734')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '26048')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '26070')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '27934')
    ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '28271')



if __name__ == "__main__":
    main()