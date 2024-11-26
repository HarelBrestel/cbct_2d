import os
import math
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import scipy
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from skimage.measure import block_reduce


def dicom_files_to_list(path,id):
    files = []
    folder = os.listdir(path + '/' + id)
    if any(os.path.isdir(os.path.join(path, id, item)) for item in folder):
        files_name_list = glob.glob(path + '/' + id + '/' + folder[0] + '/' + '*.dcm', recursive=False)
    else:
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
    slices_by_value = {}
    for item in slices:
        value = item[0x0020, 0x0011].value
        if value not in slices_by_value:
            slices_by_value[value] = []
        slices_by_value[value].append(item)
    r_slices = max(slices_by_value.values(), key=len)
    return r_slices

def slices_to_3d(slices):
    # pixel aspects, assuming all slices are the same
    ps = 0
    ss = 0
    for i in slices:
        if hasattr(i, "PixelSpacing"):
            if i.PixelSpacing != [0, 0]:
                ps = i.PixelSpacing
                break
    for i in slices:
        if hasattr(i, "SliceThickness"):
            if i.SliceThickness != 0:
                ss = i.SliceThickness
                break
    print(id)
    print(ps, ss, "\n")
    #ax_aspect = ps[1]/ps[0]
    #sag_aspect = ps[1]/ss
    #cor_aspect = ss/ps[0]
    scale = ss/ps[0]
    # create 3D array
    img_shape = 0

    for i in slices:
        if len(i.pixel_array.shape) == 2:
            img_shape = list(i.pixel_array.shape)
            break

    img_shape.append(len(slices))
    img3d = np.zeros(img_shape,dtype=np.float32)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img2d = img2d + s.RescaleIntercept
        if s.RescaleIntercept != 0:  ### if need normalization
            print("RescaleIntercept")
            img3d[:, :, i] = img2d - s.RescaleIntercept
        elif np.ndim(img2d) == 2:
            img3d[:, :, i] = img2d

    return img3d ,scale

    new_z = int(np.round(scale * img3d.shape[2]))  # New length for the z dimension, rounded to nearest integer
    original_indices = np.linspace(0, img3d.shape[2] - 1, img3d.shape[2])
    new_indices = np.linspace(0, img3d.shape[2] - 1, new_z)
    stretched_array = np.empty((img3d.shape[0], img3d.shape[1], new_z))
    for i in range(img3d.shape[0]):
        for j in range(img3d.shape[1]):
            # Use scipy's interp1d for linear interpolation along the third axis (z-axis)
            interpolator = interp1d(original_indices, img3d[i, j, :], kind='linear', fill_value='extrapolate')
            stretched_array[i, j, :] = interpolator(new_indices)
    return stretched_array

def ct_to_sagital(path,id):

    slices = dicom_files_to_list(path,id)
    img3d, scale = slices_to_3d(slices)
    #img3d = block_reduce(img3d, block_size=(2, 2, 2), func=np.mean)
    img3d = dicom_grayscale_normaliztaion(img3d, slices)


    ax_image = img3d[:, :, 3 * img3d.shape[2] // 10]
    s = 0
    hist = np.histogram(ax_image,bins=256,range=(0,256))
    for i in range(256):
        s += hist[0][255 - i]
        if s > hist[0][0]*0.1:
            s = 255 - i
            break
    ax_mask = ax_image > s
    ax_image = ax_mask * ax_image

    try:
        os.makedirs('C:/Users/User/PycharmProjects/pythonProject1/output/' + id + 'new')
    except:
        pass

    os.chdir('C:/Users/User/PycharmProjects/pythonProject1/output/' + id + 'new')

    for i in range(img3d.shape[1]):
        cv2.imwrite(str(i) + ".jpg", np.rot90(zoom(img3d[:, i, :],(1,scale),order=3))) #sagittal

        #cv2.imwrite(str(i) + ".jpg", np.rot90(img3d[:, i, :])) #sagittal
        #cv2.imwrite(str(i) + ".jpg", np.rot90(img3d[:, :, i])) #axial

    return 0

    cv2.imwrite(id + '_ax_3.jpg', img3d[:, :, 3 * img_shape[2] // 10])
    cv2.imwrite(id + '_ax_mask.jpg', ax_image)
    
    #jaw_lines(id+'_ax_3')
    r1, r2, l1, l2 = jaw_lines(id + '_ax_mask')
    m1 = (r2[1] - r1[1]) / (r2[0] - r1[0])
    m2 = (l2[1] - l1[1]) / (l2[0] - l1[0])
    theta1 = math.degrees(math.atan(m1))
    theta2 = math.degrees(math.atan(m2))
    #return 0
    array_rotated_right = np.rint(scipy.ndimage.rotate(img3d, angle=- (90 - theta1), axes=(1, 0))).astype(int)
    array_rotated_left = np.rint(scipy.ndimage.rotate(img3d, angle=90 + theta2, axes=(1, 0))).astype(int)

    lines = cv2.imread(id + '_ax_maskonly_lines.jpg')
    line_rotated_right = np.rint(scipy.ndimage.rotate(lines, angle=-(90 - theta1), axes=(1, 0))).astype(int)
    line_rotated_left = np.rint(scipy.ndimage.rotate(lines, angle=90 + theta2, axes=(1, 0))).astype(int)
    cv2.imwrite(id + 'lines_roteted_right.jpg',line_rotated_right)
    cv2.imwrite(id + 'lines_roteted_left.jpg', line_rotated_left)

    ax_mask_rotated_right = np.rint(scipy.ndimage.rotate(ax_image, angle=-(90 - theta1), axes=(1, 0))).astype(int)
    ax_mask_rotated_left = np.rint(scipy.ndimage.rotate(ax_image, angle=90 + theta2, axes=(1, 0))).astype(int)
    cv2.imwrite(id + 'ax_roteted_right.jpg',ax_mask_rotated_right)
    cv2.imwrite(id + 'ax_roteted_left.jpg', ax_mask_rotated_left)

    right = 0

    for i in range(line_rotated_right.shape[0]):
        if line_rotated_right[line_rotated_right.shape[0]//2][-i][2] != 0:
            right = line_rotated_right.shape[1] - i
            break

    min_right = right
    for i in range(right):
        k = 0
        for j in range(ax_mask_rotated_right.shape[0]//3):
            if ax_mask_rotated_right[ax_mask_rotated_right.shape[0]//3 + j][right - i] != 0 and k == 0:
                k = 1
        if k == 1:
            min_right -= 1
        else:
            break

    max_right = right
    for i in range(ax_mask_rotated_right.shape[0] - right):
        k = 0
        for j in range(ax_mask_rotated_right.shape[0]//3):
            if ax_mask_rotated_right[ax_mask_rotated_right.shape[0]//3 + j][right + i] != 0:
                k=1
        if k == 1:
            max_right += 1
        else:
            break

    for i in range(array_rotated_right.shape[1]):
        if i >= (min_right - 10) and i <= (max_right + 10):
            cv2.imwrite("r" + str(i) + ".jpg", np.rot90(array_rotated_right[:, i, :]))

    left = 0

    for i in range(line_rotated_left.shape[0]):
        if line_rotated_left[line_rotated_left.shape[0]//2][i][2] != 0:
            left = i
            break

    min_left = left
    for i in range(left):
        k = 0
        for j in range(ax_mask_rotated_left.shape[0]//3):
            if ax_mask_rotated_left[ax_mask_rotated_left.shape[0]//3 + j][left - i] != 0 and k == 0:
                k = 1
        if k == 1:
            min_left -= 1
        else:
            break

    max_left = left
    for i in range(ax_mask_rotated_left.shape[0] - left):
        k = 0
        for j in range(ax_mask_rotated_left.shape[0]//3):
            if ax_mask_rotated_left[ax_mask_rotated_left.shape[0]//3 + j][left + i] != 0:
                k=1
        if k == 1:
            max_left += 1
        else:
            break

    for i in range(array_rotated_left.shape[1]):
        if i >= (min_left - 10) and i <= (max_left + 10):
            cv2.imwrite("l" + str(i) + ".jpg", np.rot90(array_rotated_left[:, i, :]))
    print(min_right, ",  ", right, ',  ',max_right)

def dicom_grayscale_normaliztaion(img3d,slices):
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
    del slices
    mask_minimum = img3d > (w_center - w_width/2)
    mask_max = img3d < (w_center + w_width/2)

    max_values = (~mask_max) * 256
    #nimg = (w_width - (w_center + w_width / 2 - img3d)) * 256/w_width

    nimg = w_width

    nimg -= w_center + w_width / 2 - img3d

    nimg *= 256/w_width

    nimg = nimg * mask_minimum
    del mask_minimum
    nimg = nimg * mask_max
    del mask_max

    nimg = nimg +max_values
    del max_values


    return nimg

def jaw_lines(image_name):
    cv_im_ax = cv2.imread(image_name + '.jpg')

    gray = cv2.cvtColor(cv_im_ax, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv_im_ax, 50, 150, apertureSize=3)
    #cv2.imshow('edges', edges)
    i = 150
    lines = []
    while (i > 1):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, i)
        if lines is not None and lines.size > 20:
                break
        i -= 10


    d_arr = []
    xy_0 = []
    xy1 = []
    x_center = cv_im_ax.shape[0] / 2
    y_center = cv_im_ax.shape[1] / 2
    min_right = 1000
    i_mr = 0
    i_ml = 0
    min_left = 1000
    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        #cv2.line(cv_im_ax, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #plt.imshow(cv_im_ax)
        #plt.show()
        d = abs(
            (x_center * (- a / b) - y_center + r / b)
            /
            math.sqrt((- a / b) ** 2 + 1)
        )
        if theta > 1.578 and d < min_right:  # right
            min_right = d
            i_mr = (x1, y1), (x2, y2)
        elif theta < 1.578 and d < min_left:  # left
            min_left = d
            i_ml = (x1, y1), (x2, y2)

        # d_arr.append(d)
        # xy_0.append((x1, y1))
        # xy1.append((x2, y2))

    cv2.line(cv_im_ax, i_mr[0], i_mr[1], (0, 0, 255), 2)
    cv2.line(cv_im_ax, i_ml[0], i_ml[1], (0, 0, 255), 2)

    os.chdir('C:/Users/User/PycharmProjects/pythonProject1/output/' + image_name[:image_name.index('_')]  )

    cv2.imwrite(image_name +'lines.jpg', cv_im_ax)

    im_lines = np.zeros(cv_im_ax.shape)

    cv2.line(im_lines, i_mr[0], i_mr[1], (0, 0, 255), 2)
    cv2.line(im_lines, i_ml[0], i_ml[1], (0, 0, 255), 2)
    cv2.imwrite(image_name +'only_lines.jpg', im_lines)

    return i_mr[0], i_mr[1] , i_ml[0], i_ml[1]


def main():
    path_local = 'C:/Users/User/PycharmProjects/pythonProject1/ct files'
    os.chdir('C:/Users/User/PycharmProjects/pythonProject1/ax_images/')
    # ct_to_sagital(path_local,'10204')
    # ct_to_sagital(path_local,'10623')
    # ct_to_sagital(path_local, '11078')
    # ct_to_sagital(path_local, '11281')
    # ct_to_sagital(path_local, '11936')
    # ct_to_sagital(path_local, '12019')
    # ct_to_sagital(path_local, '12812')
    # ct_to_sagital(path_local, '12987')
    # ct_to_sagital(path_local, '13011')
    # ct_to_sagital(path_local, '13605')
    # ct_to_sagital(path_local, '14320')
    # ct_to_sagital(path_local, '14682')
    # ct_to_sagital(path_local, '14813')
    # ct_to_sagital(path_local, '15270')
    # ct_to_sagital(path_local, '15349')
    # ct_to_sagital(path_local, '15959')
    # ct_to_sagital(path_local, '16028')
    # ct_to_sagital(path_local, '1637')
    # ct_to_sagital(path_local, '16437')
    # ct_to_sagital(path_local, '16884')
    # ct_to_sagital(path_local, '16916')
    # ct_to_sagital(path_local, '17436')
    ## ct_to_sagital(path_local, '1744')
    # ct_to_sagital(path_local, '17442')
    # ct_to_sagital(path_local, '18308')
    # ct_to_sagital(path_local, '18716')
    # ct_to_sagital(path_local, '19242')
    # ct_to_sagital(path_local, '19890')
    # ct_to_sagital(path_local, '20016')
    # ct_to_sagital(path_local, '20669')
    # ct_to_sagital(path_local, '20687')
    # ct_to_sagital(path_local, '21848')
    # ct_to_sagital(path_local, '22962')
    # ct_to_sagital(path_local, '2459')
    # ct_to_sagital(path_local, '2663')
    # ct_to_sagital(path_local, '2767')
    # ct_to_sagital(path_local, '29005') ###
    # ct_to_sagital(path_local, '30964')
    ##ct_to_sagital(path_local, '3162')
    # ct_to_sagital(path_local, '32671')
    # ct_to_sagital(path_local, '4233')
    # ct_to_sagital(path_local, '476')
    # ct_to_sagital(path_local, '5124')
    # ct_to_sagital(path_local, '6277')
    # ct_to_sagital(path_local, '703')
    # ct_to_sagital(path_local, '7087')
    # ct_to_sagital(path_local, '8008')
    # ct_to_sagital(path_local, '8713')
    # ct_to_sagital(path_local, '92')
    # ct_to_sagital(path_local, '9673')
    #ct_to_sagital(path_local, '11281')


    #for i in os.listdir(path_local):
     #   if i[0:2] > '10':
      #      ct_to_sagital(path_local, i)
       #     break

    ##ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '13527')
    ##ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '18486')
    #ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '20963')
    ##ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '21347')
    ##ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '22467')
    #ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '22596')
    #ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '22637')
    #ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '24065')
    #ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '24734')
    ##ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '26048')
    ##ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '26070')
    #ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '27934')
    ##ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', '28271')

    # ct_to_sagital(path_local, 'R1')
    # ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', 'HMO-19465')
    # ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', 'HMO-20379')
    # ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', 'HMO-1989')
    # ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', 'HMO-3132')
    # ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', 'HMO-3798')
    # ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/drive', 'HMO-18286')
    #ct_to_sagital('C:/Users/User/PycharmProjects/pythonProject1/ct files', 'HMO - 31181')


    # ct_to_sagital(path_local, '2959') 10032
    # ct_to_sagital(path_local, '7096')
    # ct_to_sagital(path_local, '8233')

    #ct_to_sagital(path_local, '3889')
    #ct_to_sagital(path_local, '5685')
    #ct_to_sagital(path_local, '8771')
    #ct_to_sagital(path_local, '10283')
    #ct_to_sagital(path_local, '11465')
    #ct_to_sagital(path_local, '15188')
    #ct_to_sagital(path_local, '16233')
    #ct_to_sagital(path_local, '17216')
    #ct_to_sagital(path_local, '19182')
    #ct_to_sagital(path_local, '20397')
    #ct_to_sagital(path_local, '32699')
    #ct_to_sagital(path_local, '828')
    #ct_to_sagital(path_local, '3098')

    ct_to_sagital(path_local, 'AR001')

    #ct_to_sagital(path_local, '629796118369')
    #ct_to_sagital(path_local, '2723242683249A')
    #ct_to_sagital(path_local, '2723242683249B')

    #ct_to_sagital(path_local, '1763455062322A')
    #ct_to_sagital(path_local, '1763455062322B')


    #ct_to_sagital(path_local, '4023688728466')


if __name__ == "__main__":
    main()
