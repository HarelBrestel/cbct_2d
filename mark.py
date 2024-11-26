import math

import cv2
import os
import numpy as np
import scipy
import win32com.client
import matplotlib.pyplot as plt
import re
import nibabel as nib
from skimage.transform import resize
from main import dicom_files_to_list, slices_to_3d, dicom_grayscale_normaliztaion, jaw_lines


def remove_white_borders(image):

    if np.all(image == [255, 255, 255]):
        return image

    # Find the first non-white column from the left
    left = 0
    while np.all(image[:, left] > 240):
        left += 1

    # Find the first non-white column from the right
    right = image.shape[1] - 1
    while np.all(image[:, right] > 240):
        right -= 1

    # Find the first non-white row from the up
    up = 0
    while np.all(image[up, :] > 240):
        up += 1

    # Find the first non-white row from the down
    down = image.shape[0] - 1
    while np.all(image[down, :] > 240):
        down -= 1
    return up, down , left, right

    # Crop the image based on the found columns
    #cropped_image = image[up:down+1, left:right+1]
    #return cropped_image

def generateMask2(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Définir la plage de couleur rouge en RGB
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255, 100, 100])

    # Seuillage de l'image RGB pour obtenir uniquement les couleurs rouges
    mask_red = cv2.inRange(rgb, lower_red, upper_red)

    # Trouver les contours dans le masque rouge
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer un masque blanc de la même taille que le masque original
    filled_mask = np.zeros_like(mask_red)

    # Remplir les contours avec la couleur blanche
    thikness_image = cv2.drawContours(filled_mask, contours, -1, (255), thickness=4)
    contours2, _ = cv2.findContours(thikness_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_image = cv2.drawContours(thikness_image, contours2, -1, (255), thickness=cv2.FILLED)
    result_filled = cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB)


    return result_filled

# Test de la fonction
# image = cv2.imread('/content/drive/MyDrive/projet gmar/tests/newImage.jpg')
#result_filled = generateMask2(image)


def apply_filter_and_save(input_folder, output_folder):
    # Ensure that the output folder exists, otherwise create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    images= [None] * (len(files) - 1)
    scope = np.zeros(4)
    for filename in files:
        input_path = input_folder + '/' + filename
        file_number = re.findall(r'\d+', filename)
        if os.path.exists(input_path):
            image = cv2.imread(input_path)
            #image = cv2.imread(input_folder + '/' + 'copy.jpg')
            #cropped_image = remove_white_borders(image)

            if np.all(image == [255, 255, 255]) != True:
                if np.all(scope == 0):
                    scope = remove_white_borders(image)
                cropped_image = image[scope[0]:scope[1], scope[2]:scope[3]]
                filtered_image = generateMask2(cropped_image)
                output_path = os.path.join(output_folder, filename)
                #img3d[int(file_number[0]) - 1,:,:] = filtered_image[:,:,0]
                images[int(file_number[0]) - 2] = filtered_image[:,:,0]
                # Save the filtered image
                cv2.imwrite(output_path, filtered_image)

                print(f"Image processed and saved as {output_path}")
        else:
            print(f"Image not found.")
    mask3d = np.zeros((len(images), images[0].shape[0] , images[0].shape[1]))
    for i in range(len(images)):
        mask3d[i] = images[i]
    mask3d = np.transpose(mask3d, (0, 2, 1))

    del cropped_image
    del image
    del images

    slices = dicom_files_to_list('C:/Users/User/PycharmProjects/pythonProject1/ct files', input_folder.split('/')[-1])
    img3d, scale = slices_to_3d(slices)

    img3d = dicom_grayscale_normaliztaion(img3d, slices)

    del slices
    resized_mask = resize(mask3d, img3d.shape, anti_aliasing=True)
    resized_mask = resized_mask.astype(bool)
    resized_mask = np.transpose(resized_mask, (1, 0, 2))
    resized_mask = resized_mask[:,:,::-1]
    resized_mask = resized_mask * 256
    del mask3d
    try:
        os.makedirs('C:/Users/User/PycharmProjects/pythonProject1/output/' + input_folder.split('/')[-1])
    except:
        pass

    os.chdir('C:/Users/User/PycharmProjects/pythonProject1/output/' + input_folder.split('/')[-1])

    for i in range(img3d.shape[1]):
        gray_image = img3d[:, i, :]
        red_image = resized_mask[:, i, :] != 0
        rgb_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
        box_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

        for k in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1]):
                if red_image[k, j] == 1:
                    # Set the pixel to red when the mask is 1
                    rgb_image[k, j] = [0, 0, 255]  # BGR format (Red)
                else:
                    # Set the pixel to grayscale when the mask is 0
                    gray_value = gray_image[k, j]
                    rgb_image[k, j] = [gray_value, gray_value, gray_value]
        box_mask = (red_image * 255).astype("uint8")
        contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  # Get the bounding box coordinates
            cv2.rectangle(box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle

        for k in range(box_image.shape[0]):
            for j in range(box_image.shape[1]):
                if box_image[k, j, 1] == 0:
                    gray_value = gray_image[k, j]
                    box_image[k, j] = [gray_value, gray_value, gray_value]

        cv2.imwrite(str(i) + "box.jpg", np.rot90(box_image))
        cv2.imwrite(str(i) + "regular.jpg", np.rot90(gray_image))
        cv2.imwrite(str(i) + "mask.jpg", np.rot90(255 * red_image))
        cv2.imwrite(str(i) + "merge.jpg", np.rot90(rgb_image))

        rgb_gray = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
        rgb_red = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
        rgb_gray[:,:,0] = gray_image
        rgb_gray[:,:,1] = gray_image
        rgb_gray[:,:,2] = gray_image
        rgb_red[:,:,0] = red_image * 255
        rgb_red[:,:,1] = red_image * 255
        rgb_red[:,:,2] = red_image * 255
        a = np.hstack((box_image,rgb_gray))
        b = np.hstack((rgb_red,rgb_image))
        cv2.imwrite("all" + str(i) + '.jpg'  , np.rot90(np.vstack((a,b))))

    ax_image = img3d[:, :, 3 * img3d.shape[2] // 10]
    s = 0
    hist = np.histogram(ax_image, bins=256, range=(0, 256))
    for i in range(256):
        s += hist[0][255 - i]
        if s > hist[0][0] * 0.1:
            s = 255 - i
            break
    ax_mask = ax_image > s
    ax_image = ax_mask * ax_image
    folder =  input_folder.split('/')[-1]
    cv2.imwrite(folder + '_ax_3.jpg', img3d[:, :, 3 * img3d.shape[2] // 10])
    cv2.imwrite(folder + '_ax_mask.jpg', ax_image)
    r1, r2, l1, l2 = jaw_lines(folder + '_ax_mask')
    m1 = (r2[1] - r1[1]) / (r2[0] - r1[0])
    m2 = (l2[1] - l1[1]) / (l2[0] - l1[0])
    theta1 = math.degrees(math.atan(m1))
    theta2 = math.degrees(math.atan(m2))


    lines = cv2.imread(folder + '_ax_maskonly_lines.jpg')
    line_rotated_right = np.rint(scipy.ndimage.rotate(lines, angle=-(90 - theta1), axes=(1, 0))).astype(int)
    line_rotated_left = np.rint(scipy.ndimage.rotate(lines, angle=90 + theta2, axes=(1, 0))).astype(int)
    cv2.imwrite(folder + 'lines_roteted_right.jpg', line_rotated_right)
    cv2.imwrite(folder + 'lines_roteted_left.jpg', line_rotated_left)

    ax_mask_rotated_right = np.rint(scipy.ndimage.rotate(ax_image, angle=-(90 - theta1), axes=(1, 0))).astype(int)
    ax_mask_rotated_left = np.rint(scipy.ndimage.rotate(ax_image, angle=90 + theta2, axes=(1, 0))).astype(int)
    cv2.imwrite(folder + 'ax_roteted_right.jpg', ax_mask_rotated_right)
    cv2.imwrite(folder + 'ax_roteted_left.jpg', ax_mask_rotated_left)


    array_rotated_right = np.rint(scipy.ndimage.rotate(img3d, angle=- (90 - theta1), axes=(1, 0))).astype(int)
    mask_rotated_right = np.rint(scipy.ndimage.rotate(resized_mask, angle=- (90 - theta1), axes=(1, 0))).astype(int)
    mask_rotated_right = mask_rotated_right > 30
    mask_rotated_right = mask_rotated_right * 255

    right = 0

    for i in range(line_rotated_right.shape[0]):
        if line_rotated_right[line_rotated_right.shape[0] // 2][-i][2] != 0:
            right = line_rotated_right.shape[1] - i
            break

    min_right = right
    for i in range(right):
        k = 0
        for j in range(ax_mask_rotated_right.shape[0] // 3):
            if ax_mask_rotated_right[ax_mask_rotated_right.shape[0] // 3 + j][right - i] != 0 and k == 0:
                k = 1
        if k == 1:
            min_right -= 1
        else:
            break

    max_right = right
    for i in range(ax_mask_rotated_right.shape[0] - right):
        k = 0
        for j in range(ax_mask_rotated_right.shape[0] // 3):
            if ax_mask_rotated_right[ax_mask_rotated_right.shape[0] // 3 + j][right + i] != 0:
                k = 1
        if k == 1:
            max_right += 1
        else:
            break

    for i in range(array_rotated_right.shape[1]):
        if i >= (min_right - 10) and i <= (max_right + 10):
            gray_image = array_rotated_right[:, i, :]
            red_image = mask_rotated_right[:,i,:] != 0
            rgb_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
            box_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

            for k in range(gray_image.shape[0]):
                for j in range(gray_image.shape[1]):
                    if red_image[k, j] == 1:
                        # Set the pixel to red when the mask is 1
                        rgb_image[k, j] = [0, 0, 255]  # BGR format (Red)
                    else:
                        # Set the pixel to grayscale when the mask is 0
                        gray_value = gray_image[k, j]
                        rgb_image[k, j] = [gray_value, gray_value, gray_value]

            box_mask = (red_image * 255).astype("uint8")
            contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)  # Get the bounding box coordinates
                cv2.rectangle(box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle

            for k in range(box_image.shape[0]):
                for j in range(box_image.shape[1]):
                    if box_image[k, j, 1] == 0:
                        gray_value = gray_image[k, j]
                        box_image[k, j] = [gray_value, gray_value, gray_value]

            cv2.imwrite(str(i) + "box.jpg", np.rot90(box_image))
            cv2.imwrite("r" + str(i) + "normal.jpg", np.rot90(array_rotated_right[:, i, :]))
            cv2.imwrite("r" + str(i) + "mask.jpg", np.rot90(red_image * 255))
            cv2.imwrite("r" + str(i) + "merge.jpg", np.rot90(rgb_image))

            rgb_gray = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
            rgb_red = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
            rgb_gray[:, :, 0] = gray_image
            rgb_gray[:, :, 1] = gray_image
            rgb_gray[:, :, 2] = gray_image
            rgb_red[:, :, 0] = red_image * 255
            rgb_red[:, :, 1] = red_image * 255
            rgb_red[:, :, 2] = red_image * 255
            a = np.hstack((box_image, rgb_gray))
            b = np.hstack((rgb_red, rgb_image))
            cv2.imwrite("Rall" + str(i) + '.jpg', np.rot90(np.vstack((a, b))))
    return 0

    array_rotated_left = np.rint(scipy.ndimage.rotate(img3d, angle=90 + theta2, axes=(1, 0))).astype(int)
    return 0
    left = 0

    for i in range(line_rotated_left.shape[0]):
        if line_rotated_left[line_rotated_left.shape[0] // 2][i][2] != 0:
            left = i
            break

    min_left = left
    for i in range(left):
        k = 0
        for j in range(ax_mask_rotated_left.shape[0] // 3):
            if ax_mask_rotated_left[ax_mask_rotated_left.shape[0] // 3 + j][left - i] != 0 and k == 0:
                k = 1
        if k == 1:
            min_left -= 1
        else:
            break

    max_left = left
    for i in range(ax_mask_rotated_left.shape[0] - left):
        k = 0
        for j in range(ax_mask_rotated_left.shape[0] // 3):
            if ax_mask_rotated_left[ax_mask_rotated_left.shape[0] // 3 + j][left + i] != 0:
                k = 1
        if k == 1:
            max_left += 1
        else:
            break

    for i in range(array_rotated_left.shape[1]):
        if i >= (min_left - 10) and i <= (max_left + 10):
            cv2.imwrite("l" + str(i) + ".jpg", np.rot90(array_rotated_left[:, i, :]))
    print(min_right, ",  ", right, ',  ', max_right)


    #resized_array = resize(mask3d, (240, 155, 240), anti_aliasing=True)
    # nifti_img = nib.Nifti1Image(resized_array, affine=np.eye(4))
    # nifti_img.to_filename('output.nii')

    a=5
input_folder = 'C:/Users/User/PycharmProjects/pythonProject1/marking/ppt/12019'
output_folder = 'C:/Users/User/PycharmProjects/pythonProject1/marking/output/'


# Apply the filter and save the filtered images
apply_filter_and_save(input_folder, output_folder)



