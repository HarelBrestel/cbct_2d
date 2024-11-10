import cv2
import os
import numpy as np
import win32com.client
import matplotlib.pyplot as plt
import re
import nibabel as nib
from skimage.transform import resize
from main import dicom_files_to_list,slices_to_3d,dicom_grayscale_normaliztaion



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
    thikness_image = cv2.drawContours(filled_mask, contours, -1, (255), thickness=7)
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

    slices = dicom_files_to_list('C:/Users/User/PycharmProjects/pythonProject1/ct files', input_folder.split('/')[-1])
    img3d = slices_to_3d(slices)

    img3d = dicom_grayscale_normaliztaion(img3d, slices)

    resized_mask = resize(mask3d, img3d.shape, anti_aliasing=True)
    #resized_array = resize(mask3d, (240, 155, 240), anti_aliasing=True)
    # nifti_img = nib.Nifti1Image(resized_array, affine=np.eye(4))
    # nifti_img.to_filename('output.nii')

    a=5
input_folder = 'C:/Users/User/PycharmProjects/pythonProject1/marking/ppt/12019'
output_folder = 'C:/Users/User/PycharmProjects/pythonProject1/marking/output/'


# Apply the filter and save the filtered images
apply_filter_and_save(input_folder, output_folder)



