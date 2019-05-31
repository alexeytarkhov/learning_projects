from skimage.io import imread_collection


from sys import argv

from skimage.color import rgb2gray

from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.filters import median


from skimage.segmentation import find_boundaries

from matplotlib import pyplot as plt
import numpy as np

from skimage.morphology import square
from skimage.morphology import disk
from skimage.morphology import binary_opening
from skimage.morphology import star
from skimage.morphology import remove_small_objects
from skimage.morphology import remove_small_holes
from skimage.morphology import label

from scipy import ndimage as ndi
from skimage import exposure
from skimage.measure import find_contours
from skimage.measure import regionprops
from skimage.measure import approximate_polygon


def _main(*args):
    filenames = args[0]
    print(args)

    imgs = imread_collection(filenames)
    blue_imgs = receive_blue_images(imgs)
    masks = receive_card_masks(blue_imgs)
    find_objects(imgs, masks, filenames)


def receive_blue_images(imgs):
    blue_imgs = []
    for img in imgs:
        h, w, t = img.shape
        temp = img.copy()
        temp = exposure.adjust_log(temp)
        mean = int(round(np.mean(temp)) * 1.1)
        for i in range(h):
            for j in range(w):
                if (temp[i][j][2] < temp[i][j][1]) or (temp[i][j][2] < temp[i][j][0]):
                    temp[i][j][0] = mean
                    temp[i][j][1] = mean
                    temp[i][j][2] = mean
        blue_imgs.append(temp)
    return blue_imgs


def receive_card_masks(imgs):
    masks = []
    for img in imgs:
        temp = img.copy()
        temp[:, :, 1] = 0
        temp[:, :, 2] = 0
        temp = rgb2gray(temp)
        val = threshold_otsu(temp)
        mask = temp < val
        temp[mask] = 1
        temp[~mask] = 0
        temp = median(temp)
        label_objects, nb_labels = label(temp, return_num=True)

        hols = remove_small_holes(label_objects, 5000)
        temp[hols] = 1
        temp[~hols] = 0
        temp = binary_opening(temp)
        label_objects, label_num = label(temp, connectivity=2, return_num=True)
        remove_obj = remove_small_objects(label_objects, 1100)

        label_objects, label_num = label(remove_obj, connectivity=2, return_num=True)

        # temp2 = rgb2gray(img)
        mask = label_objects > 0
        # temp2[~mask] = 0
        masks.append(mask)
    return masks


def find_objects(imgs, masks, filenames):
    for i, img in enumerate(imgs):
        temp = rgb2gray(img)

        edges = canny(temp, mask=masks[i], sigma=2)

        labeled_image, num_labels = label(edges, return_num=True)
        remove_obj = remove_small_objects(labeled_image)
        boundaries = find_boundaries(remove_obj)
        temp = ndi.binary_closing(boundaries, star(1))
        temp = ndi.binary_fill_holes(temp)
        labeled_image, num_labels = label(temp, return_num=True)
        remove_obj = remove_small_objects(labeled_image, 1000)
        remove_obj, num_labels = label(remove_obj, return_num=True)
        regions = regionprops(remove_obj)

        remove_obj = ndi.binary_erosion(remove_obj, square(2))
        remove_obj = ndi.binary_erosion(remove_obj, disk(2))

        # additional removing for case, when small object were receive after erosion
        remove_obj = remove_small_objects(remove_obj, 1000)
        remove_obj, num_labels = label(remove_obj, return_num=True)

        contours = find_contours(remove_obj, 0.9, fully_connected='high')
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(imgs[i], interpolation='nearest', cmap=plt.cm.gray)

        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

            compact_value = regions[n].area / (regions[n].perimeter) ** 2
            if compact_value > 0.07 and regions[n].solidity > 0.985:
                continue
            if compact_value > 0.06 and regions[n].solidity > 0.985 and regions[n].eccentricity > 0.8:
                continue

            if regions[n].solidity > 0.95:
                added_text = 'C'
                tolerance_value = 7
            else:
                added_text = ''
                tolerance_value = 5.5
            res = approximate_polygon(contour, tolerance=tolerance_value)
            text = 'P' + str(len(res) - 1) + added_text

            coords = regions[n].centroid
            ax.text(coords[1]-10, coords[0]-10, text, style='italic', color='red', fontsize=8,
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})

            ax.plot(res[:, 1], res[:, 0], linewidth=2)

        out_filename = filenames[i][:-4] + '_out.png'
        fig.savefig(out_filename)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()



if __name__ == "__main__":
    print(argv[1:])
    _main(argv[1:])
