from skimage.io import imread_collection
from skimage.color import rgb2gray
from skimage.filters import threshold_li
from skimage.filters import median
from skimage.morphology import remove_small_objects
from skimage.morphology import binary_erosion
from skimage.morphology import disk
from skimage.measure import label
from skimage.measure import find_contours
from skimage.measure import approximate_polygon


from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from sklearn.cluster import AffinityPropagation

import numpy as np
import os
from PIL import Image, ImageDraw

import math

from sys import argv


def _main(*args):
    pathname = args[0]
    if pathname[-1] != '/':
        pathname += '/'
    filenames, full_filenames = get_filenames(pathname)
    X = get_features(filenames, full_filenames)
    labels, n_clusters, similar_matrix = find_clusters(X)
    print_persons_images(filenames, labels, n_clusters)
    print_similar_images(filenames, similar_matrix)


def get_features(filenames, full_filenames):
    X = []

    imgs = imread_collection(full_filenames)
    for k, img in enumerate(imgs):
        temp = rgb2gray(img)
        val = threshold_li(temp)
        mask = temp < val
        temp[mask] = 0
        temp[~mask] = 1

        temp = binary_erosion(temp, disk(1))
        temp = median(temp, selem=disk(2))
        temp = ndi.binary_fill_holes(temp)
        labeled_image, num_labels = label(temp, return_num=True, connectivity=1)
        temp = remove_small_objects(labeled_image, 20000)

        contours = find_contours(temp, 0.1)
        approx = approximate_polygon(contours[0], tolerance=50)

        if approx[0].tolist() == approx[-1].tolist():
            approx = approx[:-1]

        convex = ConvexHull(approx)
        number_points = len(convex.points)
        inner_points_indexes = list(set(range(number_points)).difference(set(convex.vertices)))
        if len(inner_points_indexes) > 4:
            inner_points_indexes = find_waste_points(convex.points, inner_points_indexes)

        points_indexes = find_points_order(convex.points, inner_points_indexes)
        elem = convex.points[points_indexes]
        feature_vec = []
        for i in range(len(elem) - 1):
            dist = np.sqrt(sum((elem[i + 1] - elem[i]) ** 2))
            feature_vec.append(dist)
        if len(elem) < 9:
            for i in range(9 - len(elem)):
                feature_vec.append(0)
        means = find_masked_means(img, temp)
        feature_vec.append(means[0])
        feature_vec.append(means[1])
        feature_vec.append(means[2])
        X.append(feature_vec)

        im = Image.open(full_filenames[k])
        draw = ImageDraw.Draw(im)
        for i in range(len(elem) - 1):
            draw.line((elem[i + 1][1], elem[i + 1][0], elem[i][1], elem[i][0]), fill=(0, 128, 0), width=3)
        del draw

        os.makedirs('out', exist_ok=True)
        out_filename = 'out/out_' + filenames[k]
        im.save(out_filename)
    return X


def find_waste_points(points, indexes):
    indexes = find_waste_points_helper(points, indexes)
    temp_points = points[indexes]
    sums = []
    for i in indexes:
        temp_sum = sum(list(map(lambda x: sum((x - points[i])**2)**0.5, temp_points)))
        sums.append((i, temp_sum))
    sums.sort(key=sort_by_sum)
    sums = sums[:4]
    new_indexes = []
    for i in sums:
        new_indexes.append(i[0])
    return sorted(new_indexes)


def sort_by_sum(input_pair):
    return input_pair[1]


def find_waste_points_helper(points, indexes):
    new_indexes = []
    max_ind = len(points)
    for index in indexes:
        rad_andle = angle(points[index-1]-points[index], points[(index+1)%max_ind]-points[index])
        if (rad_andle > 0.13) and (rad_andle < 2.10):
            new_indexes.append(index)
    return new_indexes


def find_masked_means(image, mask):
    nrow, ncol, _ = image.shape
    counter = 0
    means = [0,0,0]
    for i in range(nrow):
        for j in range(ncol):
            if mask[i][j]:
                means[0] += image[i][j][0]
                means[1] += image[i][j][1]
                means[2] += image[i][j][2]
                counter +=1
    for i in range(3):
        means[i] /= counter
    return means


def find_points_order(points, indexes):
    max_ind = len(points)
    pred_points = list()
    next_points = list()
    index_map = dict()
    for ind in indexes:
        pred_points.append(points[ind - 1].tolist())
        next_points.append(points[(ind + 1) % max_ind].tolist())
        index_map[ind - 1] = ind
        index_map[ind] = (ind + 1) % max_ind
    for elem in pred_points:
        if elem not in next_points:
            first_point = elem
            break

    temp_points = points.tolist()
    index = temp_points.index(first_point)
    new_order = [index]

    try:
        for i in range(len(index_map)):
            index = index_map[index]
            new_order.append(index)
    except KeyError:
        print()
    return new_order


def find_clusters(X):
    af = AffinityPropagation().fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    return af.labels_, n_clusters_, af.affinity_matrix_


def print_persons_images(images_filenames, labels, n_clusters):
    with open('persons_images.txt', 'w+') as clusters_file:
        cluster_dict = dict()
        for i in range(n_clusters):
            cluster_dict[i] = list()
        for i, label in enumerate(labels):
            cluster_dict[label].append(images_filenames[i])
        for i in range(n_clusters):
            clusters_file.write('Персона №'+ str(i+1)+ ' --')
            for image_filename in cluster_dict[i]:
                clusters_file.write(' ' + image_filename)
            clusters_file.write('\n')


def print_similar_images(images_filenames, similar_matrix):
    with open('similar_images.txt', 'w+') as similar_file:
        for i, line in enumerate(similar_matrix):
            line = line.tolist()
            mins = sorted(line, reverse=True)[1:4]
            indexes = []
            for val in mins:
                indexes.append(line.index(val))
            similar_file.write(images_filenames[i] + ': ')
            for j in indexes:
                similar_file.write(images_filenames[j] + ' ')
            similar_file.write('\n')


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def get_filenames(pathname):
    filenames = sorted(list(filter(lambda x: x[-4:] == '.tif', os.listdir(path=pathname))))
    full_filenames = list(map(lambda x: 'training/' + x, filenames))
    return filenames, full_filenames


if __name__ == "__main__":
    _main(argv[1])

