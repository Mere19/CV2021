import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm


def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """

    # initialize output
    vPoints = np.zeros((nPointsX * nPointsY, 2))  # numpy array, [nPointsX*nPointsY, 2]

    # TODO
    # compute the grid
    h, w = img.shape
    x = np.linspace(border, h - border - 1, nPointsX)
    y = np.linspace(border, w - border - 1, nPointsY)
    xv, yv = np.meshgrid(x, y)

    # flatten the grid
    for i, j in np.ndindex(xv.shape):
        vPoints[i * nPointsY + j] = np.array([xv[i, j], yv[i, j]])

    return vPoints

def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)

    descriptors = []  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):
        # TODO
        # initialize descriptor for the grid point
        descriptor = []

        # compute the upper right corners of the cells around the grid point
        x = np.linspace(vPoints[i][0] - w * 2, vPoints[i][0] + w * 1, 4)
        y = np.linspace(vPoints[i][1] - h * 2, vPoints[i][1] + h * 1, 4)
        xv, yv = np.meshgrid(x, y)

        # for each cell, compute the dscriptor
        for i, j in np.ndindex(xv.shape):
            pixel_grad_x = []
            pixel_grad_y = []
            # for each pixel in the cell, get the x gradient and y gradient
            for offset_x in range(4):
                for offset_y in range(4):
                    p_x = int(xv[i, j] + offset_x)
                    p_y = int(yv[i, j] + offset_y)
                    pixel_grad_x.append(grad_x[p_x, p_y])
                    pixel_grad_y.append(grad_y[p_x, p_y])
            
            # compute the directions of the gradients
            directions = np.arctan2(pixel_grad_y, pixel_grad_x)

            # aggregate the directions of the gradients of the pixels
            # to obtain the histogram descriptor of the grid point
            hist = np.histogram(directions, bins=nBins)
            descriptor.append(hist[0])
        
        # concatenate descriptors of all cells
        descriptors.append(np.concatenate((descriptor)))

    descriptors = np.asarray(descriptors) # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)

    return descriptors


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []  # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    # extract features for all image
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # TODO
        # Collect local feature points for each image, and compute a descriptor for each local feature point
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        img_features = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        vFeatures.append(img_features)

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])  # [n_imgs*n_vPoints, 128]
    print('number of extracted features: ', len(vFeatures))


    # Cluster the features using K-Means
    print('clustering ...')
    # print(numiter)
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]

    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    M = vFeatures.shape[0]
    N = vCenters.shape[0]

    histo = np.zeros(vCenters.shape[0])

    # TODO
    for i in range(M):
        # feature to be assigned
        f = vFeatures[i]
        # index of the nearest center
        idx = 0
        # distance to the nearest center
        min_dist = np.Inf

        for j in range(N):
            # center to be compared
            c = vCenters[j]
            # compute and update idx and min_dist
            dist = np.linalg.norm(f - c)
            if dist < min_dist:
                min_dist = dist
                idx = j

        histo[idx] += 1

    return histo


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []
    for i in tqdm(range(nImgs)):
        # print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # TODO
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        vFeatures = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        bow_hist = bow_histogram(vFeatures, vCenters)
        vBoW.append(bow_hist)

    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW


def bow_recognition_nearest(histogram,vBoWPos,vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    DistPos, DistNeg = np.Inf, np.Inf

    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor
    # TODO
    nPos = vBoWPos.shape[0]
    nNeg = vBoWNeg.shape[0]

    test_img = histogram

    for i in range(nPos):
        train_img = vBoWPos[i]
        dist = np.linalg.norm(test_img - train_img)
        if dist < DistPos:
            DistPos = dist

    for i in range(nNeg):
        train_img = vBoWNeg[i]
        dist = np.linalg.norm(test_img - train_img)
        if dist < DistNeg:
            DistNeg = dist

    if (DistPos < DistNeg):
        sLabel = 1
    else:
        sLabel = 0
    return sLabel


if __name__ == '__main__':
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'


    k = 7  # TODO
    numiter = 14  # TODO

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)
