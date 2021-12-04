# References
# Feature Detection: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# Draw Bounding Box: https://stackoverflow.com/questions/51606215/how-to-draw-bounding-box-on-best-matches
# What are Keypoint and Descriptor: https://dsp.stackexchange.com/questions/10423/why-do-we-use-keypoint-descriptors/

import cv2 as cv
import numpy as np


def bf_orb_method():
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img_query, None)
    kp2, des2 = orb.detectAndCompute(img_scene, None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    return cv.drawMatches(img_query, kp1, img_scene, kp2, matches[:10], None,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def sift_orb_method():
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp_query, des_query = sift.detectAndCompute(img_query, None)
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_query, des_scene, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    return cv.drawMatchesKnn(img_query, kp_query, img_scene, kp_scene, good, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def sift_flann_method():
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp_query, des_query = sift.detectAndCompute(img_query, None)
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_query, des_scene, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    return cv.drawMatchesKnn(img_query, kp_query, img_scene, kp_scene, matches, None, **draw_params)


def sift_orb_bounding_box():
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp_query, des_query = sift.detectAndCompute(img_query, None)
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_query, des_scene, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # Find bounding box
    src_pts = np.float32([kp_query[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img_query.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv.perspectiveTransform(pts, M)
    dst += (w, 0)  # adding offset

    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    matchesMask=matchesMask,  # draw only inliers
    #                    )

    # cv.drawMatchesKnn expects list of lists as matches.
    result = cv.drawMatchesKnn(img_query, kp_query, img_scene, kp_scene, good, None,
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    result = cv.polylines(result, [np.int32(dst)], True, (0, 0, 255), None, cv.LINE_AA)

    return result


if __name__ == '__main__':
    img_query = cv.imread("test/query_images/道具_带框_模组数据块.png", cv.IMREAD_COLOR)
    img_scene = cv.imread("test/scene_images/screenshot_0_1x.jpg", cv.IMREAD_COLOR)

    img_matches = sift_orb_bounding_box()
    cv.imshow("matches", img_matches)
    cv.waitKey()
    # plt.imshow(img_matches), plt.show()
