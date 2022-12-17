import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from PIL import Image


def get_homography(pts1, pts2):
    assert len(pts1) == len(pts2), "`pts1` and `pts2` must have equal length."

    # construct A matrix
    A = np.zeros([len(pts1) * 2, 9])
    r = 0
    for i in range(len(pts1)):
        A[r, :] = [
            pts1[i][0],
            pts1[i][1],
            1,
            0,
            0,
            0,
            -pts2[i][0] * pts1[i][0],
            -pts2[i][0] * pts1[i][1],
            -pts2[i][0],
        ]
        A[r + 1, :] = [
            0,
            0,
            0,
            pts1[i][0],
            pts1[i][1],
            1,
            -pts2[i][1] * pts1[i][0],
            -pts2[i][1] * pts1[i][1],
            -pts2[i][1],
        ]
        r = r + 2

    # find smallest eigen-vector of A.T @ A
    u, s, vh = np.linalg.svd(A)

    # reconstruct t
    t = np.reshape(vh[-1], (3, 3))
    t = t / t[-1][-1]

    return t


def estimate_transformation_ransac(
    kps1, kps2, matches, transform_func, n_samples, n_trials, inlier_thresh
):
    # change kps1 and kps2 to numpy array
    kp1 = np.array([kps1[match.queryIdx].pt for match in matches])
    kp2 = np.array([kps2[match.trainIdx].pt for match in matches])

    # add one column for homogeneous transformation
    kp1_ht = np.append(kp1, np.ones((kp1.shape[0], 1)), axis=1)

    inlier_max = 0

    for nt in range(n_trials):

        # sample matches
        sp_matches = random.sample(range(len(matches)), n_samples)
        sp_kp1 = [kp1[sp_match, :] for sp_match in sp_matches]
        sp_kp2 = [kp2[sp_match, :] for sp_match in sp_matches]

        # find transformation
        sp_transform = transform_func(sp_kp1, sp_kp2)

        # find inlier and outlier
        kp2_tf = kp1_ht @ sp_transform.T
        kp2_tf = kp2_tf[:, :2] / kp2_tf[:, 2:]
        error = np.linalg.norm(kp2 - kp2_tf, axis=1)
        sp_mask = error < inlier_thresh

        # update best
        if inlier_max < sum(sp_mask):
            inlier_max = sum(sp_mask)
            transform = sp_transform
            mask = sp_mask

    return transform, mask


path = "needle_images_new/"
output_path = "needle_masks_new/"

bb_dict = {
    "needle_images_new/ref_2022-11-18_17-29-11_left_0.png": [550, 0, 750, 800],
    "needle_images_new/ref_2022-11-18_17-29-11_right_0.png": [750, 0, 950, 800],
    "needle_images_new/ref_2022-11-21_21-14-25_left_0.png": [720, 0, 920, 850],
    "needle_images_new/ref_2022-11-21_21-14-25_right_0.png": [900, 0, 1100, 850],
    "needle_images_new/ref_2022-11-23_18-42-03_left_0.png": [700, 0, 900, 850],
    "needle_images_new/ref_2022-11-23_18-42-03_right_0.png": [900, 0, 1100, 900],
    "needle_images_new/ref_2022-11-23_19-07-54_left_0.png": [650, 0, 850, 800],
    "needle_images_new/ref_2022-11-23_19-07-54_right_0.png": [850, 0, 1100, 900],
}


file_list = glob.glob(os.path.join(path, "*.png"))
# file_list = glob.glob('needle_images_new/2022-11-23_19-07-54_right_???.png')
# file_list = ["needle_images_new/2022-11-21_21-14-25_left_113.png"]

for i, filename_full in enumerate(file_list):

    filename = filename_full.split("/")[1]
    if filename[0] == "r":
        continue

    ref_filename_full = path + "ref_"
    for tmp in filename.split("_")[:-1]:
        ref_filename_full = ref_filename_full + tmp + "_"
    ref_filename_full = ref_filename_full + "0.png"

    # read image
    img = cv2.imread(filename_full)
    ref = cv2.imread(ref_filename_full)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp1 = orb.detect(ref, None)
    kp2 = orb.detect(img, None)

    # compute the descriptors with ORB
    kps1, des1 = orb.compute(ref, kp1)
    kps2, des2 = orb.compute(img, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    match_img = cv2.drawMatches(
        ref,
        kps1,
        img,
        kps2,
        matches[:50],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 255, 0),
    )

    transform, mask = estimate_transformation_ransac(
        kps1,
        kps2,
        matches,
        transform_func=get_homography,
        n_samples=5,
        n_trials=100,
        inlier_thresh=5,
    )

    warped_ref = cv2.warpPerspective(ref, transform, (img.shape[1], img.shape[0]))

    # image differce
    img_diff = np.abs(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        - cv2.cvtColor(warped_ref, cv2.COLOR_BGR2GRAY)
    )

    # bounding box
    bb = np.array(bb_dict[ref_filename_full])
    bb = np.concatenate((bb.reshape((2, 2)).T, [[1, 1]]))
    bb_ = transform @ bb
    bb_ = np.round(bb_[0:2, :] / bb_[2, :]).astype(int)
    bb_[bb_ < 0] = 0

    mask = np.zeros(img_diff.shape)
    mask[bb_[1, 0] : bb_[1, 1], bb_[0, 0] : bb_[0, 1]] = img_diff[
        bb_[1, 0] : bb_[1, 1], bb_[0, 0] : bb_[0, 1]
    ]

    # threshold on imgae diff
    mask[np.bitwise_or(mask > 220, mask < 50)] = 0
    mask[mask != 0] = 1

    # cleanup image
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 3)), iterations=4)

    # find upper bound
    mask_sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=5)
    if 0 == (np.max(mask_sobely) - np.min(mask_sobely)):
        bb = np.array(bb_dict[ref_filename_full])
        bb = np.concatenate((bb.reshape((2, 2)).T, [[1, 1]]))
        bb_ = transform @ bb
        bb_ = np.round(bb_[0:2, :] / bb_[2, :]).astype(int)

        mask = np.zeros(img_diff.shape)
        mask[bb_[1, 0] : bb_[1, 1], bb_[0, 0] : bb_[0, 1]] = img_diff[
            bb_[1, 0] : bb_[1, 1], bb_[0, 0] : bb_[0, 1]
        ]
        print(bb_)
        plt.imshow(img_diff)
        plt.show()
        break

    else:
        mask_sobely = (mask_sobely - np.min(mask_sobely)) / (
            np.max(mask_sobely) - np.min(mask_sobely)
        )
        mask_sobely[mask_sobely < 0.2] = 1
        mask_sobely[mask_sobely != 1] = 0
        mask_sobely = cv2.morphologyEx(
            mask_sobely, cv2.MORPH_OPEN, np.ones((2, 6)), iterations=1
        )
        bb_upper = max(np.where(mask_sobely)[0])
        mask[:bb_upper, :] = 0

        # plot
        img_needle = np.zeros(img.shape, dtype=int)
        img_needle[:, :, 2] = mask * 255
        overlayed_img = 0.3 * img + 0.7 * img_needle
        overlayed_img = overlayed_img.astype(np.uint8)

        # save img
        # overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
        # overlayed_img = cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2GRAY)

        mask = mask.astype(bool)

        mask_filename_full = output_path + filename.split(".")[0] + "_mask.png"
        im = Image.fromarray(mask)
        im.save(mask_filename_full)

        cv2.imwrite("needle_masks_new_ol/" + filename, overlayed_img)

    print(f"save image #{i}  {mask_filename_full}")
