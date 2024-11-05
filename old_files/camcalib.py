import numpy as np
import cv2 as cv
import glob
import pickle

# SETUP
print('setup')
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objpoints = []
imgpoints = []

images = glob.glob('data/*.jpg')
print(f'Found {len(images)} images.')

valid_images_count = 0

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        valid_images_count += 1
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

    print(f'Chessboard found in {fname}: {ret}')

cv.destroyAllWindows()
print(f'Found corners in {valid_images_count} images.')

# Calibration

print('Calibration')
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
with open('cam_params.pkl', 'wb') as f:
    pickle.dump((ret, mtx, dist, rvecs, tvecs), f)
print('Calibration done')

i = 0
for im_path in images:
    i += 1
    print('Undistortion')
    img = cv.imread(im_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print('Undistortion done')

    print('Undistort')
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite(f'calibrated/calibr_{im_path[5:13]}.jpg', dst)
    print('Undistort done')

print('Re-projection Error')
mean_error = 0
for j in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[j], rvecs[j], tvecs[j], mtx, dist)
    error = cv.norm(imgpoints[j], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("Mean error: ", mean_error/len(objpoints))
print('Re-projection Error done')


