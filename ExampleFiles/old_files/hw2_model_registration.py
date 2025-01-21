import cv2
import numpy as np


def extract_keypoints_and_descriptors_and_shape(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    im_shape=image.shape

    return keypoints, descriptors, im_shape


def register_object(im_shape, keypoints, descriptors, video_path, output_video_path):
    sift = cv2.SIFT_create()
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)
        matches = flann.knnMatch(descriptors, descriptors_frame, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        MIN_MATCH_COUNT = 10
        inliers = 0
        outliers = 0
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = im_shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                matches_mask = mask.ravel().tolist()
                inliers = sum(matches_mask)
                outliers = len(matches_mask) - inliers

                for i, m in enumerate(good_matches):
                    pt = (int(dst_pts[i][0][0]), int(dst_pts[i][0][1]))
                    if matches_mask[i]:
                        cv2.circle(frame, pt, 5, (0, 255, 255), -1)  # Inliers желтые
                    else:
                        cv2.circle(frame, pt, 5, (0, 0, 255), -1)  # Outliers красные

        cv2.putText(frame, f'Matches: {len(good_matches)}/{len(descriptors)}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Inliers: {inliers}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Outliers: {outliers}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        output_video.write(frame)

    video_capture.release()
    output_video.release()
    print('Видео сохранено')


image_path = 'reference_messy_1.jpg'
keypoints, descriptors, im_shape = extract_keypoints_and_descriptors_and_shape(image_path)
video_path = 'video before messy.mp4'
output_video_path = 'output_video messy_1 .mp4'
register_object(im_shape, keypoints, descriptors, video_path, output_video_path)
