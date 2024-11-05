import cv2
import numpy as np


def click_event(event, x, y, flags, param):
    image, points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Draw the point on the display image, not the original
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Select Points', image)
        if len(points) == 4:
            cv2.destroyAllWindows()


def crop_and_straighten_image(image_path, output_path):
    points = []
    image = cv2.imread(image_path)

    # Resize the image to fit in the window
    h, w = image.shape[:2]
    max_height = 800  # Set maximum height for display
    if h > max_height:
        scale = max_height / h
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Create a copy of the image for drawing points
    display_image = image.copy()

    cv2.imshow('Select Points', display_image)  # Show the copy for point selection
    cv2.setMouseCallback('Select Points', click_event, param=(display_image, points))
    cv2.waitKey(0)

    if len(points) == 4:
        rect = np.array(points, dtype="float32")

        widthA = np.linalg.norm(rect[0] - rect[1])
        widthB = np.linalg.norm(rect[2] - rect[3])
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[3] - rect[0])
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

        cv2.imshow('Cropped and Straightened Image', warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(output_path, warped)
    else:
        print("Insufficient points selected!")


input_path = 'messy straight.jpg'
output_path = 'reference_messy_1.jpg'
crop_and_straighten_image(input_path, output_path)

#output_path='reference messy.jpg'
#points = [(175, 285), (695, 275), (700, 1140), (155, 1115)]  # координаты для прямого месси
#crop_and_straighten_image('messy straight.jpg', points, output_path)
#points = [(340, 230), (808, 368), (570, 1140), (92, 969)]
#cropped_image = crop_and_straighten_image('messy krivoy.jpg', points)