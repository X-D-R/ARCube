import cv2 as cv
def draw_tracks(mask, frame, good_new, good_old):
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
        frame = cv.circle(frame, (int(a), int(b)), 2, (0, 0, 255), -1)
    return mask, frame