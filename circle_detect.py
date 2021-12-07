import cv2
import numpy as np

if __name__ == '__main__':

    # Read image.
    img = cv2.imread('test/scene_images/screenshot_1.PNG', cv2.IMREAD_COLOR)

    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                              cv2.THRESH_BINARY, 11, 2)

    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, img.shape[0] / 5, param1=50,
                                        param2=30, minRadius=round(img.shape[0] / 18), maxRadius=round(img.shape[0] / 10))

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", img)

    radius_arr = detected_circles[:, :, 2]
    radius_mean = np.mean(radius_arr)
    print(radius_mean)
    cv2.waitKey()
