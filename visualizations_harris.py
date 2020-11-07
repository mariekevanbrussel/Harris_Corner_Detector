from harris_corner_detector_plot import harris_corner_detector_plot
import imutils
import cv2

toy_image = cv2.imread('images/person_toy.jpg')
ping_image = cv2.imread('images/pingpong.jpeg')

thresholds = [60, 600, 6000, 3000000]

#Plotting the corner points for different threshold values
corners_toy = harris_corner_detector_plot(toy_image, thresh=6000, sigma=1, window_size=5)
corners_ping = harris_corner_detector_plot(ping_image, thresh=6000, sigma=1, window_size=5)

for t in thresholds:
    corners_toy = harris_corner_detector_plot(toy_image, thresh=t, sigma=1, window_size=5)
    corners_ping = harris_corner_detector_plot(ping_image, thresh=t, sigma=1, window_size=5)

    cv2.imwrite('images/corners_toy_' + str(t) + '.jpg', corners_toy)
    cv2.imwrite('images/corners_ping_' + str(t) + '.jpg', corners_ping)

# #Rotating the image and plotting
rot_45 = imutils.rotate_bound(toy_image, angle=45)
rot_90 = imutils.rotate_bound(toy_image, angle=90)
thresh = 300000

corner = harris_corner_detector_plot(toy_image, thresh, sigma=1, window_size=5)
corner_45 = harris_corner_detector_plot(rot_45, thresh, sigma=1, window_size=5)
corner_90 = harris_corner_detector_plot(rot_90, thresh, sigma=1, window_size=5)

cv2.imwrite('images/corner_no_rotation.jpg', corner)
cv2.imwrite('images/corner_rotation_45.jpg', corner_45)
cv2.imwrite('images/corner_rotation_90.jpg', corner_90)
