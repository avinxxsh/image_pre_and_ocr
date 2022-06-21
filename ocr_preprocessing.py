import cv2 as cv
import pytesseract
import numpy as np

# opening image file and loading it onto memory
image_file = "sample8.jpg"
img = cv.imread(image_file)

# opening loaded image in new window with a window name
cv.imshow("Original Image", img)
cv.waitKey(0)

# removing shadows from image
rgb_planes = cv.split(img)
result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv.medianBlur(dilated_img, 21)
    diff_img = 255 - cv.absdiff(plane, bg_img)
    norm_img = cv.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv.merge(result_planes)
result_norm = cv.merge(result_norm_planes)

cv.imwrite("temp/shadows_out.png", result)
cv.imshow("Shadow Removed - Non Normalized", result)
cv.imwrite("temp/shadows_out_norm.png", result_norm)
cv.imshow("Shadow Removed -  Normalized", result_norm)


# inverting image - most effective with tesseract v3
inverted_image = cv.bitwise_not(result_norm)
cv.imwrite("temp/inverted.jpeg", inverted_image)
# cv.imshow("Inverted Image", inverted_image)
cv.waitKey(0)

# rescaling image
# option-1
# resized_img = cv.resize(img, (0,0), fx=0.5, fy=0.5)
# option-2
# resized_img = cv.resize(img, (400, 400))
# cv.imwrite("temp/resized_img.jpg", resized_img)
# cv.imshow("Resized Image", resized_img)


# binarization
# first step will be to convert to greyscale
def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


gray_image = grayscale(result_norm)
cv.imwrite("temp/grey.jpeg", gray_image)
# cv.imshow("Gray Image", gray_image)
cv.waitKey(0)

# converting image to grayscale is only the 1st step of binarization - makes the process easier
thresh, im_bw = cv.threshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C + cv.THRESH_BINARY, 11 ,2)
# thresh, im_bw = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# the integers entered are pixel values where 0 is black, 127 is the mid-tone point and 255 is white
cv.imwrite("temp/bw_image.jpeg", im_bw)
cv.imshow("B/W Image", im_bw)
cv.waitKey(0)


# noise removal
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv.dilate(image, kernel, iterations=2)
    kernel = np.ones((1, 1), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    # image = cv.GaussianBlur(image, (3, 3), 0)
    image = cv.medianBlur(image, 1)
    return image


no_noise = noise_removal(im_bw)
cv.imwrite("temp/no_noise.jpeg", no_noise)
cv.imshow("Noise Removed", no_noise)
cv.waitKey(0)


# code snippet to remove shadows from images
# h, w = img.shape
# kernel = np.ones((7, 7), np.uint8)
# dilation = cv2.dilate(img, kernel, iterations=1)
# blurred_dilation = cv2.GaussianBlur(dilation, (13, 13), 0)
# resized = cv2.resize(blurred_dilation, (w, h))
# corrected = img / resized * 255

# # working with thick and thin fonts
# # making font thinner
# def thin_font(image):
#     import numpy as np
#     image = cv.bitwise_not(image)
#     kernel = np.ones((3, 3), np.uint8)
#     image = cv.erode(image, kernel, iterations=1)
#     image = cv.bitwise_not(image)
#     return image
#
#
# eroded_img = thin_font(im_bw)
# cv.imwrite("temp/eroded_img.jpg", eroded_img)
# cv.imshow("Eroded Image", eroded_img)
# cv.waitKey(0)
#
#
# making font thicker
def thick_font(image):
    import numpy as np
    image = cv.bitwise_not(image)
    kernel = np.ones((2, 1), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    image = cv.bitwise_not(image)
    return image


# dilated_img = thick_font(no_noise)
# cv.imwrite("temp/dilated_img.jpg", dilated_img)
# cv.imshow("Dilated Image", dilated_img)
# cv.waitKey(0)
#
#
# # handling rotation and skewed documents
# # the code below will not be effective if image has borders, borders have to be removed first
# skewed_img = cv.imread("skewed1.jpg")
#
#
# def getSkewAngle(cvImage) -> float:
#     # Prep image, copy, convert to gray scale, blur, and threshold
#     newImage = cvImage.copy()
#     gray = cv.cvtColor(newImage, cv.COLOR_BGR2GRAY)
#     blur = cv.GaussianBlur(gray, (9, 9), 0)
#     thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
#
#     # Apply dilate to merge text into meaningful lines/paragraphs.
#     # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
#     # But use smaller kernel on Y axis to separate between different blocks of text
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
#     dilate = cv.dilate(thresh, kernel, iterations=5)
#
#     # Find all contours
#     contours, hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key = cv.contourArea, reverse = True)
#
#     # Find largest contour and surround in min area box
#     largestContour = contours[0]
#     minAreaRect = cv.minAreaRect(largestContour)
#
#     # Determine the angle. Convert it to the value that was originally used to obtain skewed image
#     angle = minAreaRect[-1]
#     if angle < -45:
#         angle = 90 + angle
#     return -1.0 * angle
#
#
# # Rotate the image around its center
# def rotateImage(cvImage, angle: float):
#     newImage = cvImage.copy()
#     (h, w) = newImage.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv.getRotationMatrix2D(center, angle, 1.0)
#     newImage = cv.warpAffine(newImage, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
#     return newImage
#
#
# # Deskew image
# def deskew(cvImage):
#     angle = getSkewAngle(cvImage)
#     return rotateImage(cvImage, -1.0 * angle)
#
#
# # rotated_img = deskew(skewed_img)
# # cv.imwrite("temp/rotated_img.jpg", rotated_img)
# # cv.imshow("Rotated Image", rotated_img)
# # cv.waitKey(0)
#
# starting img to text process
# img_1 = cv.imread(".jpeg")


pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
img_1 = cv.cvtColor(no_noise, cv.COLOR_BGR2RGB)
print(pytesseract.image_to_string(img_1))
# print(pytesseract.image_to_boxes(img_1))

# showing character detection
# hImg, wImg = img_1.shape[0:2]
# boxes = pytesseract.image_to_boxes(no_noise)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     # print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv.rectangle(img_1, (x, hImg - y), (w, hImg - h), (0, 0, 255), 2)

# detecting words
hImg, wImg = img_1.shape[0:2]
boxes = pytesseract.image_to_data(no_noise)
for x, b in enumerate(boxes.splitlines()):
    if x != 0:
        b = b.split()
        print(b)
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv.rectangle(img_1, (x, y), (w + x, h + y), (0, 0, 225), 3)
            cv.putText(img_1, b[11], (x, y), cv.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 2)


cv.imshow("Final Image", img_1)
cv.waitKey(0)
cv.destroyAllWindows()





