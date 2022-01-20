import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
original_img = cv2.imread('image.jpg')
img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
_, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
binary_img = binary_img/255
struct_img = cv2.imread('structuring_element.jpg')
struct_img_r = cv2.cvtColor(struct_img, cv2.COLOR_BGR2GRAY)
_, struct_elem1_img = cv2.threshold(struct_img_r, 127, 255, cv2.THRESH_BINARY)
struct_elem1 = struct_elem1_img/255
def img_complement(img):
    img_shape = img.shape
    complement_img = np.zeros(img_shape)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if binary_img[i,j] == 0:
                complement_img[i, j] = 1
    return complement_img
struct_elem2 = img_complement(struct_elem1)
def padding(img, filter):
    img_shape = img.shape
    filter_shape = filter.shape
    padded_img = np.zeros((img_shape[0] + filter_shape[0] - 1, img_shape[1] + filter_shape[1] - 1))
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            padded_img[i + 1, j + 1] = img[i, j]
    return padded_img
def erosion(img, filter):
    img_shape = img.shape
    erod_img = np.zeros(img_shape)
    filter_shape = filter.shape
    padded_img = padding(img, filter)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            window = padded_img[i:i+filter_shape[0], j:j+filter_shape[1]]
            result = (window == filter)
            final = np.all(result == True)
            if final:
                erod_img[i, j] = 1
            else:
                erod_img[i, j] = 0
    return erod_img
def intersect(img1, img2):
    shape = img1.shape
    intersect_img = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i, j] == img2[i, j]:
                intersect_img[i, j] = img2[i, j]
            else:
                intersect_img[i, j] = 0
    return intersect_img
a_erod_b1 = erosion(binary_img, struct_elem1)
ac_erod_b2 = erosion(img_complement(binary_img), struct_elem2)
final_img = intersect(a_erod_b1, ac_erod_b2)
fig = plt.figure(figsize=(13, 9))
fig.suptitle('HIT AND MISS', fontsize=16)
rows = 1
columns = 3
fig.add_subplot(rows, columns, 1)
plt.imshow(binary_img, cmap='Greys_r')
plt.axis('off')
plt.title("Input Image")
fig.add_subplot(rows, columns, 2)
plt.imshow(struct_elem1, cmap='Greys_r')
plt.axis('off')
plt.title("Structuring Element")
fig.add_subplot(rows, columns, 3)
plt.imshow(final_img, cmap='Greys_r')
plt.axis('off')
plt.title("Output Image")
plt.show()
