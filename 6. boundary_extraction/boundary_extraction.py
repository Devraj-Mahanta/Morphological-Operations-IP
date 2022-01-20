import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
original_img = cv2.imread('image.jpg')
img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
_, binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
binary_img = binary_img/255
struct_elem = np.array([(1,1,1,), (1,1,1), (1,1,1)])
img_shape = binary_img.shape
filter_shape = struct_elem.shape
padded_img = np.zeros((img_shape[0]+filter_shape[0]-1,img_shape[1]+filter_shape[1]-1))
for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        padded_img[i+1, j+1] = binary_img[i, j]
erod_img = np.zeros(img_shape)
for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        window = padded_img[i:i+filter_shape[0], j:j+filter_shape[1]]
        result = (window == struct_elem)
        final = np.all(result == True)
        if final:
            erod_img[i, j] = 1
final_img = binary_img - erod_img
fig = plt.figure(figsize=(13, 9))
fig.suptitle('BOUNDARY EXTRACTION', fontsize=16)
rows = 1
columns = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(binary_img, cmap='Greys_r')
plt.axis('off')
plt.title("Original Image")
fig.add_subplot(rows, columns, 2)
plt.imshow(final_img, cmap='Greys_r')
plt.axis('off')
plt.title("Final Image")
plt.show()
