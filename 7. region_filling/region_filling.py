import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image = cv2.imread("image.jpg", 0)
im_flood_fill = input_image.copy()
h, w = input_image.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)
im_flood_fill = im_flood_fill.astype("uint8")
cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
img_out = input_image | im_flood_fill_inv
fig = plt.figure(figsize=(7, 5))
fig.suptitle('REGION FILLING', fontsize=16)
rows = 1
columns = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(input_image, cmap='Greys_r')
plt.axis('off')
plt.title("Input Image")
fig.add_subplot(rows, columns, 2)
plt.imshow(img_out, cmap='Greys_r')
plt.axis('off')
plt.title("Output Image")
plt.show()