import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image = cv2.imread('image.jpg', 0)
kernel = np.array((
        [0, 1, 0],
        [1, -1, 1],
        [0, 1, 0]), dtype="int")
hitmiss = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
thin = input_image - hitmiss
fig = plt.figure(figsize=(7, 5))
fig.suptitle('THINNING', fontsize=16)
rows = 1
columns = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(input_image, cmap='Greys_r')
plt.axis('off')
plt.title("Input Image")
fig.add_subplot(rows, columns, 2)
plt.imshow(thin, cmap='Greys_r')
plt.axis('off')
plt.title("Output Image")
plt.show()