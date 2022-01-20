from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
blur_radius = 1.0
threshold = 50
img = Image.open('image.jpg').convert('L')
img = np.asarray(img)
blur_img = ndimage.gaussian_filter(img, blur_radius)
final_img, nr_objects = ndimage.label(blur_img > threshold)
fig = plt.figure(figsize=(13, 9))
fig.suptitle('CONNECTED COMPONENT', fontsize=16)
rows = 1
columns = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(img, cmap='Greys_r')
plt.axis('off')
plt.title("Input")
fig.add_subplot(rows, columns, 2)
plt.imshow(final_img, cmap='Greys_r')
plt.axis('off')
plt.title("Output")
plt.show()
