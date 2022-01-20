import cv2
import matplotlib.pyplot as plt
image = cv2.imread('Hand.png')
original_image = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50,200)
contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(image, [hull],0,(0,255,0),2)
fig = plt.figure(figsize=(7, 5))
fig.suptitle('CONVEX HULL DETECTION', fontsize=16)
rows = 1
columns = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(original_image, cmap='Greys_r')
plt.axis('off')
plt.title("Input Image")
fig.add_subplot(rows, columns, 2)
plt.imshow(image, cmap='Greys_r')
plt.axis('off')
plt.title("Output Image")
plt.show()
