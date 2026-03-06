import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

im = Image.open("/home/mmazur/average.png")
im_arr = np.array(im)

# plt.imshow(im_arr)
# plt.show()

im_line = im_arr[540:543,540:1050]
im_mean = np.mean(im_line, axis=0)

y = list(im_mean)

x = list(range(0,len(y)))

plt.plot(x, y)
plt.show()

im = Image.open("/home/mmazur/max.png")
im_arr = np.array(im)

plt.imshow(im_arr)
plt.show()

im_line = im_arr[528:532,980:1920]
im_mean = np.mean(im_line, axis=0)

y = list(im_mean)

x = list(range(0,len(y)))

plt.plot(x, y)
plt.show()