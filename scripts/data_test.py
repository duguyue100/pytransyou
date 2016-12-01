"""Data test.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import transyou
from transyou import util
from transyou import transfun

#  util.save_dataset("image_db.h5")

data = util.load_dataset("image_db.h5")
image = imread(os.path.join(transyou.TRANSYOU_RES, "favicon.jpg"))
print image.shape

res_image = transfun.trans_you(image, data, target_size=(16, 16))

imsave(os.path.join(transyou.TRANSYOU_RES, "favicon-out.png"), res_image)
print res_image.shape

plt.figure()
plt.imshow(res_image)
plt.show()
