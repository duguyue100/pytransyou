"""Init PyTransYou package.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
from os.path import join

TRANSYOU_PATH = os.environ["TRANSYOU_PATH"]
TRANSYOU_RES = join(TRANSYOU_PATH, "res")

if not os.path.isdir(TRANSYOU_PATH):
    os.makedirs(TRANSYOU_PATH)

if not os.path.isdir(TRANSYOU_RES):
    os.makedirs(TRANSYOU_RES)
