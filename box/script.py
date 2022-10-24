import numpy as np
from PIL import Image

np.random.seed(1)

def get_crop(img:Image, width:int, height:int):
    """
    """

img = Image.open("box.png")
min_crop = 30
max_crop = 200 - min_crop

