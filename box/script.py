import numpy as np
from PIL import Image

np.random.seed(1)

def get_crop(img:Image, width:int, height:int):
    """
    """
    assert 2 <= width <= 4 and 2 <= height <= 4
    l = (4 - width) * 200
    t = (4 - height) * 200
    new = img.crop((l, t, 1200 - l, 1200 - t))
    return new

img = Image.open("6x6.png")
new = get_crop(img, 3, 2)
new.show()

