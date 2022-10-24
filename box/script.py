import numpy as np
from PIL import Image

#np.random.seed(1)

def get_crop(img:Image, width:int, height:int):
    """
    """
    assert 2 <= width <= 4 and 2 <= height <= 4
    left = (4 - width) * 200
    top = (4 - height) * 200
    l, r, t, b = np.array(np.round(200 * np.random.random(4)), dtype=np.int64)
    new = img.crop((left + l, top + t, 1200 - left - l, 1200 - top - t))
    return new

img = Image.open("6x6.png")
new = get_crop(img, 4, 4)
new.show()

