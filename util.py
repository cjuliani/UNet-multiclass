from __future__ import print_function
from __future__ import division
import numpy as np
from PIL import Image


def remove_transparency(im, bg_colour=(255, 255, 255)):
    """Remove alpha channel from RGBA image"""
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        # Need to convert to RGBA if LA format due to a bug in PIL (see http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color
        # Must be RGBA because paste requires both images have the same format
        # (see http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg.convert('RGB')

    else:
        # return image if no alpha channel
        return im


def resize_array_RGB(im, size):
    """Resize RGB image"""
    img = Image.fromarray(im.astype('uint8'), 'RGB')
    img = img.resize((size, size))

    return np.array(img)


def resize_array(im, size, resample=0):
    """Resize the blob array to given size"""
    img = Image.fromarray(im)
    if type(size) == int:
        img = img.resize((size, size), resample)
    elif (type(size) == int) or (type(size) == float):
        img = img.resize(size, resample)

    return np.array(img)


def get_image_array(imgs, img_size):
    """Get array of input images"""
    images = []
    for i in range(len(imgs)):
        img_0 = []
        for img_ in imgs[i]:
            img = remove_transparency(Image.open(img_))
            img = img.resize((img_size, img_size))
            img = np.array(img)
            img = img.astype(np.float32)
            if len(img.shape) == 3:
                # normalize by 255 if RGB mode
                img = np.multiply(img, 1.0 / 255.0)
            else:
                # otherwise, normalize data by min-max scaling
                # given that no-data values are 0s while the true minimum is beyond 0
                img = ((img - np.min(img[img != 0])) / (np.max(img) - np.min(img[img != 0]))).clip(min=0)
            img = np.reshape(img, (img_size * img_size, -1))
            img_0.append(img)
        image = np.concatenate(img_0, axis=-1)
        image = np.reshape(image, (img_size, img_size, image.shape[-1]))
        images.append(image)

    images = np.array(images)
    return images
