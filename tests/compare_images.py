"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from atmotomo import FortranFile as FF, ColoredParam
import os
import glob


def load_imgs(base_path):

    imgs_path = glob.glob(os.path.join(base_path, 'img_*.bin').replace('\\','/'))

    imgs = []
    for i in range(int(len(imgs_path)/3)):
        imgs.append({})

    for path in imgs_path:
        temp, name = os.path.split(path)
        temps = os.path.splitext(name)[0].split('_')
        color, num = temps[-2:]

        f = FF(path)
        data = f.readReals()  
        imgs[int(num)][color] = data.reshape(64, 64)
        
    return imgs


def main():
    """Main doc """

    ref_imgs_base = r'results\ref_images'
    res_imgs_base = r'results\res_images'

    ref_imgs = load_imgs(ref_imgs_base)
    res_imgs = load_imgs(res_imgs_base)

    errs = []
    for ref_img, res_img in zip(ref_imgs, res_imgs):
        err = 0
        for color in ('red', 'green', 'blue'):
            err += np.linalg.norm(ref_img[color].astype(np.float) - res_img[color].astype(np.float))

        errs.append(err)

    errs = np.array(errs)

    for i in np.argsort(errs)[-6:]:
        ref_img, res_img = ref_imgs[i], res_imgs[i]

        plt.figure()

        img1 = np.dstack((ref_img['red'], ref_img['green'], ref_img['blue']))
        img2 = np.dstack((res_img['red'], res_img['green'], res_img['blue']))

        img1 = img1/img1.max()
        img2 = img2/img2.max()

        plt.subplot(121)
        plt.imshow(img1)

        plt.subplot(122)
        plt.imshow(img2)

    plt.show()


if __name__ == '__main__':
    main()
