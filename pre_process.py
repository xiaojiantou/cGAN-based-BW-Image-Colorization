import numpy as np
from PIL import Image
import os

HEIGHT = 175
WIDTH = 250

input_image_folder = os.path.abspath('..') + '\data\original\\'
c_image_folder = os.path.abspath('..') + '\data\colorful\\'
g_image_folder = os.path.abspath('..') + '\data\grayscale\\'


for dirs in os.listdir(input_image_folder):
    for name in os.listdir(input_image_folder+dirs):
        im = Image.open(input_image_folder+dirs+'\\'+name)
        im = im.resize((WIDTH,HEIGHT),Image.ANTIALIAS)

        # resize original image
        if not os.path.exists(c_image_folder+dirs):
            os.makedirs(c_image_folder+dirs)
        im.save(c_image_folder+dirs+'\\'+name)

        # convert rgb image to grayscale
        im = im.convert('L')
        if not os.path.exists(g_image_folder+dirs):
            os.makedirs(g_image_folder+dirs)
        im.save(g_image_folder+dirs+'\\'+name)
