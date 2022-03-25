import os.path
import numpy as np
import tifffile as tiff

from building_unet_from_base import *

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

def get_model(input_shape):
    model = build_unet(input_shape)
    model.compile(optimizer=Adam(clipvalue=0.5), loss=BinaryCrossentropy())
    return model

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

img = normalize(tiff.imread('data_zip1/bandv2/01.tif'))
img = img[:512,:512,:]

mask = tiff.imread('data_zip1/maskv2/01.tif') /255
mask =mask[:512,:512,1]

from numpy import newaxis
img = img[newaxis,...]
mask = mask[newaxis,...]

def train_net():
    model = get_model((512,512,7))
    model.fit(img, mask, epochs=1,verbose=2)
    return model

train_net()