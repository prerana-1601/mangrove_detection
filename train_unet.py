from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.transform import resize
import math


print("hi1")

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

print("hi2")

N_BANDS = 7
N_CLASSES = 2  # mangrove, not mangrove
CLASS_WEIGHTS = [0.8, 0.2]
N_EPOCHS = 2
UPCONV = True
PATCH_SZ = 160   # should divide by 16
BATCH_SIZE = 12
TRAIN_SZ = 4000  # train size
VAL_SZ = 1000    # validation size

print("hi2")

def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)

#don't know
weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

print("hi3")

trainIds = [str(i).zfill(2) for i in range(1, 5)]  # all availiable ids: from "01" to "32"


if __name__ == '__main__':
    print("hi4")
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading images')
    for img_id in trainIds:
        #img_m = np.nan_to_num(img_m) 
        #img_m = normalize(np.nan_to_num(tiff.imread(r'C:\Users\prerana\Documents\capstone\bandv2\{}.tif'.format(img_id))))
        #np.where(np.isinf(img_m), img_m, 0)
        img=(tiff.imread(r'C:\Users\prerana\Documents\capstone\bandv2\{}.tif'.format(img_id)))
        m = np.isnan(img)
        img[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), img[~m])
        img=img*1000
        img = img.astype('uint16')
        img=np.where(img==0,img+10,img)
        print("7 bands before normalization")
        print('Datatype:', img.dtype)
        print(img.min())
        print(img.max())
        #m = np.isnan(img)
        #img[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), img[~m])
        #img_m =normalize(np.where(np.isinf(img),img,1))
        img_m=normalize(img)
        print("7 bands after normalization")
        print('Datatype:', img_m.dtype)
        print(img_m.min())
        print(img_m.max())
        #img_m=img
        #print(img_m.shape)
        
        
        #img_m = normalize(tiff.imread('./data/mband/{}.tif'.format(img_id)).transpose([1, 2, 0]))
        #mask = tiff.imread('./data/gt_mband/{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
        #mask = tiff.imread(r'C:\Users\prerana\Documents\capstone\maskv2\{}.tif'.format(img_id)) / 255
        #mask = tiff.imread(r'C:\Users\prerana\Documents\capstone\trial_mask\{}.tif'.format(img_id))
        mask = tiff.imread(r'C:\Users\prerana\Documents\capstone\trial_mask\01.tif')
        #print(mask)
        #print(mask.min())
        #print(mask.max())
        l=img_m.shape[0]
        b=img_m.shape[1]
        mask=resize(mask,(l,b,2))
        #mask=mask*1000
        #mask = mask.astype('uint16')
        #mask=np.where(mask==0,mask+10,mask)
        m_m = np.isnan(mask)
        mask[m_m] = np.interp(np.flatnonzero(m_m), np.flatnonzero(~m_m), mask[~m_m])
        #mask=np.nan_to_num(mask)
        mask[mask>0]=1
        mask = mask.astype('float64')
        print(mask.min())
        print(mask.max())
        print(mask.shape)
        print('Datatype:', mask.dtype)
        train_xsz = int(0.75 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')

    def train_net():
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        model = get_model()
        if os.path.isfile(weights_path):
            print("load")
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet1.csv', append=True, separator=';')
        cwd = os.getcwd()
        NAME='tensorboard_unet'
        tboard_log_dir = os.path.join(cwd,NAME)
        tensorboard = TensorBoard(log_dir = tboard_log_dir, write_graph=True, write_images=True)
        #tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        model.save(r'C:\Users\prerana\Documents\capstone\my_model4.h5')
        return model
    

    train_net()
