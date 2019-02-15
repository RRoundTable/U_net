import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


#
from model import U_net
from model import mean_iou
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.python.keras.layers import Input, Conv2D,Dropout,MaxPooling2D, Concatenate,Lambda,Conv2DTranspose
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Model
import numpy as np
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(description='U-net for semantic segmentation')

parser.add_argument('--version', default="predict", type=str,
                    help='train OR predict')




# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = './input/stage1_train/'
TEST_PATH = './input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1] # os.walk 하위 디렉토리 검색
test_ids = next(os.walk(TEST_PATH))[1]


X_train=np.zeros((len(train_ids), IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
Y_train=np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')

sys.stdout.flush() # 화면에 출력을


for n,id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path=TRAIN_PATH+id_

    img=imread(path+"/images/"+id_+'.png')[:,:,:IMG_CHANNELS]
    img=resize(img, (IMG_HEIGHT,IMG_WIDTH), mode="constant",preserve_range=True)
    X_train[n]=img
    mask=np.zeros((IMG_HEIGHT,IMG_WIDTH,1),dtype=np.bool)
    for mask_file in next(os.walk(path+"/masks/"))[2]:
        mask_=imread(path+"/masks/"+mask_file)
        mask_=np.expand_dims(resize(mask_,(IMG_HEIGHT,IMG_WIDTH),mode="constant",preserve_range=True),axis=-1)
        mask=np.maximum(mask,mask_) # ?
    Y_train[n]=mask


# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img


print('Done!')

def train():
    model=U_net(np.array(X_train))

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
                        callbacks=[earlystopper, checkpointer])



def predict(X_train,X_test):

    model=load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
    preds_train=model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    preds_val=model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
    preds_test=model.predict(X_test, verbose=1)
    print("preds_train : {}".format(preds_train.shape))
    print("preds_val : {}".format(preds_val.shape))
    print("preds_test_t : {}".format(preds_test.shape))

    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    print("preds_train : {}".format(preds_train_t))
    print("preds_val : {}".format(preds_val_t))
    print("preds_test_t : {}".format(preds_test_t))
    preds_test_upsampled=[]
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze([preds_test[i]]),(sizes_test[i][0], sizes_test[i][1]),
                                           mode="constant",preserve_range=True))

    # Perform a sanity check on some random training samples
    X_train=np.array(X_train)
    preds_val_t=np.array(preds_val_t)
    preds_train_t=np.array(preds_train_t)

    ix = random.randint(0, len(preds_train_t))
    # save image
    fig=plt.figure()
    ax=[]
    for i in range(3):
        ax.append(fig.add_subplot(1,3,i+1))

    ax[0].imshow(X_train[ix])
    ax[0].set_title("X_train")
    ax[1].imshow(np.squeeze(Y_train[ix]))
    ax[1].set_title("Y_train")
    ax[2].imshow(np.squeeze(preds_train_t[ix]))
    ax[2].set_title("X_pred")
    plt.savefig("./result/training_sample_{}.png".format(ix))

    # plt.subplot(2,3)
    # imshow(X_train[ix])
    # plt.savefig("./result/X_train_{}.png".format(ix))
    # imshow(np.squeeze(Y_train[ix]))
    # plt.savefig("./result/Y_train_{}.png".format(ix))
    # imshow(np.squeeze(preds_train_t[ix]))
    # plt.savefig("./result/preds_train_t{}.png".format(ix))

    # Perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t))
    fig=plt.figure()
    ax=[]
    for i in range(3):
        ax.append(fig.add_subplot(1,3,i+1))

    ax[0].imshow(X_train[int(X_train.shape[0] * 0.9):][ix])
    ax[0].set_title("X_val")
    ax[1].imshow(np.squeeze(Y_train[int(Y_train.shape[0] * 0.9):][ix]))
    ax[1].set_title("Y_val")
    ax[2].imshow(np.squeeze(preds_val_t[ix]))
    ax[2].set_title("X_val_pred")
    plt.savefig("./result/validation_sample_{}.png".format(ix))

    # imshow(X_train[int(X_train.shape[0] * 0.9):][ix])
    # plt.savefig("./result/X_train_val_{}.png".format(ix))
    # imshow(np.squeeze(Y_train[int(Y_train.shape[0] * 0.9):][ix]))
    # plt.savefig("./result/Y_train_val_{}.png".format(ix))
    # imshow(np.squeeze(preds_val_t[ix])) # error
    # plt.savefig("./result/preds_val_t{}.png".format(ix))
if __name__=="__main__":
    args=parser.parse_args()
    if args.version =="predict": # prediction
        predict(X_train,X_test)
    else :
        train()
