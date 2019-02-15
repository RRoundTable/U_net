

from tensorflow.python.keras.layers import Input, Conv2D,Dropout,MaxPooling2D,Lambda,Conv2DTranspose
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.merge import concatenate as Concatenate
from tensorflow.python.keras import Model
import numpy as np
import tensorflow as tf

# build U-net Model

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


def mean_iou(y_true, y_pred):
    """
    https://www.tensorflow.org/api_docs/python/tf/metrics/mean_iou
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    object detection에서 주로 활용되는 평가지표
    :param y_true:
    :param y_pred:
    :return: IOU
    """
    prec=[]
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_=tf.to_int32(y_pred>t)
        score, up_opt=tf.metrics.mean_iou(y_true,y_pred_,2)  # label, pred, num_classes
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]): # up_opt를 먼저 실행하고 with 아래의 문장을 실행
            score=tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec),axis=0)




def U_net(inputs):
    """

    :param inputs: train data
    :return:
    """
    input_shape=inputs[0].shape
    inputs=Input(input_shape)
    s=Lambda(lambda  x : x/255)(inputs)

    c1=Conv2D(16,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(s)
    c1=Dropout(0.1)(c1)
    c1=Conv2D(16,(3,3),activation='elu', kernel_initializer="he_normal",padding="same")(c1)
    p1=MaxPooling2D((2,2))(c1)

    c2=Conv2D(32,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(p1)
    c2=Dropout(0.1)(c2)
    c2=Conv2D(32,(3,3),activation='elu', kernel_initializer="he_normal",padding="same")(c2)
    p2=MaxPooling2D((2,2))(c2)

    c3=Conv2D(64,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(p2)
    c3=Dropout(0.1)(c3)
    c3=Conv2D(64,(3,3),activation='elu', kernel_initializer="he_normal",padding="same")(c3)
    p3=MaxPooling2D((2,2))(c3)

    c4=Conv2D(128,(3,3), activation="elu", kernel_initializer="he_normal",padding="same")(p3)
    c4=Dropout(0.1)(c4)
    c4=Conv2D(128,(3,3),activation='elu', kernel_initializer="he_normal",padding="same")(c4)
    p4=MaxPooling2D((2,2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6=Conv2DTranspose(128,(2,2),strides=(2,2),activation="elu", kernel_initializer="he_normal", padding="same")(c5)
    u6=Concatenate([u6,c4])
    c6=Conv2D(128,(3,3),activation="elu",kernel_initializer="he_normal",padding="same")(u6)
    c6=Dropout(0.2)(c6)
    c6=Conv2D(128, (3,3),activation="elu", kernel_initializer="he_normal",padding="same")(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs=Conv2D(1,(1,1), activation='sigmoid')(c9)

    model=Model(inputs, outputs)
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[mean_iou])
    model.summary()

    return model