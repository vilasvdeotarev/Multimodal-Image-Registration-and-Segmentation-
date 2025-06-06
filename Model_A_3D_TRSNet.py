import numpy as np
import cv2 as cv
from keras.src.models import Model
from keras.src.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split


def A3D_TRegnet(input_shape, num_classes, sol):
    inputs = Input(shape=input_shape)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    up2 = UpSampling3D(size=(2, 2, 2))(pool1)
    up2 = Conv3D(64, (2, 2, 2), activation='relu', padding='same')(up2)
    merge2 = concatenate([conv1, up2], axis=4)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge2)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    output = Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv2)
    model = Model(inputs=inputs, outputs=output)
    return model


def Model_A_3D_TRSNet(Data, Target, sol=None):
    if sol is None:
        sol = [5, 0.01, 100]
    input_shape = (32, 32, 32, 3)
    num_classes = Target.shape[-1]
    tar = []
    for i in range(len(Target)):
        if len(Target[i].shape) != 3:
            targ = cv.cvtColor(Target[i], cv.COLOR_GRAY2RGB)
            tar.append(targ)
        else:
            tar.append(Target[i])
    tar = np.asarray(tar)

    IMG_SIZE = 32

    Images = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE * IMG_SIZE, 3))
        Images[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, IMG_SIZE, 3))

    Targets = np.zeros((tar.shape[0], IMG_SIZE, IMG_SIZE, IMG_SIZE, 3))
    for i in range(tar.shape[0]):
        temp_1 = np.resize(tar[i], (IMG_SIZE * IMG_SIZE * IMG_SIZE, 3))
        Targets[i] = np.reshape(temp_1, (IMG_SIZE, IMG_SIZE, IMG_SIZE, 3))

    X_train, X_test, y_train, y_test = train_test_split(Images, Targets, test_size=0.25, random_state=0)

    model = A3D_TRegnet(input_shape, num_classes, sol)
    model.summary()
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=sol[1]), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=4, steps_per_epoch=int(sol[2]), validation_data=(X_test, y_test))
    predictions = model.predict(Images)
    Pred = np.zeros((predictions.shape[0], Data.shape[1], Data.shape[2], 3))
    for i in range(predictions.shape[0]):
        temp_1 = np.resize(predictions[i], (Data.shape[1] * Data.shape[2], 3))
        Pred[i] = np.reshape(temp_1, (Data.shape[1], Data.shape[2], 3))
    registration = model.predict(Images)
    Reg = np.zeros((registration.shape[0], Data.shape[1], Data.shape[2], 3))
    for i in range(registration.shape[0]):
        temp_2 = np.resize(registration[i], (Data.shape[1] * Data.shape[2], 3))
        Reg[i] = np.reshape(temp_2, (Data.shape[1], Data.shape[2], 3))

    return Reg, Pred
