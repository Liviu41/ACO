import time
import numpy
import numpy as np
from operator import truediv
import scipy.io as sio
import os
import pandas as pd
import seaborn as sns
import spectral
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from sklearn.decomposition import IncrementalPCA
import keras
from keras.layers import Dropout, Input, Conv2D, Conv3D, MaxPool3D, Flatten, Dense, Reshape, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.optimizers import Adam
import Auxiliary_Methods as am


classLabels = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn' ,'Grass-pasture', 
                'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 
                'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat', 
                'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers']


HSI = sio.loadmat('..\..\DB\Indian_Pines\Indian_pines_corrected.mat')['indian_pines_corrected']
gt = labels = sio.loadmat('..\..\DB\Indian_Pines\Indian_pines_gt.mat')['indian_pines_gt']

# applu pca
HSIwPCA = am.pca(HSI, 20)
HSIwPCA.shape

# create image patches
HSIwPCA, gt = am.HSI3dPatches(HSIwPCA, gt, windowSize = 11)

# split trai - test
X_train, X_test, y_train, y_test = am.trainTestSplit(HSIwPCA, gt, 0.70)
#X_train.shape, X_test.shape, y_train.shape, y_test.shape

# split train - val
X_train, X_val, y_train, y_val = am.trainTestSplit(X_train, y_train, 0.30)
#X_train.shape, X_val.shape, y_train.shape, y_val.shape

X_train = X_train.reshape(-1, 11, 11, 20, 1)
y_train = np_utils.to_categorical(y_train)
X_val = X_val.reshape(-1, 11, 11, 20, 1) 
y_val = np_utils.to_categorical(y_val)
X_train.shape, y_train.shape, X_val.shape, y_val.shape


input_layer = Input((11, 11, 20, 1))

conv_layer1 = Conv3D(filters = 8, kernel_size=(3, 3, 7), activation = 'relu')(input_layer)
conv_layer2 = Conv3D(filters = 16, kernel_size=(3, 3, 5), activation = 'relu')(conv_layer1)
conv_layer3 = Conv3D(filters = 32, kernel_size=(3, 3, 3), activation = 'relu')(conv_layer2)
conv_layer4 = Conv3D(filters = 64, kernel_size=(3, 3, 3), activation = 'relu')(conv_layer3)

flatten_layer = Flatten()(conv_layer4)

dense_layer1 = Dense(units = 256, activation = 'relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units = 128, activation = 'relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=16, activation = 'softmax')(dense_layer2)

# define the model with input layer and output layer
model = Model(inputs = input_layer, outputs = output_layer)
model.summary()

# compiling the model
adam = Adam(lr = 0.001, decay = 1e-06)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

# Train and fit
start = time.time()
resTrain = model.fit(x = X_train, y = y_train, batch_size = 256, epochs = 27, validation_data = (X_val, y_val))
end = time.time()
trainingTime = end - start


# Plot Training and Validation loss and Accuracy 
plt.figure(figsize=(7,7))
plt.rcParams.update({'font.size': 18})
plt.grid()
plt.plot(resTrain.history['loss'])
plt.plot(resTrain.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Training and Validation loss')
plt.legend(['Training','Validation'], loc='upper right')
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(7,7))
plt.ylim(0,1.1)
plt.grid()
plt.plot(resTrain.history['accuracy'])
plt.plot(resTrain.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Training and Validation Accuracy')
plt.legend(['Training','Validation'])
plt.show()

# test
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

# reshape test data
X_test = X_test.reshape(-1, 11, 11, 20, 1)
y_test = np_utils.to_categorical(y_test)
#X_test.shape, y_test.shape

# computing accuracy
classification, confusion, Test_loss, Test_accuracy, each_acc, kappa, classLabels, y_pred = am.results(X_test, y_test, model, classLabels)
classification = str(classification)
confusion = str(confusion)
print("classification = " + str(classification))
print("Test_accuracy = " + str(Test_accuracy))
print("Training time = " + str(trainingTime))

## Draw Confusion Matrix
confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred, labels=np.unique(np.argmax(y_test, axis=1)))
cm_sum = np.sum(confusion, axis=1, keepdims=True)
cm_perc = confusion / cm_sum.astype(float) * 100
annot = np.empty_like(confusion).astype(str)
nrows, ncols = confusion.shape
for i in range(nrows):
    for j in range(ncols):
        c = confusion[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
            annot[i, j] = ''
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)
cm = pd.DataFrame(confusion, index=np.unique(classLabels), columns=np.unique(classLabels))
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=(15,10))
plt.rcParams.update({'font.size': 12})
sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# data_path = os.path.join(os.getcwd(),'C:\Projects\ACO\DB\Indian_Pines')
# HSI = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
# gt = labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
# # Check the Dimensions of HSI
# height = gt.shape[0]
# width = gt.shape[1]
# winSize = 11
# # Dimensional Reduction and zero padding
# HSI = am.pca(HSI, 20)
# HSI = am.padding(HSI, 11//2)
# # Calculate the predicted Ground Truths
# outputs = np.zeros((height,width))
# for i in range(height):
#     for j in range(width):
#         target = int(gt[i,j])
#         if target == 0 :
#             continue
#         else :
#             image_patch = am.Patch(HSI,i,j, winSize)
#             X_test_image = image_patch.reshape(1,image_patch.shape[0],
#                                                 image_patch.shape[1], image_patch.shape[2], 1).astype('float32')                                   
#             prediction = (model.predict(X_test_image))
#             prediction = np.argmax(prediction, axis=1)
#             outputs[i][j] = prediction + 1

# plt.figure(figsize=(8, 6))
# plt.imshow(outputs, cmap='jet')
# plt.colorbar()
# plt.axis('off')
# plt.title('Prediction')
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.imshow(gt, cmap='jet')
# plt.colorbar()
# plt.axis('off')
# plt.title('Ground Truth')
# plt.show()