from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Convolution2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
#One Hot Encode the Labels

num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
(x_train,x_valid) = x_train[5000:],x_train[:5000]
(y_train,y_valid) = y_train[5000:],y_train[:5000]

print('\n\tX Train Shape ',x_train.shape)

print('\nNumber of Training Samples {}'.format(x_train.shape[0]))
print('\nNumber of Validation Samples {}'.format(x_valid.shape[0]))
print('\nNumber of Test Samples{}'.format(x_test.shape[0]))


#Create augmented Image Generator
datagen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,vertical_flip=True,rotation_range=40,shear_range=0.2,zoom_range=0.2)
datagen.fit(x_train)



print('''\n\n
        Steps for constructing any CNN / Model\n
        1.  Load the Data set and split into Train / Test Set .
        2.  Normalize the data sets (x_train -  x_test / Largest(value of ele)
        3.  One Hot Encode the Labels [to_categorical]
        4.  Break Training Set to Training + VaLidation Set
        5.  Start by starting with a Sequential Model 
        6.  Add a Convolutional2D layer and start with basic rules above.
                Remember, the 1st Convilutional Layer needs to have input shape defined as per our input
        7.  Add a MaxPool with Pool_size =2, to reduce width and height
        8.  Keep repeating steps 6&7 each time increasing filter size.
        9.  Add a Dropout before feeding it into a Regular MLP
        10. First Flatten the model for the outputs from the Convolutional Layer.
        11. Now define a Regular MLP, with a Dense (with relu)-> Dropout -> Dense (number of outputs,'softmax')
        12. The output of the softmax will give the probabilities of the answers.
        13. Rest of the Steps are same as a Regular MLP 
        14. Above steps finalize Model Architechture Design. Next is to compile and add a loss function.
        15. Use model.compile to add a loss function and optimizer, with a target metric
        16. Next is to Train the Model by doing the below
        17. Use ModelCheckPoint, to save the best weights only and resuse
        18. Use Model.Fit which adds in callbacks, Batch_Size, num of epochs and passes in training and validation sets
        19. Once the Model is trained, call the .load_weights to load the same file in Model CheckPoint
        20. Now call Model.Evaluate on the Test set with the loaded weights to test accuracy\n''')

cnn_model = Sequential()
cnn_model.add(Convolution2D(filters =16,kernel_size=2,strides=1,padding='same',activation='relu',input_shape=(32,32,3)))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Convolution2D(filters=32,kernel_size=2,strides=1,padding='same',activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Convolution2D(filters=64,kernel_size=2,strides=1,padding='same',activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.3))
cnn_model.add(Flatten())
cnn_model.add(Dense(500,activation='relu'))
cnn_model.add(Dropout(0.4))
cnn_model.add(Dense(num_classes,activation='softmax'))
cnn_model.summary()

cnn_model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='rmsprop')

cnn_checkpointer = ModelCheckpoint(filepath='aug_cnn_mlp_check.hdf5',save_best_only=True,verbose=1)
batch_size =32
n_epoch=100
hist = cnn_model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
                               steps_per_epoch=x_train.shape[0]//batch_size,
                               epochs=n_epoch,verbose=2,callbacks=[cnn_checkpointer],validation_data=(x_valid,y_valid))

#hist = cnn_model.fit(x_train,y_train,batch_size=32,epochs=100,verbose=2,callbacks=[cnn_checkpointer],validation_data=(x_valid,y_valid),shuffle=True)

cnn_model.load_weights(filepath='aug_cnn_mlp_check.hdf5')
cnn_score = cnn_model.evaluate(x_test,y_test,verbose=0)
print('\nAccuracy of the CNN Model = {}'.format(cnn_score[1]))