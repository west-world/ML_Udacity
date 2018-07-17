import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Convolution2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

'''
Conv2D(filters, kernel_size, strides, padding, activation='relu', input_shape)
filters - The number of filters.
kernel_size - Number specifying both the height and width of the (square) convolution window.
There are some additional, optional arguments that you might like to tune:

strides - The stride of the convolution. If you don't specify anything, strides is set to 1.
padding - One of 'valid' or 'same'. If you don't specify anything, padding is set to 'valid'.

Valid ==  edges will get chopped off as filters will fall outside of the image
Same - Adds a padding of 0 to the image to make sure that filter does not fall outside the image.
activation - Typically 'relu'. If you don't specify anything, no activation is applied. 
You are strongly encouraged to add a ReLU activation function to every convolutional layer in your networks.

NOTE: Do not include the input_shape argument if the convolutional layer is not the first layer in your network.

Example #1
Say I'm constructing a CNN, and my input layer accepts grayscale images that are 200 by 200 pixels 
(corresponding to a 3D array with height 200, width 200, and depth 1). 
Then, say I'd like the next layer to be a convolutional layer with 16 filters, 
each with a width and height of 2. When performing the convolution,'
 I'd like the filter to jump two pixels at a time. 
 I also don't want the filter to extend outside of the image boundaries; 
 in other words, I don't want to pad the image with zeros. Then, to construct this convolutional layer, 
 I would use the following line of code:

Conv2D(filters=16, kernel_size=2, strides=2, activation='relu', input_shape=(200, 200, 1))


Example #2
Say I'd like the next layer in my CNN to be a convolutional layer that takes the
 layer constructed in Example 1 as input. 
 Say I'd like my new layer to have 32 filters, each with a height and width of 3. 
 When performing the convolution, I'd like the filter to jump 1 pixel at a time. 
 I want the convolutional layer to see all regions of the previous layer, 
 and so I don't mind if the filter hangs over the edge of the previous 
 layer when it's performing the convolution. Then, to construct this convolutional layer, 
 I would use the following line of code:

Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'

Example #3
If you look up code online, it is also common to see convolutional layers in Keras in this format:

Conv2D(64, (2,2), activation='relu')

In this case, there are 64 filters, each with a size of 2x2, and the layer has a ReLU activation function. 
The other arguments in the layer use the default values, 
so the convolution uses a stride of 1, and the padding has been set to 'valid'.
'''

'''
K - the number of filters in the convolutional layer
F - the height and width of the convolutional filters
S - the stride of the convolution
H_in - the height of the previous layer
W_in - the width of the previous layer
D_in is the last value in the input_shape tuple.

If padding = 'same', then the spatial dimensions of the convolutional layer are the following:

height = ceil(float(H_in) / float(S))
width = ceil(float(W_in) / float(S))

If padding = 'valid', then the spatial dimensions of the convolutional layer are the following:

height = ceil(float(H_in - F + 1) / float(S))
width = ceil(float(W_in - F + 1) / float(S))

The number of parameters in the convolutional layer is given by K*F*F*D_in + K[One Bias applied to each of the K Filters].


'''


print('''\n\n
         The whole idea of CNN's is to create filters which increase the depth and 
         reduce the height and width of an image.
         An image generally, has say 32*32*3(RGB) , which means depth of 3, and 32 for ht. n width\n\n ''')

print('''\n\n                     Basic Loose laws of CNN \n
        1. Padding is generally kept 'same' to get maximum Coverage
        2. Kernel size varies from 2X2 to maximum of 5X5
        3. Activation Function is always Relu
        4. Number of Filters gradually increase at each CNN layer to increase depth. 
        5. Mostly its a power of 2, starting @ 16 ->32->64 as we progress
        6. Stride is generally kept = 1 to get same ht n width of convolutional layer as the image.
        7. Hence to reduce spatial Dimensions(ht / width) we perform Max Pooling 
        8. One MaxPooling layer comes between two or more Convolutional Layers.
        9. Generally, MaxPooling Layer Pool Size = 2, Stride = 2, which effectively halves spatial Dim
           and padding of 'valid'
        \n\n''')



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
cnn_model.add(Convolution2D(filters =64,kernel_size=2,strides=1,padding='same',activation='relu',input_shape=(32,32,3)))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Convolution2D(filters=128,kernel_size=2,strides=1,padding='same',activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Convolution2D(filters=256,kernel_size=2,strides=1,padding='same',activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=2))
cnn_model.add(Dropout(0.3))
cnn_model.add(Flatten())
cnn_model.add(Dense(500,activation='relu'))
cnn_model.add(Dropout(0.4))
cnn_model.add(Dense(num_classes,activation='softmax'))
cnn_model.summary()
cnn_model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='rmsprop')
cnn_checkpointer = ModelCheckpoint(filepath='cnn_mlp_check.hdf5',save_best_only=True,verbose=2)
hist = cnn_model.fit(x_train,y_train,batch_size=32,epochs=50,verbose=2,callbacks=[cnn_checkpointer],validation_data=(x_valid,y_valid),shuffle=True)

cnn_model.load_weights(filepath='cnn_mlp_check.hdf5')
cnn_score = cnn_model.evaluate(x_test,y_test,verbose=0)
print('\nAccuracy of the CNN Model = {}'.format(cnn_score[1]))