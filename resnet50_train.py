from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, ReLU, Layer, Input, ZeroPadding2D, AveragePooling2D, Flatten
from tensorflow.keras.initializers import glorot_uniform
from layers import IdentityBlock, ProjectionBlock


# Define
input_layer = Input((224, 224, 1))

layers = ZeroPadding2D((3, 3))(input_layer)

layers = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer=glorot_uniform(seed=0))(layers)
layers = BatchNormalization()(layers)
layers = ReLU()(layers)

layers = MaxPool2D((3, 3), strides=(2, 2))(layers)

layers = ProjectionBlock(filters=64, strides=(1, 1))(layers)
layers = IdentityBlock(filters=64)(layers)
layers = IdentityBlock(filters=64)(layers)

layers = ProjectionBlock(filters=128, strides=(2, 2))(layers)
layers = IdentityBlock(filters=128)(layers)
layers = IdentityBlock(filters=128)(layers)
layers = IdentityBlock(filters=128)(layers)

layers = ProjectionBlock(filters=256, strides=(2, 2))(layers)
layers = IdentityBlock(filters=256)(layers)
layers = IdentityBlock(filters=256)(layers)
layers = IdentityBlock(filters=256)(layers)
layers = IdentityBlock(filters=256)(layers)
layers = IdentityBlock(filters=256)(layers)


layers = ProjectionBlock(filters=512, strides=(2, 2))(layers)
layers = IdentityBlock(filters=512)(layers)
layers = IdentityBlock(filters=512)(layers)

layers = AveragePooling2D(pool_size=(2, 2))(layers)
layers = Flatten()(layers)

layers = Dense(units=256, activation='relu', kernel_initializer=glorot_uniform(seed=0))(layers)
layers = Dense(units=128, activation='relu', kernel_initializer=glorot_uniform(seed=0))(layers)
layers = Dense(units=4, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(layers)

model = Model(inputs=input_layer, outputs=layers)

from tensorflow.keras.optimizers import Adam

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# Train
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("models/resnet50.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq="epoch")
    
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

trdata = ImageDataGenerator(rescale=1./255, rotation_range=30.0)
traindata = trdata.flow_from_directory(directory="images/train",target_size=(224,224), color_mode='grayscale')

vdata = ImageDataGenerator(rescale=1./255)
valdata = vdata.flow_from_directory(directory="images/test", target_size=(224,224), color_mode='grayscale')


from sklearn.utils import class_weight 
import numpy as np

class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(traindata.classes), 
            y=traindata.classes)

train_class_weights = dict(enumerate(class_weights))

hist = model.fit(traindata, steps_per_epoch=100, validation_steps=10, epochs=100, class_weight=train_class_weights, validation_data=valdata, callbacks=[early, checkpoint])


# Plot
import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model Training History")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training Accuracy","Validation Accuracy","Training Loss","Validation Loss"])
plt.show()
plt.savefig('data/resnet50_training_hist.png')
