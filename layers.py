from tensorflow.keras.layers import Conv2D, Add, BatchNormalization, ReLU
from tensorflow.keras.initializers import glorot_uniform

def projection_block(X, filters, s):
    X_shortcut = X

    X = Conv2D(filters=filters, kernel_size=(1, 1), strides=s, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = ReLU()(X)

    X = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = ReLU()(X)

    X = Conv2D(filters=filters * 4, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    X_shortcut = Conv2D(filters=filters * 4, kernel_size=(1, 1), strides=s, padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Add()([X, X_shortcut])
    X = ReLU()(X)

    return X


# Wanted to use subclass of Layer but for some reason using class based approach led to Out of memory error while training on batch size > 5.
def ProjectionBlock(filters, strides):
    return lambda layer: projection_block(layer, filters=filters, s=strides)


def identity_block(X, filters):
    X_shortcut = X
   
    X = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = ReLU()(X)

    X = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = ReLU()(X)

    X = Conv2D(filters=filters * 4, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, X_shortcut])
    X = ReLU()(X)

    return X

# Wanted to use subclass of Layer but for some reason using class based approach led to Out of memory error while training on batch size > 5.
def IdentityBlock(filters):
    return lambda layer: identity_block(layer, filters=filters)
    


# class ProjectionBlock(Layer):
#     def __init__(self, filters, strides):
#         super(ProjectionBlock, self).__init__()
#         self.filters = filters
#         self.strides = strides


#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'filters': self.filters,
#             'strides': self.strides
#         })
#         return config


#     def build(self, input_shape):
#         self.conv_1 = Conv2D(input_shape=input_shape, filters=self.filters, kernel_size=(1, 1), strides=self.strides, padding="same", kernel_initializer=glorot_uniform(seed=0))
#         self.conv_2 = Conv2D(filters=self.filters, kernel_size=(3, 3), padding="same", kernel_initializer=glorot_uniform(seed=0))
#         self.conv_3 = Conv2D(filters=self.filters * 4, kernel_size=(1, 1), kernel_initializer=glorot_uniform(seed=0))
#         self.conv_4 = Conv2D(filters=self.filters * 4, kernel_size=(1, 1), strides=self.strides, kernel_initializer=glorot_uniform(seed=0))
        
#         self.bn_1 = BatchNormalization()
#         self.bn_2 = BatchNormalization()
#         self.bn_3 = BatchNormalization()
#         self.bn_4 = BatchNormalization()

#         self.relu_1 = ReLU()
#         self.relu_2 = ReLU()
#         self.relu_3 = ReLU()

#         self.add = Add()

#         super(ProjectionBlock, self).build(input_shape=input_shape)


#     def call(self, inputs):
#         self.X = inputs
#         self.X2 = inputs
#         self.X = self.relu_1(self.bn_1(self.conv_1(self.X)))
#         self.X = self.relu_2(self.bn_2(self.conv_2(self.X)))
#         self.X = self.bn_3(self.conv_3(self.X))
#         self.X2 = self.bn_4(self.conv_4(self.X2))
#         self.X = self.relu_3(self.add([self.X, self.X2]))
#         return self.X

# class IdentityBlock(Layer):
#     def __init__(self, filters):
#         super(IdentityBlock, self).__init__()
#         self.filters = filters


#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'filters': self.filters,
#         })
#         return config
    

#     def build(self, input_shape): 
#         self.conv_1 = Conv2D(input_shape=input_shape, filters=self.filters, kernel_size=(1, 1), padding="same", kernel_initializer=glorot_uniform(seed=0))
#         self.conv_2 = Conv2D(filters=self.filters, kernel_size=(3, 3), padding="same", kernel_initializer=glorot_uniform(seed=0))
#         self.conv_3 = Conv2D(filters=self.filters * 4, kernel_size=(1, 1), kernel_initializer=glorot_uniform(seed=0))
        
#         self.bn_1 = BatchNormalization()
#         self.bn_2 = BatchNormalization()
#         self.bn_3 = BatchNormalization()
        
#         self.relu_1 = ReLU()
#         self.relu_2 = ReLU()
#         self.relu_3 = ReLU()

#         self.add = Add()

#         super(IdentityBlock, self).build(input_shape=input_shape)
    

#     def call(self, inputs):
#         self.X = inputs
#         self.X2 = inputs
#         self.X = self.relu_1(self.bn_1(self.conv_1(self.X)))
#         self.X = self.relu_2(self.bn_2(self.conv_2(self.X)))
#         self.X = self.bn_3(self.conv_3(self.X))
#         self.X = self.relu_3(self.add([self.X, self.X2]))
#         return self.X