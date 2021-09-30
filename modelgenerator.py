from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets.fashion_mnist as fashion
from tensorflow.keras.utils import to_categorical

# Unpacking the train data
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# This transforms the dataset into a one hot encoding
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

def get_label(one_hot_encoding):
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return labels[one_hot_encoding.argmax()]

# Definition of the model
cnn_model = Sequential([
    layers.Conv2D(128, kernel_size=(6,6), input_shape=(28,28, 1), activation='relu'), # (X,Y,Pixel(colour=3,greyscale=1)) can also take padding
    layers.MaxPooling2D(pool_size=(3,3)), #you can pass stride.
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dropout(.01),
    layers.Dense(500, activation='relu'),
    layers.Dense(250, activation='relu'),
    layers.Dropout(.01),
    layers.Dense(10, activation='softmax')
])

# explain the parameters
cnn_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
pre_loss, pre_acc = cnn_model.evaluate(x_test.reshape([-1, 28, 28, 1]), y_test)
print("Untrained model, accuracy: {:5.2f}%".format(100 * pre_acc))
print(pre_loss)
cnn_model.fit(x_train.reshape([-1, 28, 28, 1]), y_train, epochs=30)
post_loss, post_acc = cnn_model.evaluate(x_test.reshape([-1, 28, 28, 1]), y_test)
print("Trained model, accuracy: {:5.2f}%".format(100 * post_acc))
print(post_loss)
cnn_model.save("MODELS\\latest_model")
