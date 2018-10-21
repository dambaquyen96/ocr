import keras
from keras.applications.mobilenet import MobileNet
import tensorflowjs as tfjs

N_CLASSES = 3095
IMG_SIZE = 100

model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=True, classes=N_CLASSES, weights=None)
model.load_weights("models/weights-improvement-26-0.00.hdf5")

tfjs.converters.save_keras_model(model, "model_js")