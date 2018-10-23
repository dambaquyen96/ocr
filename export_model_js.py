import keras
from keras.applications.mobilenet import MobileNet
from keras.models import model_from_json
import tensorflowjs as tfjs

N_CLASSES = 3095
IMG_SIZE = 100

model = model_from_json(open('models/model.json', 'r').read())
model.load_weights("models/weights-improvement-26-0.00.hdf5")

tfjs.converters.save_keras_model(model, "model_js")