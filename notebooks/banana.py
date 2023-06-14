import io
import base64
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array

# models
FAUNA_H5 = "../models/fauna/h5/fauna-95-89.h5"
FLORA_H5 = "../models/flora/h5/inception-model-fruits360do.h5"
MODEL_H5 = FLORA_H5
# labels
FAUNA_LABEL_PATH = "../models/fauna/fauna_labels.txt"
FLORA_LABEL_PATH = "../models/flora/flora_labels.txt"
LABEL_PATH = FAUNA_LABEL_PATH
# image specs
IMAGE_PATH = "image_tests/flora/banana-2.jpeg"
TARGET_SIZE = (100, 100)

with open(IMAGE_PATH, "rb") as image_file:
    image_b64 = base64.b64encode(image_file.read())

data = base64.b64decode(image_b64)
image = io.BytesIO(data)

img_b64 = load_img(image, target_size=TARGET_SIZE)
x_b64 = img_to_array(img_b64)
x_b64 /= 255.0
x_b64 = np.expand_dims(x_b64, axis=0)

img_raw = load_img(IMAGE_PATH, target_size=TARGET_SIZE)
x_raw = img_to_array(img_raw)
x_raw /= 225.0
x_raw = np.expand_dims(x_raw, axis=0)

#with open(LABEL_PATH) as f:
#    LABEL = f.read().splitlines()
LABEL = ['Apple Braeburn', 'Apricot', 'Avocado', 'Banana', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Carambula', 'Cauliflower', 'Cherry 1', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Cucumber Ripe', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grapefruit Pink', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Onion Red', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepino', 'Pepper Green', 'Pineapple', 'Pitahaya Red', 'Plum', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Watermelon']
model = load_model(MODEL_H5)

# B64
pred_b64 = model.predict(x_b64)
# Raw
pred_raw = model.predict(x_raw)

result_b64 = LABEL[pred_b64.argmax(axis=1)[0]]
result_raw = LABEL[pred_raw.argmax(axis=1)[0]]
print(f"Base64: {result_b64}")
print(f"Raw: {result_raw}")

tuple_b64 = tuple(zip(LABEL, pred_b64.ravel()))
tuple_b64 = sorted(tuple_b64, key=lambda x: x[1], reverse=True)
print("Base64: ")
print(tuple_b64[:5])

tuple_raw = tuple(zip(LABEL, pred_raw.ravel()))
tuple_raw = sorted(tuple_raw, key=lambda x: x[1], reverse=True)
print("Raw: ")
print(tuple_raw[:5])