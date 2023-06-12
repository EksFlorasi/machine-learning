from keras.models import load_model

H5_PATH = "./models/flora/h5/inception-model-fruitsflowers.h5"
PB_PATH = H5_PATH[:-3].replace("h5", "pb")

model = load_model(H5_PATH)
model.save(PB_PATH)
