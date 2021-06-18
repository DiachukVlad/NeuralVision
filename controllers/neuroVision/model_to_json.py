import keras

model = keras.models.load_model("model.h5")

json = model.to_json()

print(json)