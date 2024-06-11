from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt

dataset = loadtxt('diabetes_dataset.csv',delimiter = ',')

x = dataset[:,0:8]
y = dataset[:,8]

print("Value of X are : " , x , " And value of Y are : " , y)

model = Sequential()

model.add(Dense(12,input_dim = 8,activation = "relu")) 
model.add(Dense(8, activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=500, batch_size=100)
predictions = model.predict_classes(x)
for i in range(5):
    print(f'{x[i].tolist()} => {predictions[i]} expected {y[i]}')