from numpy import loadtxt
from keras.models import model_from_json
dataset=loadtxt('diabetes.csv',delimiter=',')
x=dataset[:,0:8]
y=dataset[:,8]

#reading file from local drive
json_file=open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()
model=model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("loaded model from disk")

#making predictions of future values
predict=model.predict_classes(x)
for i in range(100,200):
    print('%s=> %d(expected %d)' % (x[i].tolist(), predict[i], y[i]))
