'''
parameters passed in terms of columns
1. pregnancy
2.plasma glucose concentration
3.diastolic blood pressure
4.triceps skin fold
5.2-hour serum insulin
6.Bmi
7.diabetes pedigree
8.Age
9.class variable(0 or 1)
'''
from numpy import loadtxt # used for loading dataset(csv files)
from keras.models import Sequential# it will create an empty stack of layers-input ,output ,hidden layers and so on
from keras.layers import Dense# layer fomat - convolution ,dense , recurrent and so on
from keras.models import model_from_json# loading , saving the model
dataset = loadtxt('diabetes.csv',delimiter=',')# loading csv filr in ld
x=dataset[:,0:8] # reading dataset(all rows,8 columns)
y=dataset[:,8]# final value in class format
# creating model
model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))# creating a input  layer of 12 - number of neurons ,reading 8 columns,activation -relu
model.add(Dense(8,activation='relu'))# creating hidden layer with 8 neurons , activation -relu format
model.add(Dense(1,activation='sigmoid'))# creating output layer which has 1 neuron ,activation is sigmoid format (probablility)
#model.summary()# creates the model or shows structure of the model

#computation /working of model(optimisinsation)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])# specifies the loss , optimisation ,factors to increase effiency like accuracy
model.fit(x,y,epochs=200,batch_size=10)#to imporve accuracy update epochs in range of 100
_,accuracy=model.evaluate(x,y)# running the model
print('Accuracy:%.2f'%(accuracy*100))# accuracy in multiples of 100

#saving model to local drive
model_json=model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("saved model to disk")
