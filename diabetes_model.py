from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

        #load dataset in sklearn
ld=load_diabetes()
ld.data
ld.target


x=ld.data
y=ld.target
y
x[0]

        #Data Preprocessing in dataset
ss=StandardScaler()
x1=ss.fit_transform(x)
x1[0]

        #data set spilt in test and train data set
xtrain,xtest,ytrain,ytest=train_test_split(x1,y,test_size=0.2)

        # Earlystopping to select number of  echops
es=EarlyStopping(monitor='loss',patience=3,min_delta=0.0101,mode='auto')

        #create model 

        #create model object
model=Sequential()
        
        #this is input layer
model.add(Dense(12,activation='relu',input_dim=10))
        
        #this is hidden layers
model.add(Dense(12,activation='relu'))
model.add(Dense(5,activation='relu'))
        
        #this is output layer
model.add(Dense(1,activation='relu'))
model.compile(optimizer='adam',loss='mean_squared_error')

        #Now train the model or fit
hs=model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=600,callbacks=[es])#callbacks for#callbacks for

        #Now train  predict model
ypred=model.predict(xtest)

#cross validetion
r2_score(ytest,ypred)


hs.history.keys()
plt.plot(hs.history['val_loss'])
plt.plot(hs.history['loss'])
