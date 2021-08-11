import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import os
import math
from sklearn.metrics import confusion_matrix


  
'''Bag Of Visual Words '''
'''Image Data'''
classes={"botanical garden":0,"bus interior":1,"elevator shaft":2}
#Training data of botanical garden
train_class=[]
train_images=[]

validation_class=[]
validation_images=[]

for filename in os.listdir("Dataset/train/botanical_garden"):
    img = Image.open("Dataset/train/botanical_garden/"+filename) 
    train_images.append(np.asarray(img))
    train_class.append(classes["botanical garden"])

#validation data for botanical garden
for i in range(40,50):
    validation_images.append(train_images.pop())
    validation_class.append(train_class.pop())


#training data for bus interior
for filename in os.listdir("Dataset/train/bus_interior"):
    img = Image.open("Dataset/train/bus_interior/"+filename) 
    train_images.append(np.asarray(img))
    train_class.append(classes["bus interior"])
    
#validation data for bus interior
for i in range(40,50):
    validation_images.append(train_images.pop())
    validation_class.append(train_class.pop())
    

#training data for elevator shaft
for filename in os.listdir("Dataset/train/elevator_shaft"):
    img = Image.open("Dataset/train/elevator_shaft/"+filename) 
    train_images.append(np.asarray(img))
    train_class.append(classes["elevator shaft"])
    
#validation data for elevator shaft
for i in range(40,50):
    validation_images.append(train_images.pop())
    validation_class.append(train_class.pop())

#print(len(train_class),len(train_images),train_images[115].shape,train_images[119].shape)
#print(validation_class)

test_class=[]
test_images=[]

for filename in os.listdir("Dataset/test/botanical_garden"):
    img = Image.open("Dataset/test/botanical_garden/"+filename) 
    test_images.append(np.asarray(img))
    test_class.append(classes["botanical garden"])
    

for filename in os.listdir("Dataset/test/bus_interior"):
    img = Image.open("Dataset/test/bus_interior/"+filename) 
    test_images.append(np.asarray(img))
    test_class.append(classes["bus interior"])


for filename in os.listdir("Dataset/test/elevator_shaft"):
    img = Image.open("Dataset/test/elevator_shaft/"+filename) 
    test_images.append(np.asarray(img))
    test_class.append(classes["elevator shaft"])
    

#function to make a feature vector of 24 dimensional
def feature_vector(img):
    
    feature=np.zeros((24,1))#this array will store the feature vector of each image
    #print(img.shape)
    width=img.shape[1]#width of image
    height=img.shape[0]#height of image
    
    if height%32!=0 :
        extra=height%32#no of extra pixels required to add to the hieght
        extra_rows=img[:32-extra,:,:]#we will be adding firs n rows to make it equal to patches
        img=np.vstack((img,extra_rows))
        
    if width%32!=0 :
        extra=width%32#no of extra pixels required to add to the hieght
        extra_columns=img[:,:32-extra,:]#we will be adding firs n rows to make it equal to patches
        img=np.hstack((img,extra_columns))       
    
    #travelling in every patch of 32x32
    start_height=0
    end_height=start_height +32
    
    start_width=0
    end_width=start_width + 32
    #print(img.shape)
    while end_height<=height :
        patch=img[start_height:end_height,start_width:end_width,:]
    
        color_hist=np.zeros((1,1))#this will store a 24 dimensional color histogram for each patch will be assembled to anothe array
        #print(patch.shape)
        #travelling in each color channel of the patch
        for color in range(patch.shape[2]):
            color_patch=np.array(patch[:,:,color])

            hist=np.zeros((8,1))#to calculate the color histogram of a particular channel of 8 bins
            
            for i in range(32):
                for j in range(32):
                    
                    x=int(color_patch[i,j]//32)
                    hist[x,0]+=1.0#adding the count of pixels belonging to each bin
            
            color_hist=np.vstack((color_hist,hist))
        
        color_hist=np.delete(color_hist,0,0)#deleting first row of 0's from color_hist
        
        #color_hist is a 24 dimensional vector formed by stacking 3 8-dimensional vectors for each color channel
        feature=np.hstack((feature,color_hist))
        
        
        if end_width>=img.shape[1]:
            start_width=0
            end_width=start_width+32
            start_height=end_height
            end_height+=32
        else:
            start_width=end_width
            end_width+=32
    
    feature=np.delete(feature,0,1)#deleting first column of 0's from feature
            
    return feature


train_features=np.zeros((24,1))#matrix to store feature vectors of patches of training images
for image in train_images:
    train_features=np.hstack((train_features,feature_vector(image)))
    
train_features=np.delete(train_features,0,1)
print(train_features.shape)

validation_features=np.zeros((24,1))#matrix to store feature vectors of patches of training images
for image in validation_images:
    validation_features=np.hstack((validation_features,feature_vector(image)))
    
validation_features=np.delete(validation_features,0,1)
print(validation_features.shape)

test_features=np.zeros((24,1))#matrix to store feature vectors of patches of training images
for image in test_images:
    test_features=np.hstack((test_features,feature_vector(image)))
    
test_features=np.delete(test_features,0,1)
print(test_features.shape)

#applying KMeans
kmeans = KMeans(n_clusters=32)
kmeans.fit(train_features.T)
centers=kmeans.cluster_centers_
print(centers)


bovw_rep={}#dictionary to store bag of visual words representation

start=0#starting column for the image
end=0#Ending column for image's patch
for i in range(len(train_images)):
  height=train_images[i].shape[0]
  width=train_images[i].shape[1]
  no_of_patches=math.ceil(width/32)*math.ceil(height/32)#to find the no of patches belonging to this image
  end+=no_of_patches
  patches_image=train_features[:,start:end]#to get the patches of this image

  image_bovw=np.zeros((32,1))#vector to represent bovw of this image

  for j in range(patches_image.shape[1]):
    patch=patches_image[:,j]
    pred_cluster=kmeans.predict(patch.reshape(1,-1))[0]#predicted cluster for this class
    image_bovw[pred_cluster,0]+=1.0
  
  image_bovw=image_bovw/no_of_patches#dividing with no of patches
  bovw_rep[i]=image_bovw#assigning image bovw representation in a dictionary

  start=end

bovw_rep_val={}#dictionary to store bag of visual words representation

start=0#starting column for the image
end=0#Ending column for image's patch
for i in range(len(validation_images)):
  height=validation_images[i].shape[0]
  width=validation_images[i].shape[1]
  no_of_patches=math.ceil(width/32)*math.ceil(height/32)#to find the no of patches belonging to this image
  end+=no_of_patches
  patches_image=validation_features[:,start:end]#to get the patches of this image

  image_bovw=np.zeros((32,1))#vector to represent bovw of this image

  for j in range(patches_image.shape[1]):
    patch=patches_image[:,j]
    pred_cluster=kmeans.predict(patch.reshape(1,-1))[0]#predicted cluster for this class
    image_bovw[pred_cluster,0]+=1.0
  
  image_bovw=image_bovw/no_of_patches#dividing with no of patches
  bovw_rep_val[i]=image_bovw#assigning image bovw representation in a dictionary

  start=end


bovw_rep_test={}#dictionary to store bag of visual words representation

start=0#starting column for the image
end=0#Ending column for image's patch
for i in range(len(test_images)):
  height=test_images[i].shape[0]
  width=test_images[i].shape[1]
  no_of_patches=math.ceil(width/32)*math.ceil(height/32)#to find the no of patches belonging to this image
  end+=no_of_patches
  patches_image=test_features[:,start:end]#to get the patches of this image

  image_bovw=np.zeros((32,1))#vector to represent bovw of this image

  for j in range(patches_image.shape[1]):
    patch=patches_image[:,j]
    pred_cluster=kmeans.predict(patch.reshape(1,-1))[0]#predicted cluster for this class
    image_bovw[pred_cluster,0]+=1.0
  
  image_bovw=image_bovw/no_of_patches#dividing with no of patches
  bovw_rep_test[i]=image_bovw#assigning image bovw representation in a dictionary

  start=end

df=pd.DataFrame(data=train_features.T,columns=list(range(1,25)))
df.reset_index(drop=True, inplace=True)


#Making a dataframe for representation of BOVW of training images
bovw_dataframe=np.zeros((1,32))
bovw_dataframe[0,:]=list(range(1,33))

for i in bovw_rep.values():
  bovw_dataframe=np.vstack((bovw_dataframe,i.T))

bovw_dataframe=np.delete(bovw_dataframe,0,0)
df1=pd.DataFrame(bovw_dataframe,columns=list(range(1,33)))
df1["Class"]=train_class

df1.reset_index(drop=True, inplace=True)

Y=np.array(pd.get_dummies(df1.Class)).T
Class=np.array([df1.Class])

df1=df1.drop(columns=["Class"])
X=np.array(df1)
X=X.T


#Making a dataframe for representation of BOVW of validation images
bovw_dataframe_val=np.zeros((1,32))
bovw_dataframe_val[0,:]=list(range(1,33))

for i in bovw_rep_val.values():
  bovw_dataframe_val=np.vstack((bovw_dataframe_val,i.T))

bovw_dataframe_val=np.delete(bovw_dataframe_val,0,0)
df2=pd.DataFrame(bovw_dataframe_val,columns=list(range(1,33)))
df2["Class"]=validation_class

df2.reset_index(drop=True, inplace=True) 
Y_val=np.array(pd.get_dummies(df2.Class)).T
Class_val=np.array([df2.Class])

df2=df2.drop(columns=["Class"])
X_val=np.array(df2)
X_val=X_val.T

#Making a dataframe for representation of BOVW of test images
bovw_dataframe_test=np.zeros((1,32))
bovw_dataframe_test[0,:]=list(range(1,33))

for i in bovw_rep_test.values():
  bovw_dataframe_test=np.vstack((bovw_dataframe_test,i.T))

bovw_dataframe_test=np.delete(bovw_dataframe_test,0,0)
df3=pd.DataFrame(bovw_dataframe_test,columns=list(range(1,33)))
df3["Class"]=test_class
df3.reset_index(drop=True, inplace=True) 

Y_test=np.array(pd.get_dummies(df3.Class)).T
Class_test=np.array([df3.Class])

df3=df3.drop(columns=["Class"])
X_test=np.array(df3)
X_test=X_test.T




'''MLFFNN for Image Data'''
def activation(z,activate):
    if activate=="tanh":
        return np.tanh(z)
    elif activate == "relu" :
        return z*(z>0)
    else:
        return np.exp(z)/(1+np.exp(z))


def output_image(X,parameters,Y,Class_image):
  Z1=np.dot(parameters["W1"],X)+parameters["b1"]
  A1=activation(Z1,"tanh")
  Z2=np.dot(parameters["W2"],A1)+parameters["b2"]
  A2=activation(Z2,"sigmoid")
  Z3=np.dot(parameters["W3"],A2)+parameters["b3"]
  A3=activation(Z3,"sigmoid")
    
  Y_pred=np.argmax(A3,axis=0).reshape(1,-1)

  #Error
  m=X.shape[1]
  E_mat=((Y-A3)**2)/(2*m)
  missclass=np.count_nonzero(Class_image-Y_pred)
  Error=np.sum(E_mat)
    
  return Y_pred, Error, missclass


def weight_inialization_image(n_x,n_h1,n_h2,n_o):#function to inialize weight parameters
    '''
    n_x:dimension of input data
    n_h1:dimension of hidden layer 1
    n_h2:dimension of hidden layer 2
    n_o: dimension of output layer
    '''
    """"
    W1,B1:Weight matrix of 1st layer
    W2,B2:Weight matrix of second layer
    W3,B3:Weight matrix of output layer
    """
    W1=np.random.randn(n_h1,n_x)*0.09
    B1=np.zeros((n_h1,1))
    W2=np.random.randn(n_h2,n_h1)*0.09
    B2=np.zeros((n_h2,1))
    W3=np.random.randn(n_o,n_h2)*0.09
    B3=np.zeros((n_o,1))
    
    assert(W1.shape==(n_h1,n_x))
    assert(B1.shape==(n_h1,1))
    assert(W2.shape==(n_h2,n_h1))
    assert(B2.shape==(n_h2,1))
    assert(W3.shape==(n_o,n_h2))
    assert(B3.shape==(n_o,1))
    
    return {"W1" : W1,
            "b1" : B1,
            "W2" : W2,
            "b2" : B2,
            "W3" : W3,
            "b3" : B3
           } 
    
    
def forward_back_propagation_images(X,n_h1,n_h2,n_o,Y,Class,epochs,learning_rate):
  #here Y is One-hot vector of the class and class is the vector from the dataframe
    
    n_x=X.shape[0]#dimension of X input
    parameters=weight_inialization_image(n_x,n_h1,n_h2,n_o)#Initializing weights
    m=X.shape[1]#number of examples of input vector
    Error_list=[]#list to store average error
    misclassification_list=[]
    

    for i in range(epochs):
        #learning_rate=learning_rate/(i+1)
        #performing forward propagation
        #Layer1
        Z1=np.dot(parameters["W1"],X) + parameters["b1"]
        A1=activation(Z1,"tanh")
        
        #Layer2
        Z2=np.dot(parameters["W2"],A1) + parameters["b2"]
        A2=activation(Z2,"sigmoid")
        
        #output layer
        Z3=np.dot(parameters["W3"],A2) + parameters["b3"]
        A3=activation(Z3,"sigmoid")
        Y_pred=np.argmax(A3,axis=0).reshape(1,-1)

        #Error
        E_mat=((Y-A3)**2)/(2*m)
        missclass=np.count_nonzero(Class-Y_pred)
        misclassification_list.append(missclass)
        
        #performing back propagation
        #output Layer
        dW3=(-1/m)*np.dot((Y-A3)*A3*(1-A3),A2.T)
        db3=(-1/m)*np.sum((Y-A3)*A3*(1-A3),axis=1).reshape((n_o,1))
        parameters["W3"]=parameters["W3"]-learning_rate*dW3
        parameters["b3"]=parameters["b3"]-learning_rate*db3
        
        #Layer2
        dW2=(-1/m)*np.dot(np.dot(parameters["W3"].T,(Y-A3)*A3*(1-A3))*A2*(1-A2),A1.T)
        db2=(-1/m)*np.sum(np.dot(parameters["W3"].T,(Y-A3)*A3*(1-A3))*A2*(1-A2),axis=1).reshape((n_h2,1))
        parameters["W2"]=parameters["W2"]-learning_rate*dW2
        parameters["b2"]=parameters["b2"]-learning_rate*db2
        
        #Layer1
        dW1=(-1/m)*np.dot(np.dot(parameters["W2"].T,(np.dot(parameters["W3"].T,(Y-A3)*A3*(1-A3))*A2*(1-A2)))*(1-A1**2),X.T)
        db1=(-1/m)*np.sum(np.dot(parameters["W2"].T,(np.dot(parameters["W3"].T,(Y-A3)*A3*(1-A3))*A2*(1-A2)))*(1-A1**2),axis=1).reshape((n_h1,1))
        parameters["W1"]=parameters["W1"]-learning_rate*dW1
        parameters["b1"]=parameters["b1"]-learning_rate*db1
        #print(db1.shape)
        Error_list.append(np.sum(E_mat))
        
        
    return parameters,Error_list,misclassification_list 


epoch=35000

#np.random.shuffle(X)
parameters_image, Error_image, misclass=forward_back_propagation_images(X,64,32,3,Y,Class,epoch,0.09)

print("Error on Train Data", Error_image[-1])
print("MisClassification Count on Train Data", misclass[-1])
print("Accuracy =",(1-(misclass[-1])/X.shape[1]))

plt.plot(list(range(epoch)),Error_image)
plt.xlabel("Epochs",fontsize=18)
plt.ylabel("Error",fontsize=18)
plt.title("Epoch Vs Error",fontsize=18)
plt.show()

plt.plot(list(range(epoch)),misclass)
plt.xlabel("Epochs",fontsize=18)
plt.ylabel("Misclassification Count",fontsize=18)
plt.title("Epoch Vs Misclassification Count",fontsize=18)
plt.show()



#Feeding Validation Data
Class_pred_val, Error_image_val, misclass_val=output_image(X_val,parameters_image,Y_val,Class_val)

print("Error on Validation Data", Error_image_val)
print("MisClassification Count on Validation Data", misclass_val)

print(confusion_matrix(Class_val.reshape(-1,),Class_pred_val.reshape(-1,)))
print("Accuracy =",(1-(misclass_val)/X_val.shape[1]))


#Feeding Test Data
Class_pred_test, Error_image_test, misclass_test=output_image(X_test,parameters_image,Y_test,Class_test)

print("Error on Test Data", Error_image_test)
print("MisClassification Count on test Data", misclass_test)

print(confusion_matrix(Class_test.reshape(-1,),Class_pred_test.reshape(-1,)))
print("Accuracy =",(1-(misclass_test)/X_test.shape[1]))



