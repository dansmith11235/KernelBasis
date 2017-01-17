import numpy as np 
from numpy import random
from numpy import ndarray
import pylab as pl

#dataset 1 

'''DataClass = np.loadtxt('data_3class.csv')

xTrain = DataClass[0:400:,0:2]
yTrain = DataClass[0:400:,2]

xVal = DataClass[400:600:,0:2]
yVal = DataClass[400:600:,2]

xTest =  DataClass[600:800:,0:2]
yTest = DataClass[600:800:,2]

numClass = 3

numInput = xTrain.shape[1]'''



#dataset  2
'''
name = '1' #make sure it is str(n), n from 1 to 4.
# load data from csv files
train = np.loadtxt('data'+name+'_train.csv')
xTrain = train[:, 0:2]
yTrain = train[:, 2:3]

val = np.loadtxt('data'+name+'_validate.csv')

xVal =  val[:, 0:2]
yVal = val[:, 2:3]

test = np.loadtxt('data'+name+'_test.csv')

xTest =  test[:, 0:2]
yTest = test[:, 2:3]

#Number of unique items in the output 
numClass = 2

#number of inputs 
numInput = xTrain.shape[1]

'''
#dataset 3 

train = np.loadtxt('mnist_digit_'+str(0)+'.csv')
xTrain = train[:200, :]
yTrain = np.ones(len(xTrain))*0

xVal = train[200:300, :]
yVal = np.ones(len(xVal))*0

xTest = train[300:500, :]
yTest = np.ones(len(xTest))*0

for digit in range(1,10):

    train = np.loadtxt('mnist_digit_'+str(digit)+'.csv')
    xTrain = np.append(xTrain,train[:200, :],axis = 0)
    yTrain = np.append(yTrain,np.ones(len(train[:200, :]))*digit)

    xVal = np.append(xVal,train[200:300, :],axis = 0)
    ytemp = np.ones(len(train[200:300, :]))*digit
    yVal = np.append(yVal,ytemp)  

    xTest = np.append(xTest,train[300:500, :], axis = 0)
    yTest = np.append(yTest,ytemp)  


#Number of unique items in the output 
numClass = 10

#number of inputs 
numInput = xTrain.shape[1]

#One Hot Encodes the output 
# X is the output data, N is how many categories there are in the data 
def oneHot(X, N):

    length = X.shape[0]

    X = X.astype(int)

    b = np.zeros((length, N))

    unique = np.unique(X)

    for i in range(length):

        for j in range(unique.shape[0]): 
            if X[i] == unique[j]:
                b[i,j] = 1

    #b[np.arange(length), X] = 1

    return(b)

#X is a vector of N length 
#softmax function
def softMax(X):
    
    scores = np.exp(X)

    return scores / scores.sum()

# Returns the derivative of the softmax 
# X is a vector  of length N.
# Function only returns the diagnoal elements of the 
# derivative matrix since thats all thats used to compute 
# output error. 
def devSoftMax(X):

    return np.multiply(softMax(X),(1 - softMax(X)))


#RELU function 
def reLu(X):

    return np.maximum(0,X)

#derivatrive of the RELU function
def devReLu(X):

    diag = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if X[i] > 0:
            diag[i] = 1
    return(diag)

# Layers is an integer
# NumNodes is an array with how many nodes are in each layer
# The number of elements in NumNodes must equalt the number of layers
def IntalizeWeights(Layers, NumNodes):

    weights = []

    b =  []

    for i in range(1,Layers):

        w = np.zeros((NumNodes[i-1] ,NumNodes[i]))

        w = np.random.normal(0,1/NumNodes[i],size = (NumNodes[i-1] ,NumNodes[i]))

        b.append(np.random.normal(0,1/NumNodes[i], NumNodes[i]))

        weights.append(w)

    return weights, b


# X is the input data 
# W is list of  the Weight matrixs. each element
# of the list is the weight matrix for that layer 
# b is the bias with is a vector 1 X numlayers
# Layers is the number of layers in the network
# numnodes is a vector of lenth Hayers with each
# element being how many nodes are in that layer
# First element of this vecot is the number of inputs
# last element is the number of output units   
def feedForwardNN(X, W, b,Layers,NumNodes):

    for i in range(Layers - 2):

        z = np.reshape(np.dot(np.transpose(W[i]),X),(NumNodes[i+1],1)) + np.reshape(b[i],(NumNodes[i+1],1))
        
        
        X = np.reshape(reLu(z),(NumNodes[i+1],1))
        

    z = np.reshape(np.dot(np.transpose(W[Layers-2]),X),(NumNodes[Layers-2+1],1)) + np.reshape(b[Layers-2],(NumNodes[Layers-2+1],1))

    return softMax(z)


# Back Propagation 
# Takes a matrix X
# Y is a one hot matrix
# Layer is the number of layers including input and output
# NumNodes is the number of nodes for each layer
# first element must be number of inputs
# last element must be number of outputs
# SGD rate is the learning rate
# max_epochs is how many iterations for the SGD
def backProp(X,Y,Layers,NumNodes,SGDRate, max_epochs):


    W, b = IntalizeWeights(Layers,NumNodes)

    epoch = 1

    while epoch < max_epochs:


        # Generate Random number for X row and Y Row 
        rand = random.randint(0,X.shape[0])
        a = X[rand,]
        y = Y[rand,]

        # omegas follow syntax of algorithm in course notes
        delta = []        
        
        #Work through the network saving the Zs and As along the way.
        zList =[]
        aList = [np.reshape(a,(1,a.shape[0]))]
        for i in range(Layers - 1):

            z = np.reshape(np.dot(a,W[i]),(NumNodes[i+1],1)) + np.reshape(b[i],(NumNodes[i+1],1))
            zList.append(z)

            a = np.reshape(reLu(z),(1,NumNodes[i+1]))
            aList.append(a)


        #Get the output error  and save it as one of the omegas
        delta.append(softMax(zList[-1]) - np.reshape(y,(numClass,1)))


        # work backwards through the neural net to get the omegas at each level    
        for i in range(1,Layers - 1):

            DiagF = np.diag(devReLu(zList[-i-1]))

            delta.append(np.dot(np.dot(DiagF,W[-i]),delta[-1]))

        #Count is used to index the omegas
        count = 1
        #Now work backthrough the network updating the weights at each step
        for i in range(0,Layers-1):

            W[i] = W[i] - (SGDRate * np.dot(np.transpose(aList[i]), np.transpose(delta[-count])))
            
            b[i] = np.reshape(b[i],(NumNodes[i + 1],1)) - (SGDRate * delta[-count])

            count = count + 1        

        epoch = epoch + 1      

    return W, b


def SoftmaxtoOneHot(Output):

    OneHot = np.zeros(Output.shape)

    for row in range(Output.shape[0]):


        indices = np.where(Output[row,] == Output[row,].max())

        OneHot[row,indices[0][0]] = 1
    
    return OneHot



def errorRate(Output, Y):

    a = SoftmaxtoOneHot(Output)

    count = 0

    for i in range(Y.shape[0]):
        
        if all(Y[i,] == a[i,]):
            count = count + 1

    return count / Y.shape[0]

def plotDecisionBoundary(X, Y, Weights, bias, values,Layers,NumNodes, title = ""):
    '''
    sv_a is support vector alphas, sv_y is support vector y values,
    sv_x is support vector x values (each entry has dim 2), b is intercept
    values is = -1, 0, 1 preset for margin/widest road
    '''
    # global clf
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
    zz = np.array([x for x in np.c_[xx.ravel(), yy.ravel()]])
    y_predicts = np.zeros(len(zz))
    for i in range(len(zz)):
        TempHolder =  SoftmaxtoOneHot(np.transpose(feedForwardNN(zz[i,],weights,bias,Layers,NumNodes)))
        if TempHolder[0][0]== 1:
            y_predicts[i] = -1
        else:
            y_predicts[i] = 1
    y_predicts = y_predicts
    zz = y_predicts.reshape(xx.shape)
    # zz = np.array([clf.predict(x) for x in c_[xx.ravel(), yy.ravel()]])
    # zz = zz.reshape(xx.shape)
    pl.figure()
    CS = pl.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    # Plot the training points
    pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
    pl.title(title)
    pl.axis('tight')
    return 



print('#######################Training#######################')

#random.seed(20)

yTrainOneHot = oneHot(yTrain,numClass)

# number of items in NumNode must equal the number of layers
# The first element of NumNodes is the number of inputs
# the last element of NumNodes is the number of outputs 
# Layers is the number of layers including input and output


numnodes = np.array([numInput,20, 25 ,numClass]) 
layers = len(numnodes)
weights, bias = backProp(xTrain,yTrainOneHot,Layers = layers, NumNodes= numnodes,SGDRate =.001,max_epochs= 10000)

Results = np.zeros((xVal.shape[0],numClass))
Trains = np.zeros((xTrain.shape[0],numClass))

for i in range(xVal.shape[0]):
    Results[i,] = np.transpose(feedForwardNN(xVal[i,],weights,bias,Layers = layers,NumNodes = numnodes))
for i in range(xTrain.shape[0]):
    Trains[i,] = np.transpose(feedForwardNN(xTrain[i,],weights,bias,Layers = layers,NumNodes = numnodes))

yValOneHot = oneHot(yVal,numClass)
yTrainOneHot = oneHot(yTrain,numClass)


# print (xVal.shape)
print('Training Correct:' + str(errorRate(Trains,yTrainOneHot)))
print('Validation Correct:' + str(errorRate(Results,yValOneHot)))


#Uncomment for plots for dataset 2

'''plotDecisionBoundary(xTrain,yTrain,weights,bias,[0],Layers = layers,NumNodes = numnodes)

pl.show()
'''