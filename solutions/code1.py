# Using the create_dataset function given beforehand, reshape test and train into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))