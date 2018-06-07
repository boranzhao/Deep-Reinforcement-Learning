from keras import optimizers

RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
print(RMSprop.rho)