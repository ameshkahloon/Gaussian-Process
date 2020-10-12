# Import libraries into the code
import time
import math
import numpy as np  # Used for matrix operations
import matplotlib.pyplot as plt  # Used for plotting
from matplotlib import cm  # Used for plotting the training and test data
from mpl_toolkits.mplot3d import Axes3D  # Use for plotting in higher dimensions
from sklearn.metrics import mean_squared_error  # Used to calculate the root mean sqaure error (RMSE)
from decimal import Decimal  # Used to display scientific notation


########################################################################################################################
# By: Amesh Kahloon NSERC Position at Queen's Univeristy with Professor Tucker Carrington 
# Date: August 28 2020
# This program is supplemented with the report, Gaussian processes with applications towards computational chemistry
########################################################################################################################


# This program runs Gaussian processes in ten dimensions, four dimensions, two dimensions and one dimension
# The hyper parameters we can modify are the length parameter and the variance parameter for the exponential squared kernel
# A summation method is used to reduce the time complexity and memory usage in the compuations of the Kronecker product

# In Gaussian processes the data we can modify are:
# 1. Training data (where we know the output of the function)
# 2. Test data (Where we want to know the output of the function)
# 3. Length parameter (Determines the influence each of the vectors in the training data and test data have on each other)
# 4. Variance parameter (Determines the spread of the distribution from the mean) (in the report this was set to 1)
# 5. Regularization constant (Reduces the condition number of the covariance matrices)

# We define functions which allow us to modify the above data (TrainingPoints, TestPoints and HyperParameters)
# TrainingData: Sets the training data in each coordinate of the Gaussian process
# TestData: Sets the test data in each coordinate of the Gaussian process
# HyperParameters: Set the length parameter, variance parameter and the regularization constants
# The rest of the code are functions which obtain these parameters and run the entire Gaussian process

# The Gaussian process is initiated using the main() function


########################################################################################################################
# Covariance functions
########################################################################################################################


# Calculates the entire covariance matrix using the exponential squared kernel with the Euclidean norm
def Covariance(X1, X2, length, variance):
    # X1: The first set of inputs for the covariance matrix
    # X2: The second set of inputs for the covariance matrix
    # length: The length parameter for the exponential squared kernel
    # variance: The variance parameter for the exponential squared kernel

    # Calculate the numerator in the exponential of the kernel
    norm_squared = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)

    # Return a X1 by X2 matrix
    return variance ** 2 * np.exp(-0.5 / length ** 2 * norm_squared)


# Calculates the covariance between two vectors in R^2 or greater using the exponential squared kernel with the Euclidean norm
def KernelVector(X1, X2, length, variance):
    # X1: The first vector to calculate the covariance
    # X2: The second vector to calculate the covariance4
    # length: The length parameter for the exponential squared kernel
    # variance: The variance parameter for the exponential squared kernel

    # Returns the covariance of two vectors
    return variance ** 2 * math.exp(-0.5 / length ** 2 * np.linalg.norm(np.subtract(X1, X2), ord=2) ** 2)


# Calculates the covariance between two scalar values using the exponential squared kernel
def KernelSingular(X1, X2, length, variance):
    # X1: The first input
    # X2: The second input
    # length: The length parameter for the exponential squared kernel
    # variance: The variance parameter for the exponential squared kernel

     # Returns the covariance of two scalars
    return variance ** 2 * math.exp(-0.5 / length ** 2 * (X1 - X2) ** 2)


########################################################################################################################
# 1-D Gaussian Process
########################################################################################################################


# Plots the Gaussian process in the 1-D case
def Plot_gp_1D(mu, cov, domain_test, domain_train, codomain_train, samples):
    # mu: The mean of the multinormal distribution
    # cov: The covariance of the multinormal distribution
    # domain_test: The test data
    # domain_train: The training data
    # codomain_train: The output of the training data
    # samples: The samples from the multinormal distribution
     
    X = domain_test.ravel()
    mu = mu.ravel()

    # Adds a shaded region of uncertainty of a factor of 1.9 (only used in the 1D Gaussian process)
    uncertainty = 1.9 * np.sqrt(np.diag(cov))

    # Legend for the sample plots and inserting the uncertainty to the plot
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    if domain_train is not None:
        plt.plot(domain_train, codomain_train, 'rx')
    plt.legend()
    plt.show()

    return


# The posterior predictive of the 1-D case calculates the mean and covariance of multinormal distribution conditioned on the training data
# The same computations in the line above are applied towards higher dimension Gaussian processes
def Posterior_predictive1D(domain_test, domain_train, codomain_train, length, variance, reg_constant):
    # domain_test: The test data
    # domain_train: The training data
    # codomain_train: The output of the training data
    # length: The length parameter of the exponential squared kernel
    # variance: The variance parameter of the exponential squared kernel
    # reg_constant: The regularization constant for the covariance matrix

    # Calculate the covariance matrix (K)
    K = Covariance(X1=domain_train, X2=domain_train, length=length, variance=variance)

    # Calculate the condition number of the covariance matrix before regularization
    print('Condition number of the covariance matrix before regularization')
    print('%.4E' % Decimal(np.linalg.cond(K)))

    # Regularization section
    K = K + reg_constant * np.identity(len(K))

    # Calculate the condition number after regularization
    print('Condition number of the covariance matrix after regularization')
    print('%.4E' % Decimal(np.linalg.cond(K)))

    # Calculate the matrix K* (takes in the training data and the test data)
    K_s = Covariance(X1=domain_train, X2=domain_test, length=length, variance=variance)

    # Calculate the matrix K** (takes in the test data as both parameters)
    K_ss = Covariance(X1=domain_test, X2=domain_test, length=length, variance=variance)

    # Invert the covariance matrix to calculate the mean
    K_inv = np.linalg.inv(K)

    # Calculate the mean of the multinormal distribution
    mu = K_s.T.dot(K_inv.dot(codomain_train))

    # Calculate the covariance of the multinormal distribution
    cov = K_ss - K_s.T.dot(K_inv.dot(K_s))

    # Return the covariance and the mean of the multinormal distribution
    return mu, cov


# Output of the training data and test data
def Output1D(domain):
    # domain: The training data or test data that will be passed into the formula

    # sine function
    sine = np.sin(domain)

    # parabola
    parabola = np.power(domain, 2)

    # 1-D Morse potential function
    D_e = 37255
    a = 1.8677
    r_e = 1.275
    morse_potential = D_e*np.power(1 - np.exp(-a*(domain-r_e)), 2)

    return morse_potential


# Returns the length parameter and variance for the kernel, the number of samples from the multinormal distribution and the regularization constant
def HyperParameters1D():
    length = 0.2
    variance = 1
    size = 0
    reg_constant = 0

    return length, variance, size, reg_constant


# The test data for the domain
def TestPoints1D():
    test_data = np.random.normal(loc=1.275, scale=0.1, size=400)

    return test_data


# The training data for the domain
def TrainingPoints1D():
    X = np.arange(0.975, 1.575, 0.06)

    return X


# The Gaussian process for a 1-D problem
def GaussianProcess1D():
    # Test data for the Gaussian process
    domain_test = TestPoints1D()

    # Training data for the Gaussian process
    domain_train = TrainingPoints1D()

    # Output of the training data
    codomain_train = Output1D(domain=domain_train)

    # Obtain the length and variance of the exponential squared kernel, number of samples from the mulitnormal distribution and regularization constant
    length, variance, size, reg_constant = HyperParameters1D()

    # Obtain the mean and covariance of the multinormal distribution
    mu, cov = Posterior_predictive1D(domain_test=domain_test.reshape(-1, 1), domain_train=domain_train.reshape(-1, 1),
                                     codomain_train=codomain_train, length=length, variance=variance,
                                     reg_constant=reg_constant)

    # Generate samples of the multinormal distribution from the calculated mean and covariance
    # mean is the mean of the multinormal distribution we obtained from above
    # cov is the covariance of the multinormal distribution we obtained from above
    # size is the how many times we pick from the sample we defined in the HyperParameters1D() function
    samples = np.random.multivariate_normal(mean=mu.ravel(), cov=cov, size=size)

    # Calculate the output of the training data
    predicted = Output1D(domain=domain_test)

    # Calculate the RMSE of the mean (observed data) and the training data (predicted data)
    rmse = math.sqrt(mean_squared_error(mu, predicted))
    print('The RMSE of the mean: ')
    print(rmse)

    # Plot the mean of the multinormal distribution, training data and samples form the multinormal distribution
    Plot_gp_1D(mu=mu, cov=cov, domain_test=domain_test, domain_train=domain_train, codomain_train=codomain_train,
               samples=samples)

    return


########################################################################################################################
# 2-D Gaussian Process
########################################################################################################################


# Plots the Gaussian process in the 2-D case
def Plot_gp_2D(gx, gy, mu, domain_test, predicted, domain_train, codomain_train, title, i):
    # gx: The x-axis
    # gy: The y-axis
    # mu: The mean of the multinormal distribution
    # domain_test: The test data
    # predicted: The output of the test data
    # domain_train: The set of training data
    # codomain_title: The output of the training data
    # title: Title of the plot
    # i: Constant required for the plot

    # Create the plot, no sample data is plotted unlike the 1-D case
    plt.figure(figsize=(14, 7))
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')

    # Training data are red and blue dots in the plot
    # Red dots: high output values. Blue dots: low output values
    # Test data are purple and green dots in the plot
    # Green dots: high output values. Purple dots: low output values
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(domain_train[:, 0], domain_train[:, 1], codomain_train, c=codomain_train, cmap=cm.coolwarm)
    ax.scatter(domain_test[:, 0], domain_test[:, 1], predicted, c=predicted, cmap=cm.PRGn)
    ax.set_title(title)
    plt.show()

    return


# Calculate the mean without calculating the K* matrix directly
def Mean2D(domain_test, domain_train, K_Y, length, variance):
    # domain_test: The test data
    # domain_train: The training data
    # length: The length parameter for the exponential squared kernel
    # variance: The variance for the exponential squared kernel
    # K_Y: K^(-1)*codomain_train tensor obtained from the summation method

    mu = []

    summation = 0
    for i in range(len(domain_test)):
        for j in range(len(domain_train)):
            summation += KernelVector(X1=domain_test[i], X2=domain_train[j], length=length, variance=variance) * K_Y[j]
        mu.append(summation)
        summation = 0

    # convert the mean from a list to an array
    mu_array = np.array(mu)

    return mu_array

    
# Generates a column vector from the tensor K^(-1)*codomain_train
def ColumnGenerator2D(K_InvY_unshaped):
    # K_InvY_unshaped: The tensor K^(-1)*Y obtained from the summation method

    col = np.zeros((len(K_InvY_unshaped)*len(K_InvY_unshaped[0]), 1))

    counter = 0
    for i in range(    len(K_InvY_unshaped)):
        for j in range(len(K_InvY_unshaped[0])):
            col[counter][0] = K_InvY_unshaped[j][i]
            counter = counter + 1

    return col


# Calculate K^(-1)*codomain_train using the summation method
def Summation2D(K1, K2, codomain_2D_mapper):
    # K1: The covariance matrix of only the w-coordinates
    # K2: The covariance matrix of only the x-coordinates
    # Y_2D_mapper: The output of the training data as a tensor

    K1_inverse = np.linalg.inv(K1)  # We get K_1 inverse
    K2_inverse = np.linalg.inv(K2)  # We get K_2 inverse
    
    Z = np.zeros((len(K2_inverse), len(codomain_2D_mapper[0])))  
    K_inverse_Y = np.zeros((len(K1_inverse), len(Z[0])))

    # Calculate the Z intermediate matrix
    for i in range(len(K2_inverse)):
        for j in range(len(codomain_2D_mapper[0])):
            for k in range(len(codomain_2D_mapper)):
                Z[i][j] += K2_inverse[i][k] * codomain_2D_mapper[k][j]

    # We take the transpose of the intermediate matrix
    Z = Z.T

    for i in range(len(K1_inverse)):
        for j in range(len(Z[0])):
            for k in range(len(Z)):
                K_inverse_Y[i][j] += K1_inverse[i][k] * Z[k][j]

    result = K_inverse_Y

    return result


# Setting up the parameters for the suammtion method
def PreSummation2D(X, Y, length, variance, codomain_2D_mapper, reg1, reg2):
    # X: The values in the x-coordinate
    # Y: The values in the y-coordinate
    # length: The length parameter for the exponential squared kernel
    # variance: The variance parameter for the exponential squared kernel
    # codomain_2D_mapper: The output of the training data as a tensor
    # reg1: Regularization constant for K1
    # reg2: Regularization constant for K2

    # Calculate the matrix of the x-component
    K1 = Covariance(X1=X.reshape(-1, 1), X2=X.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the y-component
    K2 = Covariance(X1=Y.reshape(-1, 1), X2=Y.reshape(-1, 1), length=length, variance=variance)

    # Calculate the condition numbers before regularization
    print('The condition numbers before regularization')
    print('Condition number of K1:')
    print('%.4E' % Decimal(np.linalg.cond(K1)))
    print('Condition number of K2:')
    print('%.4E' % Decimal(np.linalg.cond(K2)))

    # Regularization section
    K1 = K1 + reg1 * np.identity(len(K1))
    K2 = K2 + reg2 * np.identity(len(K2))

    # Calculate the condition numbers after regularization
    print('The condition numbers after regularization')
    print('Condition number of K1:')
    print('%.4E' % Decimal(np.linalg.cond(K1)))
    print('Condition number of K2:')
    print('%.4E' % Decimal(np.linalg.cond(K2)))

    # Obtain K^(-1)*codomain_train using the summation method
    K_InvY = Summation2D(K1=K1, K2=K2, codomain_2D_mapper=codomain_2D_mapper)

    return K_InvY


# The posterior predictive of the 2-D case caculates the mean of the condtional mutlinormal distribution
def Posterior_predictive2D(X, Y, domain_test, domain_train, length, variance, codomain_2D_mapper, reg1, reg2):
    # X: The training data in the x-coordinate 
    # Y: The training data in the y-coordinate
    # domain_test: The test data
    # domain_train: The training data
    # length: The length parameter of the exponential squared kernel
    # variance: The variance parameter of the exponential squared kernel
    # Y_2D_mapper: The output of the training data as a tensor
    # reg1: Regularization constant for K1
    # reg2: Regularization constant for K2

    # Use the summation method to find K^(-1)*codomain_train
    K_InvY_unshaped = PreSummation2D(X=X, Y=Y, length=length, variance=variance, codomain_2D_mapper=codomain_2D_mapper,
                                    reg1=reg1, reg2=reg2)

    # Reshape the matrix to a column vector to ensure valid matrix multiplication with the matrix K*
    K_InvY_shaped = np.array(ColumnGenerator2D(K_InvY_unshaped=K_InvY_unshaped))

    # Obtain the mean by calculating the matrix multiplication of K* with the result of the summation: K^(-1)*Y
    mu = Mean2D(domain_test=domain_test, domain_train=domain_train, K_Y=K_InvY_shaped, length=length, variance=variance)

    return mu


# The output of the training data in the form of a tensor
def OutputGridMap2D(X, Y):
    # X: The values in the x-coordinate
    # Y: The values in the y-coordinate

    Y_2D_mapper = np.zeros((len(X), len(Y)))

    for i in range(len(X)):      # x coordinate
        for j in range(len(Y)):  # y coordinate
            Y_2D_mapper[i][j] = Output2D(X=X[i], Y=Y[j])

    return Y_2D_mapper


# Calculates the output of the training and test data given the formula
def Output2DVector(domain):
    # domain: The training data or test data

    output = np.zeros((len(domain), 1))
    for i in range(len(domain)):
        output[i][0] = Output2D(X=domain[i][0], Y=domain[i][1])

    return output


# Calculates the output of the training and test data given the formula
def Output2D(X, Y):
    # X: The x-coordinates that will be passed into the formula
    # Y: The y-coordinates that will be passed into the formula

    # Paraboloid
    paraboloid = X ** 2 + Y ** 2

    # 2-D Morse potential function
    D_e = 37255
    a = 1.8677
    r_e = 1.275
    morse_potential = D_e * math.pow(1 - math.exp(-a * (X - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (Y - r_e)), 2)

    return paraboloid


# Returns the length parameter and variance parameter for the exponential squared kernel
def HyperParameters2D():
    length = 0.2
    variance = 1
    reg1 = 10 ** (-11)
    reg2 = 10 ** (-11)

    return length, variance, reg1, reg2


# The test data in each coordinate
def TestPoints2D():
    rx = np.random.normal(loc=1.275, scale=0.1, size=40)
    ry = np.random.normal(loc=1.275, scale=0.1, size=40)

    return rx, ry


# The training data in each coordinate
def TrainingPoints2D():
    X = np.arange(0.975, 1.575, 0.06)
    Y = np.arange(0.975, 1.575, 0.06)

    return X, Y


# The Gaussian process for a 2-D problem using the summation method
def GaussianProcess2D():
    # Start the timer to see how long the program takes to calculate the mean
    start_time = time.time()

    # Create the test data in each coordinate
    rx, ry = TestPoints2D()

    # Mesh the 2 coordinates of the test data together
    gx, gy = np.meshgrid(rx, ry)

    # Create the test data using a tensor grid product
    domain_2D_test = np.c_[gx.ravel(), gy.ravel()]

    # Create the training data in each coordinate
    X, Y = TrainingPoints2D()

    # Mesh the 2 coordinates of the training data together
    gs, gr = np.meshgrid(X, Y)

    # Create the training data using a tensor grid product
    domain_2D_train = np.c_[gs.ravel(), gr.ravel()]

    # Obtain the length and variance parameters for the exponential squared kernel and regularization constants
    length, variance, reg1, reg2 = HyperParameters2D()

    # Generate the output of the training data in the form of a tensor
    codomain_2D_train = Output2DVector(domain=domain_2D_train)

    # Create a list to store the training data for the plot
    trainer = []
    for i in range(len(codomain_2D_train)):
        trainer.append(codomain_2D_train[i][0])

    # Obtain the output of the training map in the form of a tensor
    codomain_2D_mapper = OutputGridMap2D(X=Y, Y=Y)

    # Obtain the mean of the multinormal distribution
    mu = Posterior_predictive2D(X=X, Y=Y, domain_test=domain_2D_test, domain_train=domain_2D_train, length=length,
                                variance=variance, codomain_2D_mapper=codomain_2D_mapper, reg1=reg1, reg2=reg2)

    # Print how long it took to compute the mean
    print("--- %s seconds to calculate the mean ---" % (time.time() - start_time))

    # Calculate the output of the test data
    predicted = Output2DVector(domain=domain_2D_test)

    # Create a list to store the predicted data needed for the plot
    predictedlist = []
    for i in range(len(predicted)):
        predictedlist.append(predicted[i][0])

    # Calculate the RMSE of the mean (observed data) and the output of the test data (predicted data)
    print('The RMSE of the mean: ')
    rmse = math.sqrt(mean_squared_error(mu, predicted))
    print(rmse)

    # Plot the mean, training data and the test data
    Plot_gp_2D(gx=gx, gy=gy, mu=mu, domain_test=domain_2D_test, predicted=predictedlist, domain_train=domain_2D_train,
               codomain_train=trainer, title="Gaussian Plot", i=1)

    return


########################################################################################################################
# 4-D Gaussian Process
########################################################################################################################


# Calculate the mean without calculating the K* matrix directly
def Mean4D(domain_test, domain_train, length, variance, K_Y):
    # domain_test: The test data
    # domain_train: The training data
    # length: The length parameter for the exponential squared kernel
    # variance: The variance parameter for the exponential squared kernel
    # K_Y: K^(-1)*codomain_train tensor obtained from the summation method

    mu = []

    summation = 0
    for i in range(len(domain_test)):
        for j in range(len(domain_train)):
            summation += KernelVector(X1=domain_test[i], X2=domain_train[j], length=length, variance=variance) * K_Y[j]
        mu.append(summation)
        summation = 0

    # convert the mean from a list to an array
    mu_array = np.array(mu)

    return mu_array

# Generates a column vector from the tensor K^(-1)*codomain_train
def ColumnGenerator4D(K_InvY_notcolumn):
    # K_InvY_notcolumn: The tensor K^(-1)*Y obtained from the summation method

    col = np.zeros((len(K_InvY_notcolumn)
                    * len(K_InvY_notcolumn[0])
                    * len(K_InvY_notcolumn[0][0]) 
                    * len(K_InvY_notcolumn[0][0][0])
                    , 1))

    counter = 0
    for i in range(            len(K_InvY_notcolumn)):
        for j in range(        len(K_InvY_notcolumn[0])):
            for k in range(    len(K_InvY_notcolumn[0][0])):
                for l in range(len(K_InvY_notcolumn[0][0][0])):
                    col[counter][0] = K_InvY_notcolumn[l][k][j][i]
                    counter += 1

    return col


# Calculate K^(-1)*codomain_train using the summation method
def Summation4D(K1, K2, K3, K4, Y_4D_mapper):
    # K1: The covariance matrix of only the w-coordinates
    # K2: The covariance matrix of only the x-coordinates
    # K3: The covariance matrix of only the y-coordinates
    # K4: The covariance matrix of only the z-coordinates
    # Y_4D_mapper: The output of the training data as a tensor

    # Inverting the matrices
    K1_inverse = np.linalg.inv(K1)  # We get K_1 inverse
    K2_inverse = np.linalg.inv(K2)  # We get K_2 inverse
    K3_inverse = np.linalg.inv(K3)  # We get K_3 inverse
    K4_inverse = np.linalg.inv(K4)  # We get K_4 inverse

    Z = np.zeros((len(K4_inverse), 
                  len(Y_4D_mapper[0][0][0]),
                  len(Y_4D_mapper[0][0]), 
                  len(Y_4D_mapper[0])))
    
    W = np.zeros((len(K3_inverse), 
                  len(Z[0][0][0]), 
                  len(Z[0][0]), 
                  len(Z[0])))

    V = np.zeros((len(K2_inverse), 
                  len(W[0][0][0]), 
                  len(W[0][0]), 
                  len(W[0])))
    
    K_inverse_Y = np.zeros((len(K1_inverse), 
                            len(V[0][0][0]), 
                            len(V[0][0]), 
                            len(V[0])))

    for j in range(len(K4_inverse)):
        for k in range(len(Y_4D_mapper              [0][0][0])):
            for l in range(len(Y_4D_mapper          [0][0])):
                for m in range(len(Y_4D_mapper      [0])):
                    for n in range( len(Y_4D_mapper)):
                        Z[j][k][l][m] += K4_inverse[j][n] * Y_4D_mapper[n][m][l][k]
    Z = Z.T
    for j in range(len(K3_inverse)):
        for k in range(len(Z             [0][0][0])):
            for l in range(len(Z         [0][0])):
                for m in range(len(Z     [0])):
                    for n in range(len(Z)):
                        W[j][k][l][m] += K3_inverse[j][n] * Z[n][m][l][k]
    W = W.T
    for j in range(len(K2_inverse)):
        for k in range(len(W             [0][0][0])):
            for l in range(len(W         [0][0])):
                for m in range(len(W     [0])):
                    for n in range(len(W)):
                        V[j][k][l][m] += K2_inverse[j][n] * W[n][m][l][k]
    V = V.T
    for j in range(len(K1_inverse)):
        for k in range(len(V             [0][0][0])):
            for l in range(len(V         [0][0])):
                for m in range(len(V     [0])):
                    for n in range(len(V)):
                        K_inverse_Y[j][k][l][m] += K1_inverse[j][n] * V[n][m][l][k]
    result = K_inverse_Y

    return result


# Setting up the parameters for the summation method
def PreSummation4D(W, X, Y, Z, length, variance, Y_4D_mapper, reg1, reg2, reg3, reg4):
    # W: The values in the w-coordinate
    # X: The values in the x-coordinate
    # Y: The values in the y-coordinate
    # Z: The values in the z-coordinate
    # length: The length parameter for the exponential squared kernel
    # variance: The variance of the exponential squared kernel
    # Y_4D_mapper: The output of the training data as a tensor 
    # reg1: Regularization constant for K1
    # reg2: Regularization constant for K2
    # reg3: Regularization constant for K3
    # reg4: Regularization constant for K4

    # Calculate the matrix of the w-coordinate
    K1 = Covariance(X1=W.reshape(-1, 1), X2=W.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the x-coordinate
    K2 = Covariance(X1=X.reshape(-1, 1), X2=X.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the y-coordinate
    K3 = Covariance(X1=Y.reshape(-1, 1), X2=Y.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the z-coordinate
    K4 = Covariance(X1=Z.reshape(-1, 1), X2=Z.reshape(-1, 1), length=length, variance=variance)

    # Calculate the condition numbers before regularization
    print('Condition number for K1')
    print('%.4E' % Decimal(np.linalg.cond(K1)))
    print('Condition number for K2')
    print('%.4E' % Decimal(np.linalg.cond(K2)))
    print('Condition number for K3')
    print('%.4E' % Decimal(np.linalg.cond(K3)))
    print('Condition number for K4')
    print('%.4E' % Decimal(np.linalg.cond(K4)))

    # Regularization section
    K1 = K1 + reg1 * np.identity(len(W))
    K2 = K2 + reg2 * np.identity(len(X))
    K3 = K3 + reg3 * np.identity(len(Y))
    K4 = K4 + reg4 * np.identity(len(Z))

    # Calculate the condition numbers after regularization
    print('Condition numbers after regularization')
    print('Condition number for K1')
    print('%.4E' % Decimal(np.linalg.cond(K1)))
    print('Condition number for K2')
    print('%.4E' % Decimal(np.linalg.cond(K2)))
    print('Condition number for K3')
    print('%.4E' % Decimal(np.linalg.cond(K3)))
    print('Condition number for K4')
    print('%.4E' % Decimal(np.linalg.cond(K4)))

    # Obtain K^(-1)*Y using the summation method
    K_InvY = Summation4D(K1=K1, K2=K2, K3=K3, K4=K4, Y_4D_mapper=Y_4D_mapper)
    
    return K_InvY

# The output of the training data in the form of a tensor
def OutputGridMap4D(W, X, Y, Z):
    # W: The training data in the w-coordinate
    # X: The training data in the x-coordinate
    # Y: The training data in the y-coordinate
    # Z: The training data in the z-coordinate

    Y_4D_mapper = np.zeros((len(W), len(X), len(Y), len(Z)))
    for i in range(len(W)):  # w coordinate
        for j in range(len(X)):  # x coordinate
            for k in range(len(Y)):  # y coordinate
                for l in range(len(Z)):  # Z coordinate
                    Y_4D_mapper[i][j][k][l] = Output4D(W=W[i], X=X[j], Y=Y[k], Z=Z[l])

    return Y_4D_mapper


# The posterior predictive of the 4-D case calculates the mean of the condtional mutlinormal distribution
def Posterior_predictive4D(W, X, Y, Z, domain_test, domain_train, length, variance, Y_4D_mapper, reg1, reg2, reg3, reg4):
    # W: The training data in the w-coordinate  
    # X: The training data in the x-coordinate 
    # Y: The training data in the y-coordinate
    # Z: The training data in the z-coordinate
    # domain_test: The test data
    # domain_train: The training data
    # length: The length parameter of the exponential squared kernel
    # variance: The variance of the exponential squared kernel
    # Y_4D_mapper: The output of the training data as a tensor
    # reg1: Regularization constant for K1
    # reg2: Regularization constant for K2
    # reg3: Regularization constant for K3
    # reg4: Regularization constant for K4

    # Use the summation method to find K^(-1)*codomain_train
    K_InvY_notcolumn = PreSummation4D(W=W, X=X, Y=Y, Z=Z, length=length, variance=variance, Y_4D_mapper=Y_4D_mapper,
                                   reg1=reg1, reg2=reg2, reg3=reg3, reg4=reg4)

    # Reshape the matrix to a column vector to ensure valid matrix multiplication with the matrix K*
    K_InvY_shaped = ColumnGenerator4D(K_InvY_notcolumn=K_InvY_notcolumn)

    # Calculate the mean using a summation method to reduce the space complexity increases the time complexity
    mu = Mean4D(domain_test=domain_test, domain_train=domain_train, length=length, variance=variance, K_Y=K_InvY_shaped)

    # Return the mean of the multinormal distribution
    return mu


# Calculates the output of the training and test data given the formula
def Output4DVector(domain):
    # domain: The training data or test data

    output = np.zeros((len(domain), 1))
    for i in range(len(domain)):
        output[i][0] = Output4D(W=domain[i][0], X=domain[i][1], Y=domain[i][2], Z=domain[i][3])

    return output


# Calculates the output of the training and test data given the formula
def Output4D(W, X, Y, Z):
    # W: The w-coordinates that will be passed into the formula
    # X: The x-coordinates that will be passed into the formula
    # Y: The y-coordinates that will be passed into the formula
    # Z: The z-coordinates that will be passed into the formula

    # paraboloid
    paraboloid = W ** 2 + X ** 2 + Y ** 2 + Z ** 2

    # 4-D Morse potential function
    D_e = 37255
    a = 1.8677
    r_e = 1.275
    morse_potential = D_e * math.pow(1 - math.exp(-a * (W - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (X - r_e)), 2)\
                      + D_e * math.pow(1 - math.exp(-a * (Y - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (Z - r_e)), 2)

    return morse_potential


# Returns the length parameter and variance parameter for the exponential squared kernel
def HyperParameters4D():
    length = 0.065
    variance = 1
    reg1 = 0
    reg2 = 0
    reg3 = 0
    reg4 = 0

    return length, variance, reg1, reg2, reg3, reg4


# The test data in each coordinate
def TestPoints4D():
    rw = np.random.normal(loc=1.275, scale=0.09, size=6)
    rx = np.random.normal(loc=1.275, scale=0.09, size=6)
    ry = np.random.normal(loc=1.275, scale=0.09, size=6)
    rz = np.random.normal(loc=1.275, scale=0.09, size=6)

    return rw, rx, ry, rz


# The training data in each coordinate
def TrainingPoints4D():
    W = np.arange(1, 1.55, 0.036)
    X = np.arange(1, 1.55, 0.036)
    Y = np.arange(1, 1.55, 0.036)
    Z = np.arange(1, 1.55, 0.036)

    return W, X, Y, Z


# The Gaussian process for a 4-D problem using the summation method
def GaussianProcess4D():
    # Start the timer to see how long the program takes to calculate the mean
    start_time = time.time()

    # Create the test data in each coordinate
    rw, rx, ry, rz = TestPoints4D()

    # Mesh the 4 coordinates of the test data together
    gw, gx, gy, gz = np.meshgrid(rw, rx, ry, rz)

   # Create the test data using a tensor grid product
    domain_4D_test = np.c_[gw.ravel(), gx.ravel(), gy.ravel(), gz.ravel()]

    # Create the training data in each coordinate
    W, X, Y, Z = TrainingPoints4D()

    # Mesh the 4 coordinates of the training data together
    gs, gr, gq, gv = np.meshgrid(W, X, Y, Z)

    # Create the training data using a tensor grid product
    domain_4D_train = np.c_[gs.ravel(), gr.ravel(), gq.ravel(), gv.ravel()]

    # Obtain the length and variance parameters for the exponential squared kernel and regularization constants
    length, variance, reg1, reg2, reg3, reg4 = HyperParameters4D()

    # Generate the output of the training data in the form of a tensor
    codomain_4D_mapper = OutputGridMap4D(W=W, X=X, Y=Y, Z=Z)

    # Obtain the mean of the conditioned multinormal distribution
    mu = Posterior_predictive4D(W=W, X=X, Y=Y, Z=Z, domain_test=domain_4D_test, domain_train=domain_4D_train,
                                length=length, variance=variance, Y_4D_mapper=codomain_4D_mapper, reg1=reg1, reg2=reg2,
                                reg3=reg3, reg4=reg4)

    # Print how long it took to compute the mean
    print("--- %s seconds to calculate the mean ---" % (time.time() - start_time))

    # Calculate the output of the test data
    predicted = Output4DVector(domain=domain_4D_test)

   # Calculate the RMSE of the mean (observed data) and the output of the test data (predicted data)
    print('The RMSE of the mean: ')
    rmse = math.sqrt(mean_squared_error(mu, predicted))
    print(rmse)

    return

########################################################################################################################
# 9-D Gaussian Process
########################################################################################################################


# Calculate the mean without calculating the K* matrix directly
def Mean9D(domain_test, domain_train, length, variance, K_Y):
    # domain_test: The test data
    # domain_train: The training data
    # length: The length parameter for the exponential squared kernel
    # variance: The variance for the exponential squared kernel
    # K_Y: K^(-1)*codomain_train tensor obtained from the summation method

    mu = []
    summation = 0
    for i in range(len(domain_test)):
        for j in range(len(domain_train)):
            summation += KernelVector(X1=domain_test[i], X2=domain_train[j], length=length, variance=variance) * K_Y[j]
        mu.append(summation)
        summation = 0

    # convert the mean from a list to an array
    mu_array = np.array(mu)

    return mu_array


# Generates a column vector from the tensor K^(-1)*codomain_train
def ColumnGenerator9D(K_Y_notcolumn):
    # K_Y_notcolumn: The tensor K^(-1)*Y obtained from the summation method

    col = np.zeros((len(K_Y_notcolumn)
                    * len(K_Y_notcolumn[0])
                    * len(K_Y_notcolumn[0][0])
                    * len(K_Y_notcolumn[0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0][0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0][0][0][0][0])
                    , 1))

    counter = 0
    for a in range(len(K_Y_notcolumn)):
        for b in range(len(K_Y_notcolumn[0])):
            for c in range(len(K_Y_notcolumn[0][0])):
                for d in range(len(K_Y_notcolumn[0][0][0])):
                    for e in range(len(K_Y_notcolumn[0][0][0][0])):
                        for f in range(len(K_Y_notcolumn[0][0][0][0][0])):
                            for g in range(len(K_Y_notcolumn[0][0][0][0][0][0])):
                                for h in range(len(K_Y_notcolumn[0][0][0][0][0][0][0])):
                                    for i in range(len(K_Y_notcolumn[0][0][0][0][0][0][0][0])):
                                        col[counter][0] = K_Y_notcolumn[i][h][g][f][e][d][c][b][a]
                                        counter += 1

    return col


# Calculate K^(-1)*codomain_train using the summation method
def Summation9D(K1, K2, K3, K4, K5, K6, K7, K8, K9, Y_9D_mapper):
    # K1: The covariance matrix of only the a-coordinates
    # K2: The covariance matrix of only the b-coordinates
    # K3: The covariance matrix of only the c-coordinates
    # K4: The covariance matrix of only the d-coordinates
    # K5: The covariance matrix of only the e-coordinates
    # K6: The covariance matrix of only the f-coordinates
    # K7: The covariance matrix of only the g-coordinates
    # K8: The covariance matrix of only the h-coordinates
    # K9: The covariance matrix of only the i-coordinates
    # Y_9D_mapper: The output of the training data as a tensor

    # Inverting the matrices
    K1_inverse = np.linalg.inv(K1)  # We get K_1 inverse
    K2_inverse = np.linalg.inv(K2)  # We get K_2 inverse
    K3_inverse = np.linalg.inv(K3)  # We get K_3 inverse
    K4_inverse = np.linalg.inv(K4)  # We get K_4 inverse
    K5_inverse = np.linalg.inv(K5)  # We get K_5 inverse
    K6_inverse = np.linalg.inv(K6)  # We get K_6 inverse
    K7_inverse = np.linalg.inv(K7)  # We get K_7 inverse
    K8_inverse = np.linalg.inv(K8)  # We get K_8 inverse
    K9_inverse = np.linalg.inv(K9)  # We get K_9 inverse

    A = np.zeros((len(K9_inverse),
                  len(Y_9D_mapper[0][0][0][0][0][0][0][0]),
                  len(Y_9D_mapper[0][0][0][0][0][0][0]),
                  len(Y_9D_mapper[0][0][0][0][0][0]),
                  len(Y_9D_mapper[0][0][0][0][0]),
                  len(Y_9D_mapper[0][0][0][0]),
                  len(Y_9D_mapper[0][0][0]),
                  len(Y_9D_mapper[0][0]),
                  len(Y_9D_mapper[0])))

    B = np.zeros((len(K8_inverse),
                  len(A[0][0][0][0][0][0][0][0]),
                  len(A[0][0][0][0][0][0][0]),
                  len(A[0][0][0][0][0][0]),
                  len(A[0][0][0][0][0]),
                  len(A[0][0][0][0]),
                  len(A[0][0][0]),
                  len(A[0][0]),
                  len(A[0])))

    C = np.zeros((len(K7_inverse),
                  len(B[0][0][0][0][0][0][0][0]),
                  len(B[0][0][0][0][0][0][0]),
                  len(B[0][0][0][0][0][0]),
                  len(B[0][0][0][0][0]),
                  len(B[0][0][0][0]),
                  len(B[0][0][0]),
                  len(B[0][0]),
                  len(B[0])))

    D = np.zeros((len(K6_inverse),
                  len(C[0][0][0][0][0][0][0][0]),
                  len(C[0][0][0][0][0][0][0]),
                  len(C[0][0][0][0][0][0]),
                  len(C[0][0][0][0][0]),
                  len(C[0][0][0][0]),
                  len(C[0][0][0]),
                  len(C[0][0]),
                  len(C[0])))

    E = np.zeros((len(K5_inverse),
                  len(D[0][0][0][0][0][0][0][0]),
                  len(D[0][0][0][0][0][0][0]),
                  len(D[0][0][0][0][0][0]),
                  len(D[0][0][0][0][0]),
                  len(D[0][0][0][0]),
                  len(D[0][0][0]),
                  len(D[0][0]),
                  len(D[0])))

    F = np.zeros((len(K4_inverse),
                  len(E[0][0][0][0][0][0][0][0]),
                  len(E[0][0][0][0][0][0][0]),
                  len(E[0][0][0][0][0][0]),
                  len(E[0][0][0][0][0]),
                  len(E[0][0][0][0]),
                  len(E[0][0][0]),
                  len(E[0][0]),
                  len(E[0])))

    G = np.zeros((len(K3_inverse),
                  len(F[0][0][0][0][0][0][0][0]),
                  len(F[0][0][0][0][0][0][0]),
                  len(F[0][0][0][0][0][0]),
                  len(F[0][0][0][0][0]),
                  len(F[0][0][0][0]),
                  len(F[0][0][0]),
                  len(F[0][0]),
                  len(F[0])))

    H = np.zeros((len(K2_inverse),
                  len(G[0][0][0][0][0][0][0][0]),
                  len(G[0][0][0][0][0][0][0]),
                  len(G[0][0][0][0][0][0]),
                  len(G[0][0][0][0][0]),
                  len(G[0][0][0][0]),
                  len(G[0][0][0]),
                  len(G[0][0]),
                  len(G[0])))

    K_inverse_Y = np.zeros((len(K1_inverse),
                  len(H[0][0][0][0][0][0][0][0]),
                  len(H[0][0][0][0][0][0][0]),
                  len(H[0][0][0][0][0][0]),
                  len(H[0][0][0][0][0]),
                  len(H[0][0][0][0]),
                  len(H[0][0][0]),
                  len(H[0][0]),
                  len(H[0])))

    for a in range(len(K9_inverse)):
        for b in range(len(Y_9D_mapper                              [0][0][0][0][0][0][0][0])):
            for c in range(len(Y_9D_mapper                          [0][0][0][0][0][0][0])):
                for d in range(len(Y_9D_mapper                      [0][0][0][0][0][0])):
                    for e in range(len(Y_9D_mapper                  [0][0][0][0][0])):
                        for f in range(len(Y_9D_mapper              [0][0][0][0])):
                            for g in range(len(Y_9D_mapper          [0][0][0])):
                                for h in range(len(Y_9D_mapper      [0][0])):
                                    for i in range(len(Y_9D_mapper  [0])):
                                        for j in range(len(Y_9D_mapper)):
                                            A[a][b][c][d][e][f][g][h][i] += \
                                                K9_inverse[a][j] * Y_9D_mapper[j][i][h][g][f][e][d][c][b]
    A = A.T
    for a in range(len(K8_inverse)):
        for b in range(len(A                                [0][0][0][0][0][0][0][0])):
            for c in range(len(A                            [0][0][0][0][0][0][0])):
                for d in range(len(A                        [0][0][0][0][0][0])):
                    for e in range(len(A                    [0][0][0][0][0])):
                        for f in range(len(A                [0][0][0][0])):
                            for g in range(len(A            [0][0][0])):
                                for h in range(len(A        [0][0])):
                                    for i in range(len(A    [0])):
                                        for j in range(len(A)):
                                            B[a][b][c][d][e][f][g][h][i] += \
                                                K8_inverse[a][j] * A[j][i][h][g][f][e][d][c][b]
    B = B.T
    for a in range(len(K7_inverse)):
        for b in range(len(B                                [0][0][0][0][0][0][0][0])):
            for c in range(len(B                            [0][0][0][0][0][0][0])):
                for d in range(len(B                        [0][0][0][0][0][0])):
                    for e in range(len(B                    [0][0][0][0][0])):
                        for f in range(len(B                [0][0][0][0])):
                            for g in range(len(B            [0][0][0])):
                                for h in range(len(B        [0][0])):
                                    for i in range(len(B    [0])):
                                        for j in range(len(B)):
                                            C[a][b][c][d][e][f][g][h][i] += \
                                                K7_inverse[a][j] * B[j][i][h][g][f][e][d][c][b]
    C = C.T
    for a in range(len(K6_inverse)):
        for b in range(len(C                                [0][0][0][0][0][0][0][0])):
            for c in range(len(C                            [0][0][0][0][0][0][0])):
                for d in range(len(C                        [0][0][0][0][0][0])):
                    for e in range(len(C                    [0][0][0][0][0])):
                        for f in range(len(C                [0][0][0][0])):
                            for g in range(len(C            [0][0][0])):
                                for h in range(len(C        [0][0])):
                                    for i in range(len(C    [0])):
                                        for j in range(len(C)):
                                            D[a][b][c][d][e][f][g][h][i] += \
                                                K6_inverse[a][j] * C[j][i][h][g][f][e][d][c][b]
    D = D.T
    for a in range(len(K5_inverse)):
        for b in range(len(D                                [0][0][0][0][0][0][0][0])):
            for c in range(len(D                            [0][0][0][0][0][0][0])):
                for d in range(len(D                        [0][0][0][0][0][0])):
                    for e in range(len(D                    [0][0][0][0][0])):
                        for f in range(len(D                [0][0][0][0])):
                            for g in range(len(D            [0][0][0])):
                                for h in range(len(D        [0][0])):
                                    for i in range(len(D    [0])):
                                        for j in range(len(D)):
                                            E[a][b][c][d][e][f][g][h][i] += \
                                                K5_inverse[a][j] * D[j][i][h][g][f][e][d][c][b]
    E = E.T
    for a in range(len(K4_inverse)):
        for b in range(len(E                                [0][0][0][0][0][0][0][0])):
            for c in range(len(E                            [0][0][0][0][0][0][0])):
                for d in range(len(E                        [0][0][0][0][0][0])):
                    for e in range(len(E                    [0][0][0][0][0])):
                        for f in range(len(E                [0][0][0][0])):
                            for g in range(len(E            [0][0][0])):
                                for h in range(len(E        [0][0])):
                                    for i in range(len(E    [0])):
                                        for j in range(len(E)):
                                            F[a][b][c][d][e][f][g][h][i] += \
                                                K4_inverse[a][j] * E[j][i][h][g][f][e][d][c][b]
    F = F.T
    for a in range(len(K3_inverse)):
        for b in range(len(F                                [0][0][0][0][0][0][0][0])):
            for c in range(len(F                            [0][0][0][0][0][0][0])):
                for d in range(len(F                        [0][0][0][0][0][0])):
                    for e in range(len(F                    [0][0][0][0][0])):
                        for f in range(len(F                [0][0][0][0])):
                            for g in range(len(F            [0][0][0])):
                                for h in range(len(F        [0][0])):
                                    for i in range(len(F    [0])):
                                        for j in range(len(F)):
                                            G[a][b][c][d][e][f][g][h][i] += \
                                                K3_inverse[a][j] * F[j][i][h][g][f][e][d][c][b]
    G = G.T
    for a in range(len(K2_inverse)):
        for b in range(len(G                                [0][0][0][0][0][0][0][0])):
            for c in range(len(G                            [0][0][0][0][0][0][0])):
                for d in range(len(G                        [0][0][0][0][0][0])):
                    for e in range(len(G                    [0][0][0][0][0])):
                        for f in range(len(G                [0][0][0][0])):
                            for g in range(len(G            [0][0][0])):
                                for h in range(len(G        [0][0])):
                                    for i in range(len(G    [0])):
                                        for j in range(len(G)):
                                            H[a][b][c][d][e][f][g][h][i] += \
                                                K2_inverse[a][j] * G[j][i][h][g][f][e][d][c][b]
    H = H.T
    for a in range(len(K1_inverse)):
        for b in range(len(H                                [0][0][0][0][0][0][0][0])):
            for c in range(len(H                            [0][0][0][0][0][0][0])):
                for d in range(len(H                        [0][0][0][0][0][0])):
                    for e in range(len(H                    [0][0][0][0][0])):
                        for f in range(len(H                [0][0][0][0])):
                            for g in range(len(H            [0][0][0])):
                                for h in range(len(H        [0][0])):
                                    for i in range(len(H    [0])):
                                        for j in range(len(H)):
                                            K_inverse_Y[a][b][c][d][e][f][g][h][i] += \
                                                K1_inverse[a][j] * H[j][i][h][g][f][e][d][c][b]
    return K_inverse_Y


# Setting up the parameters for the summation method
def PreSummation9D(A, B, C, D, E, F, G, H, I, length, variance, Y_9D_mapper, reg1, reg2, reg3, reg4, reg5, reg6,
                   reg7, reg8, reg9):
    # A: The training data in the a-coordinate
    # B: The training data in the b-coordinate
    # C: The training data in the c-coordinate
    # D: The training data in the d-coordinate
    # E: The training data in the e-coordinate
    # F: The training data in the f-coordinate
    # G: The training data in the g-coordinate
    # H: The training data in the h-coordinate
    # I: The training data in the i-coordinate
    # length: The length parameter for the exponential squared kernel
    # variance: The variance parameter for the exponential squared kernel
    # Y_9D_mapper: The output of the training data as a tensor
    # reg1: Regularization constant for K1
    # reg2: Regularization constant for K2
    # reg3: Regularization constant for K3
    # reg4: Regularization constant for K4
    # reg5: Regularization constant for K5
    # reg6: Regularization constant for K6
    # reg7: Regularization constant for K7
    # reg8: Regularization constant for K8
    # reg9: Regularization constant for K9

    # Calculate the matrix of the a-coordinate
    K1 = Covariance(X1=A.reshape(-1, 1), X2=A.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the b-coordinate
    K2 = Covariance(X1=B.reshape(-1, 1), X2=B.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the c-coordinate
    K3 = Covariance(X1=C.reshape(-1, 1), X2=C.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the d-coordinate
    K4 = Covariance(X1=D.reshape(-1, 1), X2=D.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the e-coordinate
    K5 = Covariance(X1=E.reshape(-1, 1), X2=E.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the f-coordinate
    K6 = Covariance(X1=F.reshape(-1, 1), X2=F.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the g-coordinate
    K7 = Covariance(X1=G.reshape(-1, 1), X2=G.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the h-coordinate
    K8 = Covariance(X1=H.reshape(-1, 1), X2=H.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the i-coordinate
    K9 = Covariance(X1=I.reshape(-1, 1), X2=I.reshape(-1, 1), length=length, variance=variance)

    # Calculate the condition numbers before regularization
    print('Condition number for K1')
    print('%.4E' % Decimal(np.linalg.cond(K1)))
    print('Condition number for K2')
    print('%.4E' % Decimal(np.linalg.cond(K2)))
    print('Condition number for K3')
    print('%.4E' % Decimal(np.linalg.cond(K3)))
    print('Condition number for K4')
    print('%.4E' % Decimal(np.linalg.cond(K4)))
    print('Condition number for K5')
    print('%.4E' % Decimal(np.linalg.cond(K5)))
    print('Condition number for K6')
    print('%.4E' % Decimal(np.linalg.cond(K6)))
    print('Condition number for K7')
    print('%.4E' % Decimal(np.linalg.cond(K7)))
    print('Condition number for K8')
    print('%.4E' % Decimal(np.linalg.cond(K8)))
    print('Condition number for K9')
    print('%.4E' % Decimal(np.linalg.cond(K9)))

    # Regularization section
    K1 = K1 + reg1 * np.identity(len(A))
    K2 = K2 + reg2 * np.identity(len(B))
    K3 = K3 + reg3 * np.identity(len(C))
    K4 = K4 + reg4 * np.identity(len(D))
    K5 = K5 + reg5 * np.identity(len(E))
    K6 = K6 + reg6 * np.identity(len(F))
    K7 = K7 + reg7 * np.identity(len(G))
    K8 = K8 + reg8 * np.identity(len(H))
    K9 = K9 + reg9 * np.identity(len(I))

    # Calculate the condition numbers after regularization
    print('Condition numbers after regularization')
    print('Condition number for K1')
    print('%.4E' % Decimal(np.linalg.cond(K1)))
    print('Condition number for K2')
    print('%.4E' % Decimal(np.linalg.cond(K2)))
    print('Condition number for K3')
    print('%.4E' % Decimal(np.linalg.cond(K3)))
    print('Condition number for K4')
    print('%.4E' % Decimal(np.linalg.cond(K4)))
    print('Condition number for K5')
    print('%.4E' % Decimal(np.linalg.cond(K5)))
    print('Condition number for K6')
    print('%.4E' % Decimal(np.linalg.cond(K6)))
    print('Condition number for K7')
    print('%.4E' % Decimal(np.linalg.cond(K7)))
    print('Condition number for K8')
    print('%.4E' % Decimal(np.linalg.cond(K8)))
    print('Condition number for K9')
    print('%.4E' % Decimal(np.linalg.cond(K9)))

    # Obtain K^(-1)*Y using the summation method
    K_InvY = Summation9D(K1=K1, K2=K2, K3=K3, K4=K4, K5=K5, K6=K6, K7=K7, K8=K8, K9=K9, Y_9D_mapper=Y_9D_mapper)

    return K_InvY


# The output of the training data in the form of a tensor
def OutputGridMap9D(A, B, C, D, E, F, G, H, I):
    # A: The training data in the a-coordinate
    # B: The training data in the b-coordinate
    # C: The training data in the c-coordinate
    # D: The training data in the d-coordinate
    # E: The training data in the e-coordinate
    # F: The training data in the f-coordinate
    # G: The training data in the g-coordinate
    # H: The training data in the h-coordinate
    # I: The training data in the i-coordinate

    Y_9D_mapper = np.zeros((len(A), len(B), len(C), len(D), len(E), len(F), len(G), len(H), len(I)))
    for a in range(len(A)):  # a coordinate
        for b in range(len(B)):  # b coordinate
            for c in range(len(C)):  # c coordinate
                for d in range(len(D)):  # d coordinate
                    for e in range(len(E)):  # e coordinate
                        for f in range(len(F)):  # f coordinate
                            for g in range(len(G)):  # g coordinate
                                for h in range(len(H)):  # h coordinate
                                    for i in range(len(I)):  # i coordinate
                                        Y_9D_mapper[a][b][c][d][e][f][g][h][i] = Output9D(A=A[a], B=B[b],
                                                                                              C=C[c], D=D[d],
                                                                                              E=E[e], F=F[f],
                                                                                              G=G[g], H=H[h],
                                                                                              I=I[i])

    return Y_9D_mapper


# The posterior predictive of the 9-D case caculates the mean of the condtional mutlinormal distribution
def Posterior_predictive9D(A, B, C, D, E, F, G, H, I, domain_test, domain_train, length, variance, Y_9D_mapper,
                            reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9):
    # A: The training data in the a-coordinate
    # B: The training data in the b-coordinate
    # C: The training data in the c-coordinate
    # D: The training data in the d-coordinate
    # E: The training data in the e-coordinate
    # F: The training data in the f-coordinate
    # G: The training data in the g-coordinate
    # H: The training data in the h-coordinate
    # I: The training data in the i-coordinate
    # domain_test: The test data
    # domain_train: The training data
    # length: The length parameter of the exponential squared kernel
    # variance: The variance of the exponential squared kernel
    # Y_9D_mapper: The output of the training data with four indices
    # reg1: Regularization constant for K1
    # reg2: Regularization constant for K2
    # reg3: Regularization constant for K3
    # reg4: Regularization constant for K4
    # reg5: Regularization constant for K5
    # reg6: Regularization constant for K6
    # reg7: Regularization constant for K7
    # reg8: Regularization constant for K8
    # reg9: Regularization constant for K9

    # Use the summation method to find K^(-1)*codomain_train
    K_Y_notcolumn = PreSummation9D(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, length=length, variance=variance,
                                   Y_9D_mapper=Y_9D_mapper, reg1=reg1, reg2=reg2, reg3=reg3, reg4=reg4, reg5=reg5,
                                   reg6=reg6, reg7=reg7, reg8=reg8, reg9=reg9)

    # Reshape the matrix to a column vector to ensure valid matrix multiplication with the matrix K*
    K_InvY_shaped = ColumnGenerator9D(K_Y_notcolumn=K_Y_notcolumn)

    # Calculate the mean using a summation method to reduce the space complexity increases the time complexity
    mu = Mean9D(domain_test=domain_test, domain_train=domain_train, length=length, variance=variance,
                K_Y=K_InvY_shaped)

    # Return the mean of the multinormal distribution
    return mu


# Calculates the output of the training and test data given the formula
def Output9DVector(domain):
    # domain: The training data or test data

    output = np.zeros((len(domain), 1))
    for i in range(len(domain)):
        output[i][0] = Output9D(A=domain[i][0], B=domain[i][1], C=domain[i][2], D=domain[i][3], E=domain[i][4],
                                F=domain[i][5], G=domain[i][6], H=domain[i][7], I=domain[i][8])

    return output


# Calculates the output of a given formula
def Output9D(A, B, C, D, E, F, G, H, I):
    # A: The a-coordinates that will be passed into the formula
    # B: The b-coordinates that will be passed into the formula
    # C: The c-coordinates that will be passed into the formula
    # D: The d-coordinates that will be passed into the formula
    # E: The e-coordinates that will be passed into the formula
    # F: The f-coordinates that will be passed into the formula
    # G: The g-coordinates that will be passed into the formula
    # H: The h-coordinates that will be passed into the formula
    # I: The i-coordinates that will be passed into the formula

    # 9-D Morse potential function
    D_e = 37255
    a = 1.8677
    r_e = 1.275
    morse_potential = D_e * math.pow(1 - math.exp(-a * (A - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (B - r_e)),
                                                                                       2)\
                      + D_e * math.pow(1 - math.exp(-a * (C - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (D - r_e)),
                                                                                       2) \
                      + D_e * math.pow(1 - math.exp(-a * (E - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (F - r_e)),
                                                                                       2) \
                      + D_e * math.pow(1 - math.exp(-a * (G - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (H - r_e)),
                                                                                       2) \
                      + D_e * math.pow(1 - math.exp(-a * (I - r_e)), 2)

    return morse_potential


# Returns the length parameter and variance parameter for the exponential squared kernel
def HyperParameters9D():
    length = 1
    variance = 1
    reg1 = 0
    reg2 = 0
    reg3 = 0
    reg4 = 0
    reg5 = 0
    reg6 = 0
    reg7 = 0
    reg8 = 0
    reg9 = 0

    return length, variance, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9


# The test data in each coordinate
def TestPoints9D():
    ra = np.random.normal(loc=1.275, scale=0.09, size=3)
    rb = np.random.normal(loc=1.275, scale=0.09, size=3)
    rc = np.random.normal(loc=1.275, scale=0.09, size=3)
    rd = np.random.normal(loc=1.275, scale=0.09, size=3)
    re = np.random.normal(loc=1.275, scale=0.09, size=3)
    rf = np.random.normal(loc=1.275, scale=0.09, size=3)
    rg = np.random.normal(loc=1.275, scale=0.09, size=3)
    rh = np.random.normal(loc=1.275, scale=0.09, size=3)
    ri = np.random.normal(loc=1.275, scale=0.09, size=3)

    return ra, rb, rc, rd, re, rf, rg, rh, ri


# The training data in each coordinate
def TrainingPoints9D():
    A = np.arange(1, 2, 0.2)
    print(A)
    B = np.arange(1, 2, 0.2)
    C = np.arange(1, 2, 0.2)
    D = np.arange(1, 2, 0.2)
    E = np.arange(1, 2, 0.2)
    F = np.arange(1, 2, 0.2)
    G = np.arange(1, 2, 0.2)
    H = np.arange(1, 2, 0.2)
    I = np.arange(1, 2, 0.2)

    return A, B, C, D, E, F, G, H, I


# The Gaussian process for a 9-D problem using the summation method
def GaussianProcess9D():
    print("9D Gaussian process")
    # Start the timer to see how long the program takes to calculate the mean
    start_time = time.time()

    # Create the test data in each coordinate
    ra, rb, rc, rd, re, rf, rg, rh, ri = TestPoints9D()

    # Mesh the 9 coordinates of the test data together
    ga, gb, gc, gd, ge, gf, gg, gh, gi = np.meshgrid(ra, rb, rc, rd, re, rf, rg, rh, ri)

    # Create the test data using a tensor grid product
    domain_9D_test = np.c_[ga.ravel(), gb.ravel(), gc.ravel(), gd.ravel(), ge.ravel(), gf.ravel(), gg.ravel(),
                           gh.ravel(), gi.ravel()]

    # Create the training data in each coordinate
    A, B, C, D, E, F, G, H, I = TrainingPoints9D()

    # Mesh the 9 coordinates of the training data together
    gA, gB, gC, gD, gE, gF, gG, gH, gI = np.meshgrid(A, B, C, D, E, F, G, H, I)

    # Create the training data using a tensor grid product
    domain_9D_train = np.c_[gA.ravel(), gB.ravel(), gC.ravel(), gD.ravel(), gE.ravel(), gF.ravel(), gG.ravel(),
                             gH.ravel(), gI.ravel()]

    # Obtain the length and variance parameters for the exponential squared kernel and regularization constants
    length, variance, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9 = HyperParameters9D()

    # Generate the output of the training data in the form of a tensor
    codomain_9D_mapper = OutputGridMap9D(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I)

    # Obtain the mean of the conditioned multinormal distribution
    mu = Posterior_predictive9D(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, domain_test=domain_9D_test,
                                domain_train=domain_9D_train, length=length, variance=variance,
                                Y_9D_mapper=codomain_9D_mapper, reg1=reg1, reg2=reg2, reg3=reg3, reg4=reg4,
                                reg5=reg5, reg6=reg6, reg7=reg7, reg8=reg8, reg9=reg9)

    # Print how long it took to compute the mean
    print("--- %s seconds to calculate the mean ---" % (time.time() - start_time))

    # Calculate the output of the test data
    predicted = Output9DVector(domain=domain_9D_test)

    # Calculate the RMSE of the mean (observed data) and the output of the test data (predicted data)
    print('The RMSE of the mean:')
    rmse = math.sqrt(mean_squared_error(mu, predicted))
    print(rmse)

    return


########################################################################################################################
# 10-D Gaussian Process
########################################################################################################################


# Calculate the mean without calculating the K* matrix directly
def Mean10D(domain_test, domain_train, length, variance, K_Y):
    # domain_test: The test data
    # domain_train: The training data
    # length: The length parameter for the exponential squared kernel
    # variance: The variance for the exponential squared kernel
    # K_Y: K^(-1)*codomain_train tensor obtained from the summation method

    mu = []
    summation = 0
    for i in range(len(domain_test)):
        for j in range(len(domain_train)):
            summation += KernelVector(X1=domain_test[i], X2=domain_train[j], length=length, variance=variance) * K_Y[j]
        mu.append(summation)
        summation = 0

    # convert the mean from a list to an array
    mu_array = np.array(mu)

    return mu_array


# Generates a column vector from the tensor K^(-1)*codomain_train
def ColumnGenerator10D(K_Y_notcolumn):
    # K_Y_notcolumn: The tensor K^(-1)*Y obtained from the summation method

    col = np.zeros((len(K_Y_notcolumn)
                    * len(K_Y_notcolumn[0])
                    * len(K_Y_notcolumn[0][0])
                    * len(K_Y_notcolumn[0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0][0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0][0][0][0][0])
                    * len(K_Y_notcolumn[0][0][0][0][0][0][0][0][0])
                    , 1))

    counter = 0
    for a in range(                                    len(K_Y_notcolumn)):
        for b in range(                                len(K_Y_notcolumn[0])):
            for c in range(                            len(K_Y_notcolumn[0][0])):
                for d in range(                        len(K_Y_notcolumn[0][0][0])):
                    for e in range(                    len(K_Y_notcolumn[0][0][0][0])):
                        for f in range(                len(K_Y_notcolumn[0][0][0][0][0])):
                            for g in range(            len(K_Y_notcolumn[0][0][0][0][0][0])):
                                for h in range(        len(K_Y_notcolumn[0][0][0][0][0][0][0])):
                                    for i in range(    len(K_Y_notcolumn[0][0][0][0][0][0][0][0])):
                                        for j in range(len(K_Y_notcolumn[0][0][0][0][0][0][0][0][0])):
                                            col[counter][0] = K_Y_notcolumn[j][i][h][g][f][e][d][c][b][a]
                                            counter += 1

    return col


# Calculate K^(-1)*codomain_train using the summation method
def Summation10D(K1, K2, K3, K4, K5, K6, K7, K8, K9, K10, Y_10D_mapper):
    # K1: The covariance matrix of only the a-coordinates
    # K2: The covariance matrix of only the b-coordinates
    # K3: The covariance matrix of only the c-coordinates
    # K4: The covariance matrix of only the d-coordinates
    # K5: The covariance matrix of only the e-coordinates
    # K6: The covariance matrix of only the f-coordinates
    # K7: The covariance matrix of only the g-coordinates
    # K8: The covariance matrix of only the h-coordinates
    # K9: The covariance matrix of only the i-coordinates
    # K10: The covariance matrix of only the j-coordinates
    # Y_10D_mapper: The output of the training data as a tensor

    # Inverting the matrices
    K1_inverse = np.linalg.inv(K1) # We get K_1 inverse
    K2_inverse = np.linalg.inv(K2) # We get K_2 inverse
    K3_inverse = np.linalg.inv(K3) # We get K_3 inverse
    K4_inverse = np.linalg.inv(K4) # We get K_4 inverse
    K5_inverse = np.linalg.inv(K5) # We get K_5 inverse
    K6_inverse = np.linalg.inv(K6) # We get K_6 inverse
    K7_inverse = np.linalg.inv(K7) # We get K_7 inverse
    K8_inverse = np.linalg.inv(K8) # We get K_8 inverse
    K9_inverse = np.linalg.inv(K9) # We get K_9 inverse
    K10_inverse = np.linalg.inv(K10) # We get K_10 inverse

    A = np.zeros((len(K10_inverse),
                  len(Y_10D_mapper[0][0][0][0][0][0][0][0][0]),
                  len(Y_10D_mapper[0][0][0][0][0][0][0][0]),
                  len(Y_10D_mapper[0][0][0][0][0][0][0]),
                  len(Y_10D_mapper[0][0][0][0][0][0]),
                  len(Y_10D_mapper[0][0][0][0][0]),
                  len(Y_10D_mapper[0][0][0][0]),
                  len(Y_10D_mapper[0][0][0]),
                  len(Y_10D_mapper[0][0]),
                  len(Y_10D_mapper[0])))

    B = np.zeros((len(K9_inverse),
                  len(A[0][0][0][0][0][0][0][0][0]),
                  len(A[0][0][0][0][0][0][0][0]),
                  len(A[0][0][0][0][0][0][0]),
                  len(A[0][0][0][0][0][0]),
                  len(A[0][0][0][0][0]),
                  len(A[0][0][0][0]),
                  len(A[0][0][0]),
                  len(A[0][0]),
                  len(A[0])))

    C = np.zeros((len(K8_inverse),
                  len(B[0][0][0][0][0][0][0][0][0]),
                  len(B[0][0][0][0][0][0][0][0]),
                  len(B[0][0][0][0][0][0][0]),
                  len(B[0][0][0][0][0][0]),
                  len(B[0][0][0][0][0]),
                  len(B[0][0][0][0]),
                  len(B[0][0][0]),
                  len(B[0][0]),
                  len(B[0])))

    D = np.zeros((len(K7_inverse),
                  len(C[0][0][0][0][0][0][0][0][0]),
                  len(C[0][0][0][0][0][0][0][0]),
                  len(C[0][0][0][0][0][0][0]),
                  len(C[0][0][0][0][0][0]),
                  len(C[0][0][0][0][0]),
                  len(C[0][0][0][0]),
                  len(C[0][0][0]),
                  len(C[0][0]),
                  len(C[0])))

    E = np.zeros((len(K6_inverse),
                  len(D[0][0][0][0][0][0][0][0][0]),
                  len(D[0][0][0][0][0][0][0][0]),
                  len(D[0][0][0][0][0][0][0]),
                  len(D[0][0][0][0][0][0]),
                  len(D[0][0][0][0][0]),
                  len(D[0][0][0][0]),
                  len(D[0][0][0]),
                  len(D[0][0]),
                  len(D[0])))

    F = np.zeros((len(K5_inverse),
                  len(E[0][0][0][0][0][0][0][0][0]),
                  len(E[0][0][0][0][0][0][0][0]),
                  len(E[0][0][0][0][0][0][0]),
                  len(E[0][0][0][0][0][0]),
                  len(E[0][0][0][0][0]),
                  len(E[0][0][0][0]),
                  len(E[0][0][0]),
                  len(E[0][0]),
                  len(E[0])))

    G = np.zeros((len(K4_inverse),
                  len(F[0][0][0][0][0][0][0][0][0]),
                  len(F[0][0][0][0][0][0][0][0]),
                  len(F[0][0][0][0][0][0][0]),
                  len(F[0][0][0][0][0][0]),
                  len(F[0][0][0][0][0]),
                  len(F[0][0][0][0]),
                  len(F[0][0][0]),
                  len(F[0][0]),
                  len(F[0])))

    H = np.zeros((len(K3_inverse),
                  len(G[0][0][0][0][0][0][0][0][0]),
                  len(G[0][0][0][0][0][0][0][0]),
                  len(G[0][0][0][0][0][0][0]),
                  len(G[0][0][0][0][0][0]),
                  len(G[0][0][0][0][0]),
                  len(G[0][0][0][0]),
                  len(G[0][0][0]),
                  len(G[0][0]),
                  len(G[0])))

    I = np.zeros((len(K2_inverse),
                  len(H[0][0][0][0][0][0][0][0][0]),
                  len(H[0][0][0][0][0][0][0][0]),
                  len(H[0][0][0][0][0][0][0]),
                  len(H[0][0][0][0][0][0]),
                  len(H[0][0][0][0][0]),
                  len(H[0][0][0][0]),
                  len(H[0][0][0]),
                  len(H[0][0]),
                  len(H[0])))

    K_inverse_Y = np.zeros((len(K1_inverse),
                  len(I[0][0][0][0][0][0][0][0][0]),
                  len(I[0][0][0][0][0][0][0][0]),
                  len(I[0][0][0][0][0][0][0]),
                  len(I[0][0][0][0][0][0]),
                  len(I[0][0][0][0][0]),
                  len(I[0][0][0][0]),
                  len(I[0][0][0]),
                  len(I[0][0]),
                  len(I[0])))

    for a in range(len(K10_inverse)):
        for b in range(len(Y_10D_mapper                                     [0][0][0][0][0][0][0][0][0])):
            for c in range(len(Y_10D_mapper                                 [0][0][0][0][0][0][0][0])):
                for d in range(len(Y_10D_mapper                             [0][0][0][0][0][0][0])):
                    for e in range(len(Y_10D_mapper                         [0][0][0][0][0][0])):
                        for f in range(len(Y_10D_mapper                     [0][0][0][0][0])):
                            for g in range(len(Y_10D_mapper                 [0][0][0][0])):
                                for h in range(len(Y_10D_mapper             [0][0][0])):
                                    for i in range(len(Y_10D_mapper         [0][0])):
                                        for j in range(len(Y_10D_mapper     [0])):
                                            for k in range(len(Y_10D_mapper)):
                                                A[a][b][c][d][e][f][g][h][i][j] += \
                                                    K10_inverse[a][k] * Y_10D_mapper[k][j][i][h][g][f][e][d][c][b]
    A = A.T
    for a in range(len(K9_inverse)):
        for b in range(len(A                                     [0][0][0][0][0][0][0][0][0])):
            for c in range(len(A                                 [0][0][0][0][0][0][0][0])):
                for d in range(len(A                             [0][0][0][0][0][0][0])):
                    for e in range(len(A                         [0][0][0][0][0][0])):
                        for f in range(len(A                     [0][0][0][0][0])):
                            for g in range(len(A                 [0][0][0][0])):
                                for h in range(len(A             [0][0][0])):
                                    for i in range(len(A         [0][0])):
                                        for j in range(len(A     [0])):
                                            for k in range(len(A)):
                                                B[a][b][c][d][e][f][g][h][i][j] += \
                                                    K9_inverse[a][k] * A[k][j][i][h][g][f][e][d][c][b]
    B = B.T
    for a in range(len(K8_inverse)):
        for b in range(len(B                                     [0][0][0][0][0][0][0][0][0])):
            for c in range(len(B                                 [0][0][0][0][0][0][0][0])):
                for d in range(len(B                             [0][0][0][0][0][0][0])):
                    for e in range(len(B                         [0][0][0][0][0][0])):
                        for f in range(len(B                     [0][0][0][0][0])):
                            for g in range(len(B                 [0][0][0][0])):
                                for h in range(len(B             [0][0][0])):
                                    for i in range(len(B         [0][0])):
                                        for j in range(len(B     [0])):
                                            for k in range(len(B)):
                                                C[a][b][c][d][e][f][g][h][i][j] += \
                                                    K8_inverse[a][k] * B[k][j][i][h][g][f][e][d][c][b]
    C = C.T
    for a in range(len(K7_inverse)):
        for b in range(len(C                                     [0][0][0][0][0][0][0][0][0])):
            for c in range(len(C                                 [0][0][0][0][0][0][0][0])):
                for d in range(len(C                             [0][0][0][0][0][0][0])):
                    for e in range(len(C                         [0][0][0][0][0][0])):
                        for f in range(len(C                     [0][0][0][0][0])):
                            for g in range(len(C                 [0][0][0][0])):
                                for h in range(len(C             [0][0][0])):
                                    for i in range(len(C         [0][0])):
                                        for j in range(len(C     [0])):
                                            for k in range(len(C)):
                                                D[a][b][c][d][e][f][g][h][i][j] += \
                                                    K7_inverse[a][k] * C[k][j][i][h][g][f][e][d][c][b]
    D = D.T
    for a in range(len(K6_inverse)):
        for b in range(len(D                                     [0][0][0][0][0][0][0][0][0])):
            for c in range(len(D                                 [0][0][0][0][0][0][0][0])):
                for d in range(len(D                             [0][0][0][0][0][0][0])):
                    for e in range(len(D                         [0][0][0][0][0][0])):
                        for f in range(len(D                     [0][0][0][0][0])):
                            for g in range(len(D                 [0][0][0][0])):
                                for h in range(len(D             [0][0][0])):
                                    for i in range(len(D         [0][0])):
                                        for j in range(len(D     [0])):
                                            for k in range(len(D)):
                                                E[a][b][c][d][e][f][g][h][i][j] += \
                                                    K6_inverse[a][k] * D[k][j][i][h][g][f][e][d][c][b]
    E = E.T
    for a in range(len(K5_inverse)):
        for b in range(len(E                                     [0][0][0][0][0][0][0][0][0])):
            for c in range(len(E                                 [0][0][0][0][0][0][0][0])):
                for d in range(len(E                             [0][0][0][0][0][0][0])):
                    for e in range(len(E                         [0][0][0][0][0][0])):
                        for f in range(len(E                     [0][0][0][0][0])):
                            for g in range(len(E                 [0][0][0][0])):
                                for h in range(len(E             [0][0][0])):
                                    for i in range(len(E         [0][0])):
                                        for j in range(len(E     [0])):
                                            for k in range(len(E)):
                                                F[a][b][c][d][e][f][g][h][i][j] += \
                                                    K5_inverse[a][k] * E[k][j][i][h][g][f][e][d][c][b]
    F = F.T
    for a in range(len(K4_inverse)):
        for b in range(len(F                                     [0][0][0][0][0][0][0][0][0])):
            for c in range(len(F                                 [0][0][0][0][0][0][0][0])):
                for d in range(len(F                             [0][0][0][0][0][0][0])):
                    for e in range(len(F                         [0][0][0][0][0][0])):
                        for f in range(len(F                     [0][0][0][0][0])):
                            for g in range(len(F                 [0][0][0][0])):
                                for h in range(len(F             [0][0][0])):
                                    for i in range(len(F         [0][0])):
                                        for j in range(len(F     [0])):
                                            for k in range(len(F)):
                                                G[a][b][c][d][e][f][g][h][i][j] += \
                                                    K4_inverse[a][k] * F[k][j][i][h][g][f][e][d][c][b]
    G = G.T
    for a in range(len(K3_inverse)):
        for b in range(len(G                                     [0][0][0][0][0][0][0][0][0])):
            for c in range(len(G                                 [0][0][0][0][0][0][0][0])):
                for d in range(len(G                             [0][0][0][0][0][0][0])):
                    for e in range(len(G                         [0][0][0][0][0][0])):
                        for f in range(len(G                     [0][0][0][0][0])):
                            for g in range(len(G                 [0][0][0][0])):
                                for h in range(len(G             [0][0][0])):
                                    for i in range(len(G         [0][0])):
                                        for j in range(len(G     [0])):
                                            for k in range(len(G)):
                                                H[a][b][c][d][e][f][g][h][i][j] += \
                                                    K3_inverse[a][k] * G[k][j][i][h][g][f][e][d][c][b]
    H = H.T
    for a in range(len(K2_inverse)):
        for b in range(len(H                                     [0][0][0][0][0][0][0][0][0])):
            for c in range(len(H                                 [0][0][0][0][0][0][0][0])):
                for d in range(len(H                             [0][0][0][0][0][0][0])):
                    for e in range(len(H                         [0][0][0][0][0][0])):
                        for f in range(len(H                     [0][0][0][0][0])):
                            for g in range(len(H                 [0][0][0][0])):
                                for h in range(len(H             [0][0][0])):
                                    for i in range(len(H         [0][0])):
                                        for j in range(len(H     [0])):
                                            for k in range(len(H)):
                                                I[a][b][c][d][e][f][g][h][i][j] += \
                                                    K2_inverse[a][k] * H[k][j][i][h][g][f][e][d][c][b]
    I = I.T
    for a in range(len(K1_inverse)):
        for b in range(len(I                                     [0][0][0][0][0][0][0][0][0])):
            for c in range(len(I                                 [0][0][0][0][0][0][0][0])):
                for d in range(len(I                             [0][0][0][0][0][0][0])):
                    for e in range(len(I                         [0][0][0][0][0][0])):
                        for f in range(len(I                     [0][0][0][0][0])):
                            for g in range(len(I                 [0][0][0][0])):
                                for h in range(len(I             [0][0][0])):
                                    for i in range(len(I         [0][0])):
                                        for j in range(len(I     [0])):
                                            for k in range(len(I)):
                                                K_inverse_Y[a][b][c][d][e][f][g][h][i][j] += \
                                                    K1_inverse[a][k] * I[k][j][i][h][g][f][e][d][c][b]
    return K_inverse_Y


# Setting up the parameters for the summation method
def PreSummation10D(A, B, C, D, E, F, G, H, I, J, length, variance, Y_10D_mapper, reg1, reg2, reg3, reg4, reg5, reg6,
                    reg7, reg8, reg9, reg10):
    # A: The training data in the a-coordinate
    # B: The training data in the b-coordinate
    # C: The training data in the c-coordinate
    # D: The training data in the d-coordinate
    # E: The training data in the e-coordinate
    # F: The training data in the f-coordinate
    # G: The training data in the g-coordinate
    # H: The training data in the h-coordinate
    # I: The training data in the i-coordinate
    # J: The training data in the j-coordinate
    # length: The length parameter for the exponential squared kernel
    # variance: The variance parameter for the exponential squared kernel
    # Y_10D_mapper: The output of the training data as a tensor
    # reg1: Regularization constant for K1
    # reg2: Regularization constant for K2
    # reg3: Regularization constant for K3
    # reg4: Regularization constant for K4
    # reg5: Regularization constant for K5
    # reg6: Regularization constant for K6
    # reg7: Regularization constant for K7
    # reg8: Regularization constant for K8
    # reg9: Regularization constant for K9
    # reg10: Regularization constant for K10

    # Calculate the matrix of the a-coordinate 
    K1 = Covariance(X1=A.reshape(-1, 1), X2=A.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the b-coordinate 
    K2 = Covariance(X1=B.reshape(-1, 1), X2=B.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the c-coordinate 
    K3 = Covariance(X1=C.reshape(-1, 1), X2=C.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the d-coordinate 
    K4 = Covariance(X1=D.reshape(-1, 1), X2=D.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the e-coordinate 
    K5 = Covariance(X1=E.reshape(-1, 1), X2=E.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the f-coordinate 
    K6 = Covariance(X1=F.reshape(-1, 1), X2=F.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the g-coordinate 
    K7 = Covariance(X1=G.reshape(-1, 1), X2=G.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the h-coordinate 
    K8 = Covariance(X1=H.reshape(-1, 1), X2=H.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the i-coordinate 
    K9 = Covariance(X1=I.reshape(-1, 1), X2=I.reshape(-1, 1), length=length, variance=variance)

    # Calculate the matrix of the j-coordinate
    K10 = Covariance(X1=J.reshape(-1, 1), X2=J.reshape(-1, 1), length=length, variance=variance)

    # Calculate the condition numbers before regularization
    print('Condition number for K1')
    print('%.4E' % Decimal(np.linalg.cond(K1)))
    print('Condition number for K2')
    print('%.4E' % Decimal(np.linalg.cond(K2)))
    print('Condition number for K3')
    print('%.4E' % Decimal(np.linalg.cond(K3)))
    print('Condition number for K4')
    print('%.4E' % Decimal(np.linalg.cond(K4)))
    print('Condition number for K5')
    print('%.4E' % Decimal(np.linalg.cond(K5)))
    print('Condition number for K6')
    print('%.4E' % Decimal(np.linalg.cond(K6)))
    print('Condition number for K7')
    print('%.4E' % Decimal(np.linalg.cond(K7)))
    print('Condition number for K8')
    print('%.4E' % Decimal(np.linalg.cond(K8)))
    print('Condition number for K9')
    print('%.4E' % Decimal(np.linalg.cond(K9)))
    print('Condition number for K10')
    print('%.4E' % Decimal(np.linalg.cond(K10)))

    # Regularization section
    K1 = K1 + reg1 * np.identity(len(A))
    K2 = K2 + reg2 * np.identity(len(B))
    K3 = K3 + reg3 * np.identity(len(C))
    K4 = K4 + reg4 * np.identity(len(D))
    K5 = K5 + reg5 * np.identity(len(E))
    K6 = K6 + reg6 * np.identity(len(F))
    K7 = K7 + reg7 * np.identity(len(G))
    K8 = K8 + reg8 * np.identity(len(H))
    K9 = K9 + reg9 * np.identity(len(I))
    K10 = K10 + reg10 * np.identity(len(J))

    # Calculate the condition numbers after regularization
    print('Condition numbers after regularization')
    print('Condition number for K1')
    print('%.4E' % Decimal(np.linalg.cond(K1)))
    print('Condition number for K2')
    print('%.4E' % Decimal(np.linalg.cond(K2)))
    print('Condition number for K3')
    print('%.4E' % Decimal(np.linalg.cond(K3)))
    print('Condition number for K4')
    print('%.4E' % Decimal(np.linalg.cond(K4)))
    print('Condition number for K5')
    print('%.4E' % Decimal(np.linalg.cond(K5)))
    print('Condition number for K6')
    print('%.4E' % Decimal(np.linalg.cond(K6)))
    print('Condition number for K7')
    print('%.4E' % Decimal(np.linalg.cond(K7)))
    print('Condition number for K8')
    print('%.4E' % Decimal(np.linalg.cond(K8)))
    print('Condition number for K9')
    print('%.4E' % Decimal(np.linalg.cond(K9)))
    print('Condition number for K10')
    print('%.4E' % Decimal(np.linalg.cond(K10)))

    # Obtain K^(-1)*Y using the summation method
    K_InvY = Summation10D(K1=K1, K2=K2, K3=K3, K4=K4, K5=K5, K6=K6, K7=K7, K8=K8, K9=K9, K10=K10, Y_10D_mapper=Y_10D_mapper)

    return K_InvY

# The output of the training data in the form of a tensor
def OutputGridMap10D(A, B, C, D, E, F, G, H, I, J):
    # A: The training data in the a-coordinate
    # B: The training data in the b-coordinate
    # C: The training data in the c-coordinate
    # D: The training data in the d-coordinate
    # E: The training data in the e-coordinate
    # F: The training data in the f-coordinate
    # G: The training data in the g-coordinate
    # H: The training data in the h-coordinate
    # I: The training data in the i-coordinate
    # J: The training data in the j-coordinate

    Y_10D_mapper = np.zeros((len(A), len(B), len(C), len(D), len(E), len(F), len(G), len(H), len(I), len(J)))
    for a in range(len(A)):  # a coordinate
        for b in range(len(B)):  # b coordinate
            for c in range(len(C)):  # c coordinate
                for d in range(len(D)):  # d coordinate
                    for e in range(len(E)):  # e coordinate
                        for f in range(len(F)):  # f coordinate
                            for g in range(len(G)):  # g coordinate
                                for h in range(len(H)):  # h coordinate
                                    for i in range(len(I)):  # i coordinate
                                        for j in range(len(J)):  # j coordinate
                                            Y_10D_mapper[a][b][c][d][e][f][g][h][i][j] = Output10D(A=A[a], B=B[b],
                                                                                                   C=C[c], D=D[d],
                                                                                                   E=E[e], F=F[f],
                                                                                                   G=G[g], H=H[h],
                                                                                                   I=I[i], J=J[j])
    return Y_10D_mapper


# The posterior predictive of the 10-D case caculates the mean of the condtional mutlinormal distribution
def Posterior_predictive10D(A, B, C, D, E, F, G, H, I, J, domain_test, domain_train, length, variance, Y_10D_mapper,
                            reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10):
    # A: The training data in the a-coordinate
    # B: The training data in the b-coordinate
    # C: The training data in the c-coordinate
    # D: The training data in the d-coordinate
    # E: The training data in the e-coordinate
    # F: The training data in the f-coordinate
    # G: The training data in the g-coordinate
    # H: The training data in the h-coordinate
    # I: The training data in the i-coordinate
    # J: The training data in the j-coordinate
    # domain_test: The test data
    # domain_train: The training data
    # length: The length parameter of the exponential squared kernel
    # variance: The variance of the exponential squared kernel
    # Y_10D_mapper: The output of the training data with four indices
    # reg1: Regularization constant for K1
    # reg2: Regularization constant for K2
    # reg3: Regularization constant for K3
    # reg4: Regularization constant for K4
    # reg5: Regularization constant for K5
    # reg6: Regularization constant for K6
    # reg7: Regularization constant for K7
    # reg8: Regularization constant for K8
    # reg9: Regularization constant for K9
    # reg10: Regularization constant for K10

    # Use the summation method to find K^(-1)*codomain_train
    K_Y_notcolumn = PreSummation10D(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, J=J, length=length, variance=variance,
                                    Y_10D_mapper=Y_10D_mapper, reg1=reg1, reg2=reg2, reg3=reg3, reg4=reg4, reg5=reg5,
                                    reg6=reg6, reg7=reg7, reg8=reg8, reg9=reg9, reg10=reg10)

    # Reshape the matrix to a column vector to ensure valid matrix multiplication with the matrix K*
    K_InvY_shaped = ColumnGenerator10D(K_Y_notcolumn=K_Y_notcolumn)

    # Calculate the mean using a summation method to reduce the space complexity increases the time complexity
    mu = Mean10D(domain_test=domain_test, domain_train=domain_train, length=length, variance=variance, K_Y=K_InvY_shaped)

    # Return the mean of the multinormal distribution
    return mu


# Calculates the output of the training and test data given the formula
def Output10DVector(domain):
    # domain: The training data or test data

    output = np.zeros((len(domain), 1))
    for i in range(len(domain)):
        output[i][0] = Output10D(A=domain[i][0], B=domain[i][1], C=domain[i][2], D=domain[i][3], E=domain[i][4],
                                 F=domain[i][5], G=domain[i][6], H=domain[i][7], I=domain[i][8], J=domain[i][9])

    return output


# Calculates the output of a given formula
def Output10D(A, B, C, D, E, F, G, H, I, J):
    # A: The a-coordinates that will be passed into the formula
    # B: The b-coordinates that will be passed into the formula
    # C: The c-coordinates that will be passed into the formula
    # D: The d-coordinates that will be passed into the formula
    # E: The e-coordinates that will be passed into the formula
    # F: The f-coordinates that will be passed into the formula
    # G: The g-coordinates that will be passed into the formula
    # H: The h-coordinates that will be passed into the formula
    # I: The i-coordinates that will be passed into the formula
    # J: The j-coordinates that will be passed into the formula

    # 10-D Morse potential function
    D_e = 37255
    a = 1.8677
    r_e = 1.275
    morse_potential = D_e * math.pow(1 - math.exp(-a * (A - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (B - r_e)), 2)\
                    + D_e * math.pow(1 - math.exp(-a * (C - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (D - r_e)), 2)\
                    + D_e * math.pow(1 - math.exp(-a * (E - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (F - r_e)), 2)\
                    + D_e * math.pow(1 - math.exp(-a * (G - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (H - r_e)), 2)\
                    + D_e * math.pow(1 - math.exp(-a * (I - r_e)), 2) + D_e * math.pow(1 - math.exp(-a * (J - r_e)), 2)\

    return morse_potential


# Returns the length parameter and variance parameter for the exponential squared kernel
def HyperParameters10D():
    length = 1
    variance = 1
    reg1 = 0
    reg2 = 0
    reg3 = 0
    reg4 = 0
    reg5 = 0
    reg6 = 0
    reg7 = 0
    reg8 = 0
    reg9 = 0
    reg10 = 0

    return length, variance, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10


# The test data in each coordinate
def TestPoints10D():
    ra = np.random.normal(loc=1.275, scale=0.09, size=3)
    rb = np.random.normal(loc=1.275, scale=0.09, size=3)
    rc = np.random.normal(loc=1.275, scale=0.09, size=3)
    rd = np.random.normal(loc=1.275, scale=0.09, size=3)
    re = np.random.normal(loc=1.275, scale=0.09, size=3)
    rf = np.random.normal(loc=1.275, scale=0.09, size=3)
    rg = np.random.normal(loc=1.275, scale=0.09, size=3)
    rh = np.random.normal(loc=1.275, scale=0.09, size=3)
    ri = np.random.normal(loc=1.275, scale=0.09, size=3)
    rj = np.random.normal(loc=1.275, scale=0.09, size=3)

    return ra, rb, rc, rd, re, rf, rg, rh, ri, rj


# The training data in each coordinate
def TrainingPoints10D():
    A = np.arange(1, 1.55, 0.11)
    B = np.arange(1, 1.55, 0.11)
    C = np.arange(1, 1.55, 0.11)
    D = np.arange(1, 1.55, 0.11)
    E = np.arange(1, 1.55, 0.11)
    F = np.arange(1, 1.55, 0.11)
    G = np.arange(1, 1.55, 0.11)
    H = np.arange(1, 1.55, 0.11)
    I = np.arange(1, 1.55, 0.11)
    J = np.arange(1, 1.55, 0.11)

    return A, B, C, D, E, F, G, H, I, J


# The Gaussian process for a 10-D problem using the summation method
def GaussianProcess10D():
    # Start the timer to see how long the program takes to calculate the mean
    start_time = time.time()

    # Create the test data in each coordinate
    ra, rb, rc, rd, re, rf, rg, rh, ri, rj = TestPoints10D()

    # Mesh the 10 coordinates of the test data together
    ga, gb, gc, gd, ge, gf, gg, gh, gi, gj = np.meshgrid(ra, rb, rc, rd, re, rf, rg, rh, ri, rj)

    # Create the test data using a tensor grid product
    domain_10D_test = np.c_[ga.ravel(), gb.ravel(), gc.ravel(), gd.ravel(), ge.ravel(), gf.ravel(), gg.ravel(),
                           gh.ravel(), gi.ravel(), gj.ravel()]

    # Create the training data in each coordinate
    A, B, C, D, E, F, G, H, I, J = TrainingPoints10D()

    # Mesh the 10 coordinates of the training data together
    gA, gB, gC, gD, gE, gF, gG, gH, gI, gJ = np.meshgrid(A, B, C, D, E, F, G, H, I, J)

    # Create the training data using a tensor grid product
    domain_10D_train = np.c_[gA.ravel(), gB.ravel(), gC.ravel(), gD.ravel(), gE.ravel(), gF.ravel(), gG.ravel(),
                             gH.ravel(), gI.ravel(), gJ.ravel()]

    # Obtain the length and variance parameters for the exponential squared kernel and regularization constants
    length, variance, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10 = HyperParameters10D()

    # Generate the output of the training data in the form of a tensor
    codomain_10D_mapper = OutputGridMap10D(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, J=J)

    # Obtain the mean of the conditioned multinormal distribution
    mu = Posterior_predictive10D(A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, J=J, domain_test=domain_10D_test,
                                 domain_train=domain_10D_train, length=length, variance=variance,
                                 Y_10D_mapper=codomain_10D_mapper, reg1=reg1, reg2=reg2, reg3=reg3, reg4=reg4, reg5=reg5,
                                 reg6=reg6, reg7=reg7, reg8=reg8, reg9=reg9, reg10=reg10)
    
    # Print how long it took to compute the mean
    print("--- %s seconds to calculate the mean ---" % (time.time() - start_time))

    # Calculate the output of the test data
    predicted = Output10DVector(domain=domain_10D_test)

    # Calculate the RMSE of the mean (observed data) and the output of the test data (predicted data)
    print('The RMSE of the mean:')
    rmse = math.sqrt(mean_squared_error(mu, predicted))
    print(rmse)

    return


########################################################################################################################
# Main Function
########################################################################################################################


# The main function starts the above Gaussian processes
def main():
    GaussianProcess9D()

    return


# This sets the starting function in the program
if __name__ == "__main__":
    main()