# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The Neccessity for Nutrition

# %%
# %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

import seaborn as sns
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# %%
np.seterr(divide = 'ignore')

# %%
cereals = pd.read_csv("../data/cereal.csv")
mcd = pd.read_csv("../data/menu.csv")
starbucks = pd.read_csv("../data/starbucks.csv", encoding= 'unicode_escape')

cereals['mfr'].replace({'K': 1, 'G': 1, 'A': -1, 'N': -1, 'P': -1, 'Q': -1, 'R':-1}, inplace=True)

mcd['Category'].replace({'Beef & Pork': -1, 'Chicken & Fish': -1, 'Breakfast': 1}, inplace=True)
value_list = [1, -1]
mcd = mcd[mcd.Category.isin(value_list)]

starbucks['Type'].replace({'Coffee': -1, 'Espresso': 1}, inplace=True)
starbucks = starbucks.dropna(how='all')

# %%
# Organzizing cereal data
X = []
for i in cereals.itertuples(): 
    X.append(i[4:12])  
    
for count,ele in enumerate(X):
    X[count] = list(ele)
    
X = np.array(X)

Y = []
for i in cereals.mfr:
    Y.append([i])
Y = np.array(Y)

# Organzizing mcd data
X1 = []
for i in mcd.itertuples(): 
    X1.append(i[4:17])  
    
for count,ele in enumerate(X1):
    X1[count] = list(ele)
    
X1 = np.array(X1)

Y1 = []
for i in mcd.Category:
    Y1.append([i])
Y1 = np.array(Y1)

# Organzizing starbucks data
X2 = []
for i in starbucks.itertuples(): 
    X2.append(i[6:15])  
    
for count,ele in enumerate(X2):
    X2[count] = list(ele)
    
X2 = np.array(X2)

Y2 = []
for i in starbucks.Type:
    Y2.append([i])
Y2 = np.array(Y2)

# %%
X_and_Y = np.hstack((X, Y))     # Stack them together for shuffling.            
np.random.shuffle(X_and_Y)      # Shuffle the data points in X_and_Y array

print(X.shape)
print(Y.shape)
print(X_and_Y[0])

X_and_Y1 = np.hstack((X1, Y1))     # Stack them together for shuffling.            
np.random.shuffle(X_and_Y1)      # Shuffle the data points in X_and_Y array

print(X1.shape)
print(Y1.shape)
print(X_and_Y1[0]) 

X_and_Y2 = np.hstack((X2, Y2))     # Stack them together for shuffling.            
np.random.shuffle(X_and_Y2)      # Shuffle the data points in X_and_Y array

print(X2.shape)
print(Y2.shape)
print(X_and_Y2[0]) 


# %% [markdown]
# # Setting up functions

# %%
# Sigmoid function: sigmoid(z) = 1/(1 + e^(-z))
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Judge function: 1(a != b).
def judge(a, b):
    if a != b:
        return 1
    else:
        return 0
    
# Logistic regression classifier.
def f_logistic(x, W, b):
    # x should be a 2-dimensional vector, 
    # W should be a 2-dimensional vector,
    # b should be a scalar.
    # you should return a scalar which is -1 or 1.
    zigma = W.dot(x) + b
    if sigmoid(zigma) >= 0.5:
        return 1
    else:
        return -1
    
# Calculate error given feature vectors X and labels Y.
def calc_error1(X, Y, W, b):
    errors = 0 
    n = len(X)
    
    for (xi, yi) in zip(X, Y):
        errors += judge(yi,f_logistic(xi,W,b))
    
    return errors/n

# Gradient of L(W, b) with respect to W and b.
def grad_L_W_b(X, Y, W, b):
    one = np.ones(len(X))
    P = sigmoid(Y * (X.dot(W) + b * one))

    grad_W = -X.T.dot((one-P) * Y)
    grad_b = -one.T.dot((one-P) * Y)
    return grad_W, grad_b

# Loss L(W, b).
def L_W_b(X, Y, W, b):
    # You should return a scalar.
    one = np.ones(len(X))
    P = sigmoid(Y * (X.dot(W) + b * one))
    
    return -one.T.dot(np.log(P))

# Judge function: 1(a != b). It supports scalar, vector and matrix.
def judge_1(a, b):
    return np.array(a != b).astype(np.float32)

# Judge function: 1(z > 0). It supports scalar, vector and matrix.
def judge_2(z):
    return np.array(z > 0).astype(np.float32)
    
# Rectifier function: (z)_+ = max(0, z). It supports scalar, vector and matrix.
def rectifier(z):
    return np.clip(z, a_min=0, a_max=None)

# Linear SVM classifier.
def f_linear_svm(x, W, b):
    # x should be a 2-dimensional vector, 
    # W should be a 2-dimensional vector,
    # b should be a scalar.
    # you should return a scalar which is -1 or 1. 
    f_svm = W.T.dot(x) + b
    
    if f_svm >= 0:
        return 1
    else:
        return -1
    
# Calculate error given feature vectors X and labels Y.
def calc_error2(X, Y, classifier):
    Y_pred = classifier.predict(X) # Hint: Use classifier.predict()
    e = 1 - accuracy_score(Y_pred, Y) # Hint: Use accuracy_score().
    return e

# Gradient of L(W, b) with respect to W and b.
def grad_L_W_b2(X, Y, W, b, C):
    one = np.ones(len(X))
    a = one - Y * (X.dot(W) + b * one)
    grad_W = W - C * X.T.dot(judge_2(a) * Y)
    grad_b = -C * one.T.dot(judge_2(a) * Y)
    return grad_W, grad_b

# Loss L(W, b).
def L_W_b2(X, Y, W, b, C):
    one = np.ones(len(X))
    a = one - Y * (X.dot(W) + b * one)
    return 1/2 * W.T.dot(W) + C * one.T * rectifier(a)


# %%
def vis(X, Y, W=None, b=None):
    indices_neg1 = (Y == -1).nonzero()[0]
    indices_pos1 = (Y == 1).nonzero()[0]
    plt.scatter(X[:,0][indices_neg1], X[:,1][indices_neg1], 
                c='blue', label='class -1')
    plt.scatter(X[:,0][indices_pos1], X[:,1][indices_pos1], 
                c='red', label='class 1')
    plt.legend()
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    
    if W is not None:
        # w0x0+w1x1+b=0 => x1=-w0x0/w1-b/w1
        w0 = W[0]
        w1 = W[1]
        temp = -w1*np.array([X[:,1].min(), X[:,1].max()])/w0-b/w0
        x0_min = max(temp.min(), X[:,0].min())
        x0_max = min(temp.max(), X[:,1].max())
        x0 = np.linspace(x0_min,x0_max,100)
        x1 = -w0*x0/w1-b/w1
        plt.plot(x0,x1,color='black')

    plt.show()

    
def draw_heatmap(errors, D_list, title):
    plt.figure(figsize = (2,4))
    ax = sns.heatmap(errors, annot=True, fmt='.3f', yticklabels=D_list, xticklabels=[])
    ax.collections[0].colorbar.set_label('error')
    ax.set(ylabel='max depth D')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(title)
    plt.show()


# %%
r_training_data1 = 0
r_testing_data1 = 0

def logreg(X_train, Y_train, X_test, Y_test):        
    learning_rate = 0.001
    iterations  = 10000
    losses = []
    global r_training_data1
    global r_testing_data1

    # Gradient descent algorithm for logistic regression.
    # Step 1. Initialize the parameters W, b.
    W = np.zeros(2) 
    b = 0

    for i in range(iterations):
        # Step 2. Compute the partial derivatives.
        grad_W, grad_b = grad_L_W_b(X_train, Y_train, W, b)
        # Step 3. Update the parameters.
        W = W - learning_rate * grad_W
        b = b - learning_rate * grad_b

        # Track the training losses.
        losses.append(L_W_b(X_train, Y_train, W, b))

    # Show decision boundary, training error and test error.
    r_training_data1 += calc_error1(X_train, Y_train, W, b)
    r_testing_data1 += calc_error1(X_test, Y_test, W, b)
    print('Decision boundary: {:.3f}x0+{:.3f}x1+{:.3f}=0'.format(W[0],W[1],b))
    vis(X_train, Y_train, W, b)
    print('Training error: {}'.format(calc_error1(X_train, Y_train, W, b)))
    vis(X_test, Y_test, W, b)
    print('Test error: {}'.format(calc_error1(X_test, Y_test, W, b)))


# %%
r_training_data2 = 0
r_testing_data2 = 0

def svmf(X_train, Y_train, X_test, Y_test):
    # Some settings.
    learning_rate = 0.0001
    iterations    = 10000
    losses = []

    global r_training_data2
    global r_testing_data2

    # Gradient descent algorithm for linear SVM classifier.
    # Step 1. Initialize the parameters W, b.
    W = np.zeros(2) 
    b = 0
    C = 1000

    for i in range(iterations):
        # Step 2. Compute the partial derivatives.
        grad_W, grad_b = grad_L_W_b2(X_train, Y_train, W, b, C) 
        # Step 3. Update the parameters.
        W = W - learning_rate * grad_W
        b = b - learning_rate * grad_b

        # Track the training losses.
        losses.append(L_W_b2(X_train, Y_train, W, b, C))

    C_list = [0.1, 1, 10, 100, 1000]
    opt_e_training = 1.0   # Optimal training error.
    opt_classifier = None  # Optimal classifier.
    opt_C          = None  # Optimal C.

    for C in C_list:
        # Create a linear SVM classifier.
        # Hints: You can use svm.LinearSVC()
        #        Besides, we use Hinge loss and L2 penalty for weights.
        #        The max iterations should be set to 10000.
        #        The regularization parameter should be set as C.
        #        The other arguments of svm.LinearSVC() are set as default values.
        classifier = svm.LinearSVC(penalty = 'l2', loss = 'hinge', max_iter = 10000, C = C)
    
        # Use the classifier to fit the training set (use X_train, Y_train).
        # Hint: You can use classifier.fit().
        classifier.fit(X_train, Y_train)

        # Obtain the weights and bias from the linear SVM classifier.
        W = classifier.coef_[0]
        b = classifier.intercept_[0]
    
        # Show decision boundary, training error and test error.
        print("Test #1")
        print('C = {}'.format(C))
        print('Decision boundary: {:.3f}x0+{:.3f}x1+{:.3f}=0'.format(W[0],W[1],b))
        vis(X_train, Y_train, W, b)
        e_training = calc_error2(X_train, Y_train, classifier)
        print('Training error: {}'.format(e_training))
        print('\n\n\n')
    
        # Judge if it is the optimal one.
        if e_training < opt_e_training:
            opt_e_training = e_training
            opt_classifier = classifier
            opt_C = C

    r_training_data2 += opt_e_training
    
    # Obtain the weights and bias from the best linear SVM classifier .
    opt_W = opt_classifier.coef_[0]
    opt_b = opt_classifier.intercept_[0]
    print('Best parameter C* = {}'.format(opt_C))
    print('Decision boundary: {:.3f}x0+{:.3f}x1+{:.3f}=0'.format(opt_W[0],opt_W[1],opt_b))
    vis(X_test, Y_test, opt_W, opt_b)
    print('Test error: {}'.format(calc_error2(X_test, Y_test, opt_classifier)))

    r_testing_data2 += calc_error2(X_test, Y_test, opt_classifier)


# %%
r_training_data3 = 0
r_testing_data3 = 0

def trees(X_train,Y_train,X_test,Y_test):
    # Perform grid search for best max depth.
    global r_training_data3 
    global r_testing_data3 

    # 1. Create a decision tree classifier.
    estimator = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 1)
    # 2. Create a grid searcher with cross-validation.
    D_list = [1, 2, 3, 4, 5]
    param_grid = {'max_depth': D_list}
    grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = 5)
    # 3. Use the grid searcher to fit the training set.
    #    Hint: You can simply use .fit() function of the grid searcher.
    grid_search.fit(X_train,Y_train)
    
    # Draw heatmaps of cross-validation errors (in cross-validation).
    cross_val_errors = 1 - grid_search.cv_results_['mean_test_score'].reshape(-1,1)
    draw_heatmap(cross_val_errors, D_list, title='cross-validation error w.r.t D')
    
    # Show the best max depth.
    best_max_depth = grid_search.best_params_
    print("Best max depth D: {}".format(best_max_depth))
    
    # Calculate the training error.
    # Hint: You can use .best_estimator_.predict() to make predictions.
    X_train = grid_search.best_estimator_.predict(X_train)
    train_error = 1 - accuracy_score(X_train, Y_train)
    print("Training error: {}".format(train_error))
    
    # Calculate the test error.
    # Hint: You can use .best_estimator_.predict() to make predictions.
    X_test = grid_search.best_estimator_.predict(X_test)
    test_error = 1 - accuracy_score(X_test, Y_test)
    print("Test error: {}".format(test_error))

    r_training_data3 += train_error
    r_testing_data3 += test_error


# %% [markdown]
# # Dataset #1

# %% [markdown]
# ## Partition 1

# %% [markdown]
# ### Logistic regression classifier

# %% [markdown]
# #### Result #1

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]
    

X_train = X_shuffled[:15][:,[6,0]]
Y_train = Y_shuffled[:15]
X_test  = X_shuffled[15:78][:,[6,0]]
Y_test  = Y_shuffled[15:78]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:15][:,[6,0]]
Y_train = Y_shuffled[:15]
X_test  = X_shuffled[15:78][:,[6,0]]
Y_test  = Y_shuffled[15:78]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:15][:,[6,0]]
Y_train = Y_shuffled[:15]
X_test  = X_shuffled[15:78][:,[6,0]]
Y_test  = Y_shuffled[15:78]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data111 = r_training_data1/3
totalr_testing_data111 = r_testing_data1/3
r_training_data1 = 0
r_testing_data1 = 0
print("Training average :",totalr_training_data111)
print("Training average :",totalr_testing_data111)

# %% [markdown]
# ### Support Vector Machines

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:15][:,[6,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:15] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[15:78][:,[6,0]]
Y_test = Y_shuffled[15:78]

# %%
svmf(X_train, Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:15][:,[6,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:15] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[15:78][:,[6,0]]
Y_test = Y_shuffled[15:78]

# %%
svmf(X_train, Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:15][:,[6,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:15] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[15:78][:,[6,0]]
Y_test = Y_shuffled[15:78]

# %%
svmf(X_train, Y_train, X_test, Y_test)

# %%
totalr_training_data112 = r_training_data2/3
totalr_testing_data112 = r_testing_data2/3
r_training_data2 = 0
r_testing_data2 = 0
print("Training average :",totalr_training_data112)
print("Training average :",totalr_testing_data112)

# %% [markdown]
# ### Decision Tree

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]


X_train = X_shuffled[:15][:,[6,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:15] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[15:78][:,[6,0]]
Y_test = Y_shuffled[15:78]

# %%
trees(X_train,Y_train,X_test,Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]


X_train = X_shuffled[:15][:,[6,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:15] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[15:78][:,[6,0]]
Y_test = Y_shuffled[15:78]

# %%
trees(X_train,Y_train,X_test,Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]


X_train = X_shuffled[:15][:,[6,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:15] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[15:78][:,[6,0]]
Y_test = Y_shuffled[15:78]

# %%
trees(X_train,Y_train,X_test,Y_test)

# %%
totalr_training_data113 = r_training_data3/3
totalr_testing_data113 = r_testing_data3/3
r_training_data3 = 0
r_testing_data3 = 0
print("Training average :",totalr_training_data113)
print("Training average :",totalr_testing_data113)

# %% [markdown]
# ## Partition 2

# %% [markdown]
# ### Logistic regression classifier

# %% [markdown]
# #### Result #1

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:62][:,[6,0]]
Y_train = Y_shuffled[:62]
X_test  = X_shuffled[62:78][:,[6,0]]
Y_test  = Y_shuffled[62:78]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:62][:,[6,0]]
Y_train = Y_shuffled[:62]
X_test  = X_shuffled[62:78][:,[6,0]]
Y_test  = Y_shuffled[62:78]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:62][:,[6,0]]
Y_train = Y_shuffled[:62]
X_test  = X_shuffled[62:78][:,[6,0]]
Y_test  = Y_shuffled[62:78]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data121 = r_training_data1/3
totalr_testing_data121 = r_testing_data1/3
r_training_data1 = 0
r_testing_data1 = 0
print("Training average :",totalr_training_data121)
print("Training average :",totalr_testing_data121)

# %% [markdown]
# ### Support Vector Machines

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:62][:,[6,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:62] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[62:78][:,[6,0]]
Y_test = Y_shuffled[62:78]

# %%
svmf(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:62][:,[6,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:62] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[62:78][:,[6,0]]
Y_test = Y_shuffled[62:78]

# %%
svmf(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:62][:,[6,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:62] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[62:78][:,[6,0]]
Y_test = Y_shuffled[62:78]

# %%
svmf(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data122 = r_training_data2/3
totalr_testing_data122 = r_testing_data2/3
r_training_data2 = 0
r_testing_data2 = 0
print("Training average :",totalr_training_data122)
print("Training average :",totalr_testing_data122)

# %% [markdown]
# ### Decision Tree

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:62][:,[6,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:62] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[62:78][:,[6,0]]
Y_test = Y_shuffled[62:78]

# %%
trees(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:62][:,[6,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:62] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[62:78][:,[6,0]]
Y_test = Y_shuffled[62:78]

# %%
trees(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y) 
X_shuffled = X_and_Y[:,:8]
Y_shuffled = X_and_Y[:,8]

X_train = X_shuffled[:62][:,[6,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:62] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[62:78][:,[6,0]]
Y_test = Y_shuffled[62:78]

# %%
trees(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data123 = r_training_data3/3
totalr_testing_data123 = r_testing_data3/3
r_training_data3 = 0
r_testing_data3 = 0
print("Training average :",totalr_training_data123)
print("Training average :",totalr_testing_data123)

# %%
print("Training average Log Regression :",totalr_training_data111)
print("Training average SVM :",totalr_training_data112)
print("Training average Decision Tree :",totalr_training_data113)

print("Testing average Log Regression :",totalr_testing_data111)
print("Testing average SVM :",totalr_testing_data112)
print("Testing average Decision Tree :",totalr_testing_data113)

print(" ")

print("Training average Log Regression :",totalr_training_data121)
print("Training average SVM :",totalr_training_data122)
print("Training average Decision Tree :",totalr_training_data123)

print("Testing average Log Regression :",totalr_testing_data121)
print("Testing average SVM :",totalr_testing_data122)
print("Testing average Decision Tree :",totalr_testing_data123)

# %% [markdown]
# # Dataset #2

# %% [markdown]
# ## Partition 1

# %% [markdown]
# ### Logistic regression classifier

# %% [markdown]
# #### Result #1

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]
    

X_train = X_shuffled[:17][:,[2,0]]
Y_train = Y_shuffled[:17]
X_test  = X_shuffled[17:85][:,[2,0]]
Y_test  = Y_shuffled[17:85]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]
    

X_train = X_shuffled[:17][:,[2,0]]
Y_train = Y_shuffled[:17]
X_test  = X_shuffled[17:85][:,[2,0]]
Y_test  = Y_shuffled[17:85]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]
    

X_train = X_shuffled[:17][:,[2,0]]
Y_train = Y_shuffled[:17]
X_test  = X_shuffled[17:85][:,[2,0]]
Y_test  = Y_shuffled[17:85]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data211 = r_training_data1/3
totalr_testing_data211 = r_testing_data1/3
r_training_data1 = 0
r_testing_data1 = 0
print("Training average :",totalr_training_data211)
print("Training average :",totalr_testing_data211)

# %% [markdown]
# ### Support Vector Machines

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:17][:,[2,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:17] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[17:85][:,[2,0]]
Y_test = Y_shuffled[17:85]

# %%
svmf(X_train, Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:17][:,[2,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:17] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[17:85][:,[2,0]]
Y_test = Y_shuffled[17:85]

# %%
svmf(X_train, Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:17][:,[2,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:17] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[17:85][:,[2,0]]
Y_test = Y_shuffled[17:85]

# %%
svmf(X_train, Y_train, X_test, Y_test)

# %%
totalr_training_data212 = r_training_data2/3
totalr_testing_data212 = r_testing_data2/3
r_training_data2 = 0
r_testing_data2 = 0
print("Training average :",totalr_training_data212)
print("Training average :",totalr_testing_data212)

# %% [markdown]
# ### Decision Tree

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:17][:,[2,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:17] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[17:85][:,[2,0]]
Y_test = Y_shuffled[17:85]

# %%
trees(X_train,Y_train,X_test,Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:17][:,[2,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:17] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[17:85][:,[2,0]]
Y_test = Y_shuffled[17:85]

# %%
trees(X_train,Y_train,X_test,Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:17][:,[2,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:17] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[17:85][:,[2,0]]
Y_test = Y_shuffled[17:85]

# %%
trees(X_train,Y_train,X_test,Y_test)

# %%
totalr_training_data213 = r_training_data3/3
totalr_testing_data213 = r_testing_data3/3
r_training_data3 = 0
r_testing_data3 = 0
print("Training average :",totalr_training_data213)
print("Training average :",totalr_testing_data213)

# %% [markdown]
# ## Partition 2

# %% [markdown]
# ### Logistic regression classifier

# %% [markdown]
# #### Result #1

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]
    
X_train = X_shuffled[:67][:,[2,0]]
Y_train = Y_shuffled[:67]
X_test  = X_shuffled[67:85][:,[2,0]]
Y_test  = Y_shuffled[67:85]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]
    
X_train = X_shuffled[:67][:,[2,0]]
Y_train = Y_shuffled[:67]
X_test  = X_shuffled[67:85][:,[2,0]]
Y_test  = Y_shuffled[67:85]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]
    
X_train = X_shuffled[:67][:,[2,0]]
Y_train = Y_shuffled[:67]
X_test  = X_shuffled[67:85][:,[2,0]]
Y_test  = Y_shuffled[67:85]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data221 = r_training_data1/3
totalr_testing_data221 = r_testing_data1/3
r_training_data1 = 0
r_testing_data1 = 0
print("Training average :",totalr_training_data221)
print("Training average :",totalr_testing_data221)

# %% [markdown]
# ### Support Vector Machines

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:67][:,[2,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:67] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[67:85][:,[2,0]]
Y_test = Y_shuffled[67:85]

# %%
svmf(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:67][:,[2,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:67] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[67:85][:,[2,0]]
Y_test = Y_shuffled[67:85]

# %%
svmf(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:67][:,[2,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:67] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[67:85][:,[2,0]]
Y_test = Y_shuffled[67:85]

# %%
svmf(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data222 = r_training_data2/3
totalr_testing_data222 = r_testing_data2/3
r_training_data2 = 0
r_testing_data2 = 0
print("Training average :",totalr_training_data222)
print("Training average :",totalr_testing_data222)

# %% [markdown]
# ### Decision Tree

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:67][:,[2,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:67] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[67:85][:,[2,0]]
Y_test = Y_shuffled[67:85]

# %%
trees(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:67][:,[2,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:67] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[67:85][:,[2,0]]
Y_test = Y_shuffled[67:85]

# %%
trees(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y1) 
X_shuffled = X_and_Y1[:,:8]
Y_shuffled = X_and_Y1[:,13]

X_train = X_shuffled[:67][:,[2,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:67] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[67:85][:,[2,0]]
Y_test = Y_shuffled[67:85]

# %%
trees(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data223 = r_training_data3/3
totalr_testing_data223 = r_testing_data3/3
r_training_data3 = 0
r_testing_data3 = 0
print("Training average :",totalr_training_data223)
print("Training average :",totalr_testing_data223)

# %%
print("Training average Log Regression :",totalr_training_data211)
print("Training average SVM :",totalr_training_data212)
print("Training average Decision Tree :",totalr_training_data213)

print("Testing average Log Regression :",totalr_testing_data211)
print("Testing average SVM :",totalr_testing_data212)
print("Testing average Decision Tree :",totalr_testing_data213)

print(" ")

print("Training average Log Regression :",totalr_training_data221)
print("Training average SVM :",totalr_training_data222)
print("Training average Decision Tree :",totalr_training_data223)

print("Testing average Log Regression :",totalr_testing_data221)
print("Testing average SVM :",totalr_testing_data222)
print("Testing average Decision Tree :",totalr_testing_data223)

# %% [markdown]
# # Dataset #3

# %% [markdown]
# ## Partition 1

# %% [markdown]
# ### Logistic regression classifier

# %% [markdown]
# #### Result #1

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]
    

X_train = X_shuffled[:92][:,[7,0]]
Y_train = Y_shuffled[:92]
X_test = X_shuffled[92:463][:,[7,0]]
Y_test = Y_shuffled[92:463]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]
    

X_train = X_shuffled[:92][:,[7,0]]
Y_train = Y_shuffled[:92]
X_test = X_shuffled[92:463][:,[7,0]]
Y_test = Y_shuffled[92:463]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]
    

X_train = X_shuffled[:92][:,[7,0]]
Y_train = Y_shuffled[:92]
X_test = X_shuffled[92:463][:,[7,0]]
Y_test = Y_shuffled[92:463]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data311 = r_training_data1/3
totalr_testing_data311 = r_testing_data1/3
r_training_data1 = 0
r_testing_data1 = 0
print("Training average :",totalr_training_data311)
print("Training average :",totalr_testing_data311)

# %% [markdown]
# ### Support Vector Machines

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:92][:,[7,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:92] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[92:463][:,[7,0]]
Y_test = Y_shuffled[92:463]

# %%
svmf(X_train, Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:92][:,[7,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:92] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[92:463][:,[7,0]]
Y_test = Y_shuffled[92:463]

# %%
svmf(X_train, Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:92][:,[7,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:92] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[92:463][:,[7,0]]
Y_test = Y_shuffled[92:463]

# %%
svmf(X_train, Y_train, X_test, Y_test)

# %%
totalr_training_data312 = r_training_data2/3
totalr_testing_data312 = r_testing_data2/3
r_training_data2 = 0
r_testing_data2 = 0
print("Training average :",totalr_training_data312)
print("Training average :",totalr_testing_data312)

# %% [markdown]
# ### Decision Tree

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:92][:,[7,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:92] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[92:463][:,[7,0]]
Y_test = Y_shuffled[92:463]

# %%
trees(X_train,Y_train,X_test,Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:92][:,[7,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:92] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[92:463][:,[7,0]]
Y_test = Y_shuffled[92:463]

# %%
trees(X_train,Y_train,X_test,Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:92][:,[7,0]] 
X_train = np.delete(X_train, 11, axis=0) 
Y_train = Y_shuffled[:92] 
Y_train = np.delete(Y_train, 11, axis=0) 
X_test = X_shuffled[92:463][:,[7,0]]
Y_test = Y_shuffled[92:463]

# %%
trees(X_train,Y_train,X_test,Y_test)

# %%
totalr_training_data313 = r_training_data3/3
totalr_testing_data313 = r_testing_data3/3
r_training_data3 = 0
r_testing_data3 = 0
print("Training average :",totalr_training_data313)
print("Training average :",totalr_testing_data313)

# %% [markdown]
# ## Partition 2

# %% [markdown]
# ### Logistic regression classifier

# %% [markdown]
# #### Result #1

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:370][:,[7,0]]
Y_train = Y_shuffled[:370]
X_test  = X_shuffled[370:463][:,[7,0]]
Y_test  = Y_shuffled[370:463]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:370][:,[7,0]]
Y_train = Y_shuffled[:370]
X_test  = X_shuffled[370:463][:,[7,0]]
Y_test  = Y_shuffled[370:463]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
# Divide the data points into training set and test set.
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:370][:,[7,0]]
Y_train = Y_shuffled[:370]
X_test  = X_shuffled[370:463][:,[7,0]]
Y_test  = Y_shuffled[370:463]

# %%
logreg(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data321 = r_training_data1/3
totalr_testing_data321 = r_testing_data1/3
r_training_data1 = 0
r_testing_data1 = 0
print("Training average :",totalr_training_data321)
print("Training average :",totalr_testing_data321)

# %% [markdown]
# ### Support Vector Machines

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:370][:,[7,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:370] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[370:463][:,[7,0]]
Y_test = Y_shuffled[370:463]

# %%
svmf(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# ### Result #2

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:370][:,[7,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:370] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[370:463][:,[7,0]]
Y_test = Y_shuffled[370:463]

# %%
svmf(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# ### Result #3

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:370][:,[7,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:370] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[370:463][:,[7,0]]
Y_test = Y_shuffled[370:463]

# %%
svmf(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data322 = r_training_data2/3
totalr_testing_data322 = r_testing_data2/3
r_training_data2 = 0
r_testing_data2 = 0
print("Training average :",totalr_training_data322)
print("Training average :",totalr_testing_data322)

# %% [markdown]
# ### Decision Tree

# %% [markdown]
# #### Result #1

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:370][:,[7,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:370] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[370:463][:,[7,0]]
Y_test = Y_shuffled[370:463]

# %%
trees(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #2

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:370][:,[7,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:370] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[370:463][:,[7,0]]
Y_test = Y_shuffled[370:463]

# %%
trees(X_train,Y_train, X_test, Y_test)

# %% [markdown]
# #### Result #3

# %%
np.random.shuffle(X_and_Y2) 
X_shuffled = X_and_Y2[:,:9]
Y_shuffled = X_and_Y2[:,9]

X_train = X_shuffled[:370][:,[7,0]] 
X_train = np.delete(X_train, 42, axis=0) 
Y_train = Y_shuffled[:370] 
Y_train = np.delete(Y_train, 42, axis=0) 
X_test = X_shuffled[370:463][:,[7,0]]
Y_test = Y_shuffled[370:463]

# %%
trees(X_train,Y_train, X_test, Y_test)

# %%
totalr_training_data323 = r_training_data3/3
totalr_testing_data323 = r_testing_data3/3
r_training_data3 = 0
r_testing_data3 = 0
print("Training average :",totalr_training_data323)
print("Training average :",totalr_testing_data323)

# %%
print("Training average Log Regression :",totalr_training_data311)
print("Training average SVM :",totalr_training_data312)
print("Training average Decision Tree :",totalr_training_data313)

print("Testing average Log Regression :",totalr_testing_data311)
print("Testing average SVM :",totalr_testing_data312)
print("Testing average Decision Tree :",totalr_testing_data313)

print(" ")

print("Training average Log Regression :",totalr_training_data321)
print("Training average SVM :",totalr_training_data322)
print("Training average Decision Tree :",totalr_training_data323)

print("Testing average Log Regression :",totalr_testing_data321)
print("Testing average SVM :",totalr_testing_data322)
print("Testing average Decision Tree :",totalr_testing_data323)

# %%
testing_average11 = (totalr_testing_data111 + totalr_testing_data211 + totalr_testing_data311)/3
testing_average12 = (totalr_testing_data112 + totalr_testing_data212 + totalr_testing_data312)/3
testing_average13 = (totalr_testing_data113 + totalr_testing_data213 + totalr_testing_data313)/3

testing_average21 = (totalr_testing_data121 + totalr_testing_data221 + totalr_testing_data321)/3
testing_average22 = (totalr_testing_data122 + totalr_testing_data222 + totalr_testing_data322)/3
testing_average23 = (totalr_testing_data123 + totalr_testing_data223 + totalr_testing_data323)/3

# %%
data1 = {'Classifier': ['Logical Regression', 'SVM', 'Decision Tree'],
        'A (20/80) Training Accuracy': [totalr_training_data111, totalr_training_data112, totalr_training_data113],
        'A (20/80) Testing Accuracy': [totalr_testing_data111, totalr_testing_data112, totalr_testing_data113],
        'B (20/80) Training Accuracy': [totalr_training_data211, totalr_training_data212, totalr_training_data213],
        'B (20/80) Testing Accuracy': [totalr_testing_data211, totalr_testing_data212, totalr_testing_data213],
        'C (20/80) Training Accuracy': [totalr_training_data311, totalr_training_data312, totalr_training_data313],
        'C (20/80) Testing Accuracy': [totalr_testing_data311, totalr_testing_data312, totalr_testing_data313],
        'Average Testing Accuracy': [testing_average11, testing_average12, testing_average13]}

data2 = {'Classifier': ['Logical Regression', 'SVM', 'Decision Tree'],
        'A (80/20) Training Accuracy': [totalr_training_data121, totalr_training_data122, totalr_training_data123],
        'A (80/20) Testing Accuracy': [totalr_testing_data121, totalr_testing_data122, totalr_testing_data123],
        'B (80/20) Training Accuracy': [totalr_training_data221, totalr_training_data222, totalr_training_data223],
        'B (80/20) Testing Accuracy': [totalr_testing_data221, totalr_testing_data222, totalr_testing_data223],
        'C (80/20) Training Accuracy': [totalr_training_data321, totalr_training_data322, totalr_training_data323],
        'C (80/20) Testing Accuracy': [totalr_testing_data321, totalr_testing_data322, totalr_testing_data323],
        'Average Testing Accuracy': [testing_average21, testing_average22, testing_average23]}

df = pd.DataFrame (data1, columns = ['Classifier','A (20/80) Training Accuracy', 'A (20/80) Testing Accuracy', 'B (20/80) Training Accuracy',
                                    'B (20/80) Testing Accuracy', 'C (20/80) Training Accuracy', 'C (20/80) Testing Accuracy',
                                    'Average Testing Accuracy'])
df2 = pd.DataFrame (data2, columns = ['Classifier','A (80/20) Training Accuracy', 'A (80/20) Testing Accuracy', 'B (80/20) Training Accuracy',
                                    'B (80/20) Testing Accuracy', 'C (80/20) Training Accuracy', 'C (80/20) Testing Accuracy',
                                    'Average Testing Accuracy'])

# %%
df

# %%
df2

# %%
