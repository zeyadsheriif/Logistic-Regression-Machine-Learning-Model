import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('/content/heart.csv')

df.head()
df.describe()
df.shape

X = df.drop(['target'], axis=1)
y = df['target']

df['target'].value_counts()

# df_full['Activity'].value_counts().plot(kind= 'bar')
df['target'].value_counts().plot(kind= 'bar')

df = df.sample(frac=1, random_state=50)

total_rows = df.shape[0]
train_size = int(total_rows*0.8)

train = df[0:train_size]
test = df[train_size:]
print (train_size)

print("Train dataframe shape", train.shape)
print("Test dataframe shape", test.shape)

X_train = train.drop(['target'], axis=1)
y_train = train['target']

X_test = test.drop(['target'], axis=1)
y_test = test['target']

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def model(X_train, y_train, learning_rate, iterations):
    samples = X_train.shape[0]
    features = X_train.shape[1]

    weight = np.zeros((features, 1))
    bias = 0

    cost_list = []

    for i in range(iterations):
        linear_fun = np.dot(X_train, weight) + bias
        activation_fun = sigmoid(linear_fun)

        cost = -(1 / samples) * np.sum(y_train * np.log(activation_fun) + (1 - y_train) * np.log(1 - activation_fun))

        grad_weight = (1 / samples) * np.dot(X_train.T, (activation_fun - y_train))
        grad_bias = (1 / samples) * np.sum(activation_fun - y_train)

        weight = weight - learning_rate * grad_weight
        bias = bias - learning_rate * grad_bias

        cost_list.append(cost)

        if i % (iterations / 10) == 0:
            print("Cost after", i, "iterations is:", cost)

    return weight, bias, cost_list

iterations = 100000
learning_rate = 0.00010
W, B, cost_list = model(X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1), learning_rate=learning_rate, iterations=iterations)

plt.plot(np.arange(iterations), cost_list)
plt.show()

def accuracy(X_test, y_test, weight, bias):

    linear_fun = np.dot(X_test, weight) + bias
    activation_fun = sigmoid(linear_fun)

    activation_fun = activation_fun > 0.5

    activation_fun = np.array(activation_fun, dtype='int64')

    accuracy = (1 - np.sum(np.absolute(activation_fun.flatten() - y_test)) / y_test.shape[0]) * 100

    print("Accuracy:", accuracy, "%")

accuracy(X_test, y_test, W, B)

def precision(y_true, y_pred):
  true_positives = np.sum(np.logical_and(y_true, y_pred))
  predicted_positives = np.sum(y_pred)

  if predicted_positives == 0:
    return 0

  return true_positives / predicted_positives


def recall(y_true, y_pred):
  true_positives = np.sum(np.logical_and(y_true, y_pred))
  actual_positives = np.sum(y_true)

  if actual_positives == 0:
    return 0

  return true_positives / actual_positives


def f1_score(y_true, y_pred):
  p = precision(y_true, y_pred)
  r = recall(y_true, y_pred)

  if p + r == 0:
    return 0

  return 2 * p * r / (p + r)

y_pred = (sigmoid(np.dot(X_test, W) + B) > 0.5).astype(int)
p = precision(y_test, y_pred)
r = recall(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", p)
print("Recall:", r)
print("F1-score:", f1)







