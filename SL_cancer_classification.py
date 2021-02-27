import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import rand
import statsmodels.api as sm
from numpy import log, dot, e, trapz
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/wdbc.data')

"""# **Preprocessing data**"""

df.drop('id',axis=1, inplace=True) # Remove id 
df = df.iloc[:, 0:11] # Only want to use the means
df.isnull().sum() # Null values

# Count target & change M and B to 0 and 1
df['diagnosis'].value_counts()
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Figures for data analysis
df_fig = plt.figure(figsize = (25, 25))
diagnosis = df.iloc[:,0]
df_features = df.iloc[:, 1:11]
i = 0
for feature in df_features.columns:
    plt.subplot(5, 5, i+1)
    i += 1
    sns.distplot(df_features[feature][diagnosis==1], color='orange', label = 'Malignant')
    sns.distplot(df_features[feature][diagnosis==0], color='blue', label = 'Benign')
    plt.legend()
df_fig.tight_layout()
df_fig.suptitle('Breast Cancer Feature Analysis', y=1.03, fontsize = 18)
plt.show()

#Correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True, cmap="Blues")
plt.title("Feature Correlation Matrix", fontsize =20)
plt.show()

#Remove high correlation features
columns=['perimeter_mean','area_mean','concavity_mean','concave points_mean']
df = df.drop(columns, axis=1)

# shuffle & split data (70/30 split)
shuffle_df = df.sample(frac=1)
train_size = int(0.7 * len(df))
train_set = shuffle_df[:train_size]
test_set = shuffle_df[train_size:]

y_train = train_set[['diagnosis']].copy()
y_train_test = train_set[['diagnosis']].copy()
y_train = y_train.to_numpy()
y_train = y_train.T
y_train = y_train[0]

y_test = test_set[['diagnosis']].copy()
y_test = y_test.to_numpy()
y_test = y_test.T
y_test = y_test[0]

x_train = train_set.copy()
x_test = test_set.copy()
x_train.drop("diagnosis", axis=1, inplace=True)
x_test.drop("diagnosis", axis=1, inplace=True)

# Scaling the features
x_train = x_train.apply(lambda x: (x - x.min(axis=0))/(x.max(axis=0)- x.min(axis=0)))
x_test = x_test.apply(lambda x: (x - x.min(axis=0))/(x.max(axis=0)- x.min(axis=0)))

"""# **Logistic Regression Model**"""

weights = np.zeros((df.shape[1],1))

class LogisticRegression:
    
    def sigmoid(self, z): 
      return 1 / (1 + e**(-z))
    
    def cost_function(self, X, y, weights):                 
        z = dot(X, weights)        
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))        
        return -sum(predict_1 + predict_0) / len(X)
        
    def fit(self, X, y, loops=25, lr=0.05):        
        loss = []
        weights = rand(X.shape[1])
        N = len(X)
                 
        for _ in range(loops):        
            # Gradient Descent
            y_pred = self.sigmoid(dot(X, weights))
            weights -= lr * dot(X.T,  y_pred - y) / N            
            # Saving Progress
            loss.append(self.cost_function(X, y, weights)) 
            
        self.weights = weights
        self.loss = loss

    def predict(self, X):        
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        # Returning binary result
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]
         
    def predict_nonb(self, X):        
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        # Returning binary result
        return [self.sigmoid(z)]
        

logreg = LogisticRegression()
logreg.fit(x_train, y_train, loops=10000, lr=0.005)
y_pred = logreg.predict(x_test)
y_nonb_pred = logreg.predict_nonb(x_test) # non binary predictions for better plotting of ROC curve

"""# **Performance Measures**"""

#loss function
plt.style.use('seaborn-whitegrid')
plt.plot(logreg.loss)
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

#Efron's Pseudo R-squared for logistic regression
def pseudo_rsquare(x, y):
    n = len(x)
    sum_residual = np.sum(np.power(x - y, 2))
    variability = np.sum(np.power((x - (np.sum(x) / n)), 2))
    return 1 - (sum_residual / variability)

print(f"Efron's Pseudo R-square is: {pseudo_rsquare(y_test,y_pred)}")

#confusion matrix
y_pred = pd.Series(y_pred, name = 'Actual')
y_test = pd.Series(y_test, name = 'Predicted')

df_confusion = pd.crosstab(y_test, y_pred)

sns.heatmap(df_confusion, annot=True,cmap='Blues')
plt.title("Confusion Matrix", fontsize =22)
plt.show()

#test accuracy
accuracy = (len(y_test) - np.count_nonzero(y_test-y_pred)) / len(y_test)
print(f"Test Accuracy: {round(accuracy*100,2)}%")

"""## Recall, precision and f1 scores"""

# compute true/false negatives & positives
TP = df_confusion[1][1]
TN = df_confusion[0][0]
FP = df_confusion[1][0]
FN = df_confusion[0][1]

recall = TP / (TP+FN)
precision = TP / (TP+FP)
f1_score = 2*((recall*precision) / (recall+precision))
print('recall: ',recall)
print('precision: ',precision)
print('f1 score: ',f1_score)

"""## ROC and AUC"""

#True positive and false negative rates
TPR = recall
TNR = TN / (TN+FP)
FNR = 1-TNR

#get ROC points
thresholds = list(np.array(list(range(0,105,1))) / 100)
roc_value = []

for threshold in thresholds:
  TP = 0; FP = 0; FN = 0; TN = 0

  for i in range(len(y_test)):
    if y_nonb_pred[0][i] >= threshold:
      prediction_class =1
    else: 
      prediction_class =0

    if prediction_class ==1 and y_test[i] ==1:
      TP = TP + 1
    elif y_test[i] ==1 and prediction_class ==0:
      FN = FN + 1
    elif y_test[i] ==0 and prediction_class==1:
      FP = FP + 1
    elif y_test[i]==0 and prediction_class==0:
      TN = TN + 1

  TPR = TP / (TP+FN)
  FPR = FP / (TN+FP)

  roc_value.append([TPR,FPR])

pivot = pd.DataFrame(roc_value, columns = ["x","y"])
pivot["threshold"] = thresholds

#AUC
auc_score = round(abs(np.trapz(pivot.x,pivot.y)),3)
print('auc score: ', auc_score)

#ROC curve
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(pivot.y,pivot.x,label="ROC")
plt.plot([0,1],linestyle='dashed',label="Random guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
fig.text(0.3,0.5,'AUC={0:.3f}'.format(auc_score),fontsize=12)
plt.title("ROC curve", fontsize =20)
plt.show()

"""## Cross validation logistic regression"""

#shuffling data and splitting into input & target dataframe
shuffle_df = df.sample(frac=1)
train_df = shuffle_df.iloc[:,1:] # input
test_df = shuffle_df.iloc[:,0] # target

k = 10
train_folds = np.array_split(train_df, k)
test_folds = np.array_split(test_df,k)

results = [] 

#splitting into training/testing & iterating k times
for i in range(k):
  x_test = x_folds[i] 
  y_test = y_folds[i]
  x_train = x_df.drop(x_test.index)
  y_train = y_df.drop(y_test.index)

  x_train = x_train.apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
  x_test = x_test.apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))      

  logreg_fold = LogisticRegression()
  logreg_fold.fit(x_train, y_train, loops=5000, lr=0.01)
  y_pred = logreg_fold.predict(x_test)

  accuracy = abs((len(y_test)-np.count_nonzero(y_test-y_pred))/len(y_test))
  results.append(accuracy)


  if i == k-1:
    print(f"Test Accuracy for {k}-fold: {round(np.mean(results)*100,2)}%")

"""## Optimisation"""

# f-test for feature selection/ not included in assessment
#This was used to validate our feature selection results below
from sklearn.feature_selection import f_classif, SelectKBest
select = SelectKBest(score_func=f_classif, k='all')
select.fit(x_train, y_train)
scores = select.scores_

feature_f = zip(x_train.columns.values.tolist(), scores)
print(feature_f)

# feedforward feature selection
# exhibits same pattern as f-test feature selection
def feature_selection(x, y, pval):
    first_features = x.columns.tolist()
    rel_features = []
    while len(first_features) > 0:
        features = list(set(first_features) - set(rel_features))
        new_p = pd.Series(index = features)
        for new in features:
            ols = sm.Logit(y,sm.add_constant(x[rel_features+[new]])).fit(disp=0)
            new_p[new] = ols.pvalues[new]
        min_pval = min(new_p)
        if min_pval < pval:
            rel_features.append(new_p.idxmin())
        else:
            break
    print(f"Significant features: {rel_features}")

feature_selection(x_train,y_train,0.05)
