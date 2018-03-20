import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
#from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

# Calling Method 
#plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')



def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_features,predictions)*100
    #errors = abs(predictions - test_labels)
    #mape = 100 * np.mean(errors / test_labels)
    #accuracy = 100 - mape
    print('Model Performance')
    #print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = .', accuracy)
    
    return accuracy




def rating_to_stars(rating): 
	rating = int(rating)
	if (rating == 0.0 ):
		return 0.0
	elif (rating > 0 ) and (rating <= 199 ):
		return 1.0
	elif (rating >= 200 ) and (rating <= 299 ):
		return 2.0
	elif (rating >= 300 ) and (rating <= 399 ):
		return 3.0
	elif (rating >= 400 ) and (rating <= 499 ):
		return 4.0
	else:
		return 5.0


# def rating_to_stars(rating): 
# 	rating = int(rating)
# 	if (rating >= 0 ) and (rating <= 149 ):
# 		return 1.0
# 	elif (rating >= 150 ) and (rating <= 249 ):
# 		return 2.0
# 	elif (rating >= 250 ) and (rating <= 349 ):
# 		return 3.0
# 	elif (rating >= 350 ) and (rating <= 449 ):
# 		return 4.0
# 	else:
# 		return 5.0


#DATA PROCESSING
# read data in
df = pd.read_csv("flavors_of_cacao.csv")

#print(df.dtypes)


#modified the column name
df = df.rename(columns={'Company \n(Maker-if known)': 'CompanyName', 'Specific Bean Origin\nor Bar Name': 'BarName', 'Cocoa\nPercent': 'CocoaPercent', 'Company\nLocation': 'CompanyLocation','Bean\nType':'BeanType', 'Broad Bean\nOrigin':'BroadBeanOrigin'})
#drop REF and Review Date
df = df.drop(["REF","Review\nDate"],axis = 1)
#df = df.drop(["REF"],axis = 1)
df = df.dropna(0)
print(df.info())

#TODO:convert string into integers OR float?
df['CocoaPercent'] = df['CocoaPercent'].str.replace('%', '')
#df['CocoaPercent'] = df['CocoaPercent'].str.replace('.', '')
df['CocoaPercent'] = (df['CocoaPercent']).astype(float)


#convert rating to intergers Since we are using classification
df['Rating'] = (df['Rating']* 100).astype(int)
df['Rating'] = df['Rating'].apply(rating_to_stars)
#df['Rating'] = (df['Rating']).astype(float)


#convert to dummies
#company = pd.get_dummies(df['CompanyName'],drop_first=True)
company = pd.get_dummies(df['CompanyName'])
barName = pd.get_dummies(df['BarName'])
companyLocation = pd.get_dummies(df['CompanyLocation'])
beanType = pd.get_dummies(df['BeanType'])
broadBeanOrigin = pd.get_dummies(df['BroadBeanOrigin'])

df = pd.concat([df, company, barName, companyLocation, beanType, broadBeanOrigin], axis = 1)
df.drop(['CompanyName', 'BarName','CompanyLocation', 'BeanType', 'BroadBeanOrigin'], axis = 1, inplace = True )

#drop duplicated columns
df = df.loc[:,~df.columns.duplicated()]


#DEBUG
#df.info()
#print(df.columns)
#print(df.head(10))

#TRAIN MODEL 

#Split data
X = df.drop('Rating', axis = 1) #Features
y = df['Rating']   # Target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

max_features = [1,2,3]
n_estimators = [100,200,300,1000]

param_grid = {'max_features': [1,2, 3],'n_estimators': [100, 200, 300, 1000]}
rfc = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
best_grid = grid_search.best_estimator_
print(best_grid)
# #plot_grid_search(grid_search.cv_results_, n_estimators, max_features, 'N Estimators','Max Features')



# clf = GridSearchCV(clf_,
#             dict(C=Cs,
#                  gamma=Gammas),
#                  cv=2,
#                  pre_dispatch='1*n_jobs',
#                  n_jobs=1)


scores = [x[1] for x in grid_search.grid_scores_]
scores = np.array(scores).reshape(len(max_features), len(n_estimators))

for ind, i in enumerate(max_features):
    plt.plot(n_estimators, scores[ind], label='max_features: ' + str(i))
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Mean score')
plt.show()






# Scale the data to be between -1 and 1
#scaler = StandardScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

#print(X_train)

#TODO: Check randomforest Classifier

#Random forest Classifier without cross validation
# rfc = RandomForestClassifier()
# rfc.fit(X_train, y_train)
# rfc_pred = rfc.predict(X_test)
# print(classification_report(y_test,rfc_pred))
# print("RFC Accuracy:")
# print(accuracy_score(y_test,rfc_pred)*100)

# from sklearn.dummy import DummyClassifier 
# dummy = DummyClassifier(strategy='stratified', random_state = 100, constant = None) 
# dummy.fit(X_train, y_train)  
# y_pred = dummy.predict(X_test)
# print(classification_report(y_test,y_pred))
# print("Baseline Accuracy:")
# print(accuracy_score(y_test,y_pred)*100)

# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# Scale the data to be between -1 and 1
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

#print(X_train[2])



#knn = KNeighborsClassifier()
#k_range = range(1, 51)
#param_grid = dict(n_neighbors = k_range)
#grid = GridSearchCV(knn,param_grid, cv=5, scoring = 'accuracy',n_jobs = -1,verbose = 2)
#grid.fit(X_train,y_train)
#grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
#print(grid_mean_scores)
#best_grid = grid.best_estimator_
#print(best_grid)
# plot the results
# this is identical to the one we generated above
#plt.plot(k_range, grid_mean_scores)
#plt.xlabel('Value of K for KNN')
#plt.ylabel('Cross-Validated Accuracy')
#plt.show()

# knn.fit(X_train, y_train)
# knn_pred = knn.predict(X_test)


# print(classification_report(y_test,knn_pred))
# print("st Accuracy:")
# print(accuracy_score(y_test,knn_pred)*100)


#k_fold = KFold(n_splits = 10, random_state = 2)

#scores = cross_val_score(knn, X, y, scoring ='accuracy',cv = k_fold)
#print("KNN Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#print(scores)


#Random forest Classifier with cross validation
# rfc = RandomForestClassifier(n_estimators = 200)
# k_fold = KFold(n_splits = 10)

# #DEBUG
# #for train_index, test_index in k_fold.split(X):
# 	#print('Train: %s | test: %s' % (train_indices, test_indices))
# 	#X_train, X_test = X[train_index],X[test_index]
# 	#y_train, y_test = y[train_index],y[test_index]

# scores = cross_val_score(rfc, X, y, cv = k_fold)
# print("RfC Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print(scores)


#tuning Number of minimum number of samples required to be at a leaf node
# min_samples_leaf = [1, 2, 4,8,16,32,64,128]
# param_grid = {'min_samples_leaf': min_samples_leaf}
# dt = DecisionTreeClassifier()
# #dt.fit(X_train,y_train)
# #dt_pred = dt.predict(X_test)
# #rf_random = RandomizedSearchCV(estimator = dt, param_distributions = random_grid, n_iter = 9, cv = 10, verbose=2, random_state=42)
# #rf_random.fit(X_train,y_train)


# grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 2)
# grid_search.fit(X_train, y_train)
# best_grid = grid_search.best_estimator_
# print(best_grid)
# print(grid_search.grid_scores_)
# #grid_accuracy = evaluate(best_grid, X_test, y_test)



# dt_pred = best_grid.predict(X_test)
# print(classification_report(y_test,dt_pred))
# print("Accuracy:")
# print(accuracy_score(y_test,dt_pred)*100)

# # #Decision tree with cross validation
# #k_fold = KFold(n_splits = 10)
# #dt = DecisionTreeClassifier(min_sample_leaf = 50,random_state = 42)
# #dt.fit(X_train, y_train)
# #dt_pred = dt.predict(X_test)

# #scores = cross_val_score(dt, X_train, y_train, cv = k_fold)
# #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# #print(scores)

# #Decision tree without cross validation
# #print(classification_report(y_test,dt_pred))
# #print("Accuracy:")
# #print(accuracy_score(y_test,dt_pred)*100)


# #digits = datasets.load_digits()
# #X = digits.data
# #y = digits.target



# ###############################draw picture
# Gammas = [1,2,4,8,16,32,64,128]
# scores = [0.64763,0.60794, 0.64067,0.65947,0.67897,0.68175,0.68872,0.68872]
# #clf = GridSearchCV(estimator = dt, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 2)

# #clf.fit(X_train, y_train)#

# #scores = [x[1] for x in clf.grid_scores_]
# #scores = np.array(scores).reshape(len(Gammas))

# #scores = np.array(scores).reshape(len(Cs), len(Gammas))
# #scores

# plt.plot(Gammas, scores)
# plt.legend()
# plt.xlabel('min_samples_leaf')
# plt.ylabel('Mean Accuracy')
# plt.show()

#param_grid = {
#    'max_features': [1,2, 3],
#    'n_estimators': [100, 200, 300, 1000]
#}
rfc = RandomForestClassifier(n_estimators = 300, max_features = 1)
#knn = KNeighborsClassifier(n_neighbors = 29)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)



print(classification_report(y_test,rfc_pred))
print("rfc Accuracy:")
print(accuracy_score(y_test,rfc_pred)*100)






