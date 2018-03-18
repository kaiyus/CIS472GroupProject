import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
#from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets



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


#def rating_to_stars(rating): 
#	rating = int(rating)
#	if (rating >= 0 ) and (rating <= 149 ):
#		return 1.0
#	elif (rating >= 150 ) and (rating <= 249 ):
#		return 2.0
#	elif (rating >= 250 ) and (rating <= 349 ):
#		return 3.0
#	elif (rating >= 350 ) and (rating <= 449 ):
#		return 4.0
#	else:
#		return 5.0


#DATA PROCESSING
# read data in
df = pd.read_csv("flavors_of_cacao.csv")

#modified the column name
df = df.rename(columns={'Company \n(Maker-if known)': 'CompanyName', 'Specific Bean Origin\nor Bar Name': 'BarName', 'Cocoa\nPercent': 'CocoaPercent', 'Company\nLocation': 'CompanyLocation','Bean\nType':'BeanType', 'Broad Bean\nOrigin':'BroadBeanOrigin'})
#drop REF and Review Date
df = df.drop(["REF","Review\nDate"],axis = 1)

#TODO:convert string into integers OR float?
df['CocoaPercent'] = df['CocoaPercent'].str.replace('%', '')
#df['CocoaPercent'] = df['CocoaPercent'].str.replace('.', '')
df['CocoaPercent'] = df['CocoaPercent'].astype(float)

#convert rating to intergers Since we are using classification
df['Rating'] = (df['Rating']* 100).astype(int)
df['Rating'] = df['Rating'].apply(rating_to_stars)


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

# Scale the data to be between -1 and 1
#scaler = StandardScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

#print(X_train)

#TODO: Check randomforest classifiter
rfc = RandomForestClassifier(n_estimators=100)
#rfc.fit(X_train, y_train)
#rfc_pred = rfc.predict(X_test)

#digits = datasets.load_digits()
#X_digits = digits.data
#y_digits = digits.target

k_fold = KFold(n_splits = 10)
#for train_index, test_index in k_fold.split(X):
	#print('Train: %s | test: %s' % (train_indices, test_indices))
	#X_train, X_test = X[train_index],X[test_index]
	#y_train, y_test = y[train_index],y[test_index]
#score = [rfc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train, test in k_fold.split(X_digits)]

#print(score)
#crossvalidation = KFold(n=X.shape[0], n_folds=10,
# shuffle=True, random_state=1)
scores = cross_val_score(rfc, X, y, cv = k_fold)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
#print ‘Folds: %i, mean squared error: %.2f std: %.2f’%(len(scores),np.mean(np.abs(scores)),np.std(scores))


#Decision tree
dt = DecisionTreeClassifier(min_samples_leaf=20, random_state=99)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
#dt_pred = cross_val_predict(clf, iris.data, iris.target, cv=10)

#print(classification_report(y_test,rfc_pred))
#print("Accuracy:")
#print(accuracy_score(y_test,rfc_pred)*100)


print(classification_report(y_test,dt_pred))
print("Accuracy:")
print(accuracy_score(y_test,dt_pred)*100)

