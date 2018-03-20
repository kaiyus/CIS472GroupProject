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

#DATA PROCESSING
# read data in
df = pd.read_csv("flavors_of_cacao.csv")

#modified the column name
df = df.rename(columns={'Company \n(Maker-if known)': 'CompanyName', 'Specific Bean Origin\nor Bar Name': 'BarName', 'Cocoa\nPercent': 'CocoaPercent', 'Company\nLocation': 'CompanyLocation','Bean\nType':'BeanType', 'Broad Bean\nOrigin':'BroadBeanOrigin'})
#drop REF and Review Date
df = df.drop(["REF","Review\nDate"],axis = 1)
#df = df.drop(["REF"],axis = 1)

#TODO:convert string into integers OR float?
df['CocoaPercent'] = df['CocoaPercent'].str.replace('%', '')
#df['CocoaPercent'] = df['CocoaPercent'].str.replace('.', '')
df['CocoaPercent'] = (df['CocoaPercent']).astype(float)


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

X = df.drop('Rating', axis = 1) #Features
y = df['Rating']   # Target 


from sklearn.dummy import DummyClassifier 
dummy = DummyClassifier(strategy='stratified', random_state = 100, constant = None) 
dummy.fit(X, y)  
y_pred = dummy.predict(X)
print("Accuracy:")
print(accuracy_score(y,y_pred)*100)


