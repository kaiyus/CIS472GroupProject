import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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
#convert string into integers OR float?
df['CocoaPercent'] = df['CocoaPercent'].str.replace('%', '')
#df['CocoaPercent'] = df['CocoaPercent'].str.replace('.', '')
df['CocoaPercent'] = df['CocoaPercent'].astype(float)

df['Rating'] = (df['Rating']* 100).astype(int)
df['Rating'] = df['Rating'].apply(rating_to_stars)


#convert to dummies #drop REF and Review Date
#company = pd.get_dummies(df['CompanyName'],drop_first=True)
company = pd.get_dummies(df['CompanyName'])
barName = pd.get_dummies(df['BarName'])
companyLocation = pd.get_dummies(df['CompanyLocation'])
beanType = pd.get_dummies(df['BeanType'])
broadBeanOrigin = pd.get_dummies(df['BroadBeanOrigin'])

df = pd.concat([df, company, barName, companyLocation, beanType, broadBeanOrigin], axis = 1)
df.drop(['CompanyName', 'BarName','CompanyLocation', 'BeanType', 'BroadBeanOrigin'], axis = 1, inplace = True )
df = df.loc[:,~df.columns.duplicated()]

#DEBUG
#df.info()
#print(df.columns)

#TRAIN MODEL 

#Split data
X = df.drop('Rating', axis = 1) #Features
Y = df['Rating']   # Target 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=7)

#TODO: Check randomforest classifiter
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, Y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(Y_test,rfc_pred))
print("Accuracy:")
print(accuracy_score(Y_test,rfc_pred)*100)


