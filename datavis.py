import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("flavors_of_cacao.csv")

#modified the column name
df = df.rename(columns={'Company \n(Maker-if known)': 'CompanyName', 'Specific Bean Origin\nor Bar Name': 'BarName', 'Cocoa\nPercent': 'CocoaPercent', 'Company\nLocation': 'CompanyLocation','Bean\nType':'BeanType', 'Broad Bean\nOrigin':'BroadBeanOrigin'})



#drop REF and Review Date
df = df.drop(["REF","Review\nDate"],axis = 1)
#df = df.drop(["REF"],axis = 1)

# #TODO:convert string into integers OR float?
df['CocoaPercent'] = df['CocoaPercent'].str.replace('%', '')
#df['CocoaPercent'] = df['CocoaPercent'].str.replace('.', '')
df['CocoaPercent'] = (df['CocoaPercent']).astype(float)
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(df['CocoaPercent'], ax=ax)
ax.set_title('Cocoa %, Distribution')
plt.show()


sns.heatmap(df.corr(), cmap='coolwarm')


# #convert rating to intergers Since we are using classification
# df['Rating'] = (df['Rating']* 100).astype(int)
# df['Rating'] = df['Rating'].apply(rating_to_stars)


# #convert to dummies
# #company = pd.get_dummies(df['CompanyName'],drop_first=True)
# company = pd.get_dummies(df['CompanyName'])
# barName = pd.get_dummies(df['BarName'])
# companyLocation = pd.get_dummies(df['CompanyLocation'])
# beanType = pd.get_dummies(df['BeanType'])
# broadBeanOrigin = pd.get_dummies(df['BroadBeanOrigin'])

# df = pd.concat([df, company, barName, companyLocation, beanType, broadBeanOrigin], axis = 1)
# df.drop(['CompanyName', 'BarName','CompanyLocation', 'BeanType', 'BroadBeanOrigin'], axis = 1, inplace = True )

# #drop duplicated columns
# df = df.loc[:,~df.columns.duplicated()]