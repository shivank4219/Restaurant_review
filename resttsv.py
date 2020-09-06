import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download("stopwords")


dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

dataset['Review'][0]
clean_review = []

for i in range(1000):
    Review = dataset['Review'][i]
    Review = re.sub('[^a-zA-Z]',' ',Review)
    Review = Review.lower()
    Review = Review.split()
    Review = [ps.stem(token) for token in Review if not token in stopwords.words('english')]
    Review = ' '.join(Review)
    clean_review.append(Review)
    
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=4000)    
X=cv.fit_transform(clean_review)
X=X.toarray()
y=dataset['Liked'].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0) 

print(cv.get_feature_names())

from sklearn.naive_bayes import GaussianNB
gnv=GaussianNB()
gnv.fit(X_train,y_train)
gnv.score(X_test,y_test)
y_pred=gnv.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)