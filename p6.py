#import lib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#load the data
data = pd.read_csv("heart.csv")
print(data.shape)
print(data)

#check for the null data
print(data.isnull().sum())

#feature and target
features = data.drop("output", axis = "columns")
target = data["output"]

#check for cat data

#train and test
x_train, x_test, y_train, y_test = train_test_split(features, target)

#model
model = LogisticRegression(max_iter = 3000)
model.fit(x_train, y_train)

#classification report
cr = classification_report(y_test, model.predict(x_test))
print(cr)

#predict
d = [[57 , 0 , 0  ,  140 , 241 ,  0 , 1 , 123 , 1 , 0.2 , 1 , 0 , 3]]
d = [[ 63 , 1 , 3 , 145 , 233 , 1 , 0 , 150 , 0 , 2.3 , 0 , 0 , 1]]

ans = model.predict(d)

print(ans)