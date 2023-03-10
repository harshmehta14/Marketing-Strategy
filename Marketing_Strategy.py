import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Loading Data 
dataset = pd.read_csv('/content/train_data.csv')
X=dataset.drop(["Offer Accepted"],axis=1)
Y=dataset["Offer Accepted"]
dataset.head()
dataset.isna().sum()



# Data Preprocessing

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

col =['offer expiration', 'income_range', 'no_visited_Cold drinks','travelled_more_than_15mins_for_offer', 'Restaur_spend_less_than20','Marital Status', 'restaurant type', 'age','Prefer western over chinese', 'travelled_more_than_25mins_for_offer','travelled_more_than_5mins_for_offer', 'no_visited_bars', 'gender','car', 'restuarant_same_direction_house', 'Cooks regularly','Customer type', 'Qualification', 'is foodie', 'no_Take-aways','Job/Job Industry', 'restuarant_opposite_direction_house','has Children', 'visit restaurant with rating (avg)', 'temperature','Restaur_spend_greater_than20', 'Travel Time', 'Climate','drop location', 'Prefer home food']
imputer_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
    ])

X = pd.DataFrame(imputer_transform.fit_transform(X),columns=col)
X.isna().sum()


off_exp = ['2days','10hours']
inc_range = ['Less than ₹12500', '₹12500 - ₹24999','₹25000 - ₹37499','₹37500 - ₹49999' ,'₹50000 - ₹62499',  '₹62500 - ₹74999','₹75000 - ₹87499', '₹87500 - ₹99999','₹100000 or More' ]
cold_drink_visited = ['never','less1','1~3','4~8', 'gt8']
rest_spend_l20 = ['never','less1','1~3','4~8', 'gt8']
age_cust = [  'below21','21', '26','31' ,'36','41','46' , '50plus' ]
bar_visited =  ['never','less1','1~3','4~8', 'gt8']
qualification = [ 'Some High School','High School Graduate','Some college - no degree','Bachelors degree', 'Associates degree', 'Graduate degree (Masters or Doctorate)']
takeaway = ['never','less1','1~3','4~8', 'gt8']
temp = [40,67,89]
rest_spent_g20 = ['never','less1','1~3','4~8', 'gt8']
traveltime = [7,10,14,18,22]



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


col_ordinal= ['offer expiration','income_range','no_visited_Cold drinks','Restaur_spend_less_than20','age','no_visited_bars','Qualification','no_Take-aways','temperature','Restaur_spend_greater_than20','Travel Time']
# col_onehot= ['Marital Status','restaurant type','gender','car','Customer type','Job/Job Industry','Climate','drop location']
col_onehot= ['Marital Status','restaurant type','gender','Customer type','Job/Job Industry','Climate','drop location']
col_other =['travelled_more_than_15mins_for_offer', 'Prefer western over chinese', 'travelled_more_than_25mins_for_offer','travelled_more_than_5mins_for_offer',  'restuarant_same_direction_house', 'Cooks regularly', 'is foodie',  'restuarant_opposite_direction_house','has Children', 'visit restaurant with rating (avg)', 'Prefer home food', ]
  
ct = ColumnTransformer(
    remainder= 'drop',
    transformers = [
        #  ('sim_imputer',SimpleImputer(strategy='most_frequent',missing_values = np.nan),['offer expiration', 'income_range', 'no_visited_Cold drinks','travelled_more_than_15mins_for_offer', 'Restaur_spend_less_than20','Marital Status', 'restaurant type', 'age','Prefer western over chinese', 'travelled_more_than_25mins_for_offer','travelled_more_than_5mins_for_offer', 'no_visited_bars', 'gender','car', 'restuarant_same_direction_house', 'Cooks regularly','Customer type', 'Qualification', 'is foodie', 'no_Take-aways','Job/Job Industry', 'restuarant_opposite_direction_house','has Children', 'visit restaurant with rating (avg)', 'temperature','Restaur_spend_greater_than20', 'Travel Time', 'Climate','drop location', 'Prefer home food']),    
        ('pass','passthrough',col_other),
        ('ordinal_encoding',OrdinalEncoder(categories=[off_exp, inc_range,cold_drink_visited, rest_spend_l20, age_cust, bar_visited, qualification, takeaway, temp, rest_spent_g20,traveltime]),col_ordinal),
        ('one_hot_encoding',OneHotEncoder(sparse=False),col_onehot )
    ]
)


ct.fit(X)

onehot_cols = (ct.named_transformers_["one_hot_encoding"].get_feature_names_out(col_onehot))
col_name = col_other + col_ordinal + onehot_cols.tolist()
len(col_name)

delete_col = ['Marital Status_Divorced','restaurant type_2 star restaurant','gender_Female',"Customer type_Individual","Job/Job Industry_Architecture & Engineering","Climate_Spring",'drop location_Location A']

X_transformed = ct.transform(X)
X_transformed = pd.DataFrame(X_transformed, columns=col_name)
X_transformed = X_transformed.drop(delete_col,axis=1)
X_transformed.shape
X_transformed.head()

X_transformed.to_csv('test_submission.csv', index=False)

X_transformed.shape

sns.countplot(x='Offer Accepted', data = dataset)



#Training Models

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_transformed,Y,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion = 'gini',max_depth =10, min_samples_split =100,random_state =42)
rfc.fit(X_train,y_train)

y_pred_rfc =rfc.predict(X_test)
y_pred_rfc

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_rfc,digits=4))

#Decsion Tree

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred_dtc =dtc.predict(X_test)
y_pred_dtc

from sklearn.metrics import classification_report
# ?classification_report
print(classification_report(y_test, y_pred_dtc,digits=4))

#Support Vector Machine

from sklearn.svm import SVC
support_vector = SVC()
support_vector.fit(X_train, y_train)
y_pred_sv =support_vector.predict(X_test)
y_pred_sv

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_sv,digits=4))

#KNN

from sklearn.neighbors import KNeighborsClassifier
# ?KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=12)
neigh.fit(X_train, y_train)
y_pred_knn =neigh.predict(X_test)
y_pred_knn

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_knn,digits=4))

#Test Data

test_data = pd.read_csv('/content/test_data.csv')
test_col = test_data.columns
test_data = pd.DataFrame(imputer_transform.fit_transform(test_data),columns=test_col)
test_data = test_data.drop("car",axis=1)

ct.fit(test_data)
onehot_cols = (ct.named_transformers_["one_hot_encoding"].get_feature_names_out(col_onehot))
col_name = col_other + col_ordinal + onehot_cols.tolist()
len(col_name)
test_data_transformed = ct.transform(X)
test_data_transformed = pd.DataFrame(X_transformed, columns=col_name)
test_data_transformed

Y_pred = rfc.predict(test_data_transformed)
Y_pred

# ids = [i for i in range (0,len(Y_pred))]
# submission = pd.DataFrame(columns=['id','Offer Accepted'])
# submission['id'], submission['Offer Accepted'] = ids, Y_pred
# submission.to_csv('submission.csv', index=False)