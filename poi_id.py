

# !/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

for key in data_dict.keys():
    for value in data_dict[key]:
        print value
    break

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi']
for key in data_dict.keys():
    for value in data_dict[key]:
        if value in features_list:
            continue
        features_list.append(value)
    break

import pprint

pprint.pprint(features_list)
# You will need to use more features

#Removing features that are not required
features_list.remove('email_address')
pprint.pprint(features_list)

# Finding total number of NaN fields in each feature
for feature in features_list:
    cnt = 0
    for key in data_dict.keys():
        if data_dict[key][feature] == 'NaN':
            cnt += 1
    print feature + " -> " + str(cnt)

import matplotlib.pyplot as plt

# Identifying and removing outliers
for feature in features_list:
    maxim = 0
    cnt = 0
    for key in data_dict:
        cnt += 1
        point = data_dict[key][feature]
        if point > maxim and point != 'NaN':
            maxim = point
            name = key
        plt.scatter(point, cnt)
    plt.xlabel(feature)
    plt.show()
    print name
    print maxim
    print "\n ------------------------------------------------------------------------------------ \n"


data_dict.pop('TOTAL')


for feature in features_list:
    maxim = 0
    cnt = 0
    for key in data_dict:
        cnt += 1
        point = data_dict[key][feature]
        if point > maxim and point != 'NaN':
            maxim = point
            name = key
        plt.scatter(point, cnt)
    plt.xlabel(feature)
    plt.show()
    print name
    print maxim



for key in data_dict.keys():
    print key


data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

# Removing more features with too many NaN values
features_list.remove('loan_advances')
features_list.remove('restricted_stock_deferred')
features_list.remove('director_fees')


# Creating features
for key in data_dict.keys():
    try:
        data_dict[key]['ratio_from_person_to_poi'] = float(data_dict[key]['from_person_to_poi']
                                                                   ) / data_dict[key]['from_messages']
    except:
        data_dict[key]['ratio_from_person_to_poi'] = 'NaN'

    try:
        data_dict[key]['ratio_from_poi_to_person'] = float(data_dict[key]['from_poi_to_person']
                                                                   ) / data_dict[key]['to_messages']
    except:
        data_dict[key]['ratio_from_poi_to_person'] = 'NaN'


features_list.append('ratio_from_person_to_poi')
features_list.append('ratio_from_poi_to_person')

# Removing features
features_list.remove('from_messages')
features_list.remove('to_messages')
features_list.remove('from_poi_to_this_person')


pprint.pprint(features_list)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


print len(features)
print len(labels)


print len(features[0])
print labels[0]

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Feature Selection
from sklearn.feature_selection import SelectKBest


selection = SelectKBest(k=9)
features = selection.fit_transform(features, labels)
features_selected = selection.get_support(indices=True)
print selection.scores_


new_featurelist = ['poi']

for index in features_selected:
    new_featurelist.append(features_list[index + 1])

features_list = new_featurelist
print features_list


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(n_estimators=5, max_depth=10)
#clf = AdaBoostClassifier(algorithm='SAMME', n_estimators=5)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
#clf = GaussianNB()


clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn import metrics

print metrics.recall_score(labels_test, pred)
print metrics.accuracy_score(pred, labels_test)
print metrics.precision_score(labels_test, pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


