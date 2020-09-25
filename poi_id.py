#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat
from tester import dump_classifier_and_data,test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# The features are selected based on the later exploration in the code
features_list = ["poi","bonus_by_salary","to_poi_by_non_poi", "from_poi_by_non_poi"]

# trial of newly created features
# features_list = ["poi","bonus","to_poi_by_non_poi", "from_poi_by_non_poi"]
# features_list = ["poi","bonus_by_salary","from_this_person_to_poi", "to_poi_by_non_poi"]
# features_list = ["poi","bonus_by_salary","from_poi_by_non_poi", "from_poi_to_this_person"]

# features_list = ["poi", "bonus_by_salary", "from_this_person_to_poi", "to_poi_by_non_poi", "shared_receipt_with_poi"]
# features_list = ["poi", "bonus_by_salary", "from_this_person_to_poi", "from_poi_to_this_person"]

### Load the dictionary containing the dataset
# with open("my_dataset.pkl", "r") as data_file:
with open("my_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



print "Initial Length of data: ", len(data_dict)

poi=[]
with open("my_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
poi = []
for employee in data_dict:
    for feature, value in data_dict[employee].items():
        if feature == 'poi':
            poi.append(value)

print "Total poi:", sum(poi)

total_poi = 0
non_poi = 0

### Task 2: Remove outliers
for k,d in data_dict.iteritems():
    if d['salary']>9999999:
        if not d['salary'] == 'NaN':
            print k, "-Invalid Key removed."
            data_dict.pop(k,0)
            break

print "after removing outlier, length: ", len(data_dict)
print "total features used:", len(features_list)


# Check if string is 'NaN'
def isNanString(val):
    if val=='NaN':
        return True
    else:
        return False


# NaN count in every feature_list
nanInToMsg = 0
nanInFromPoiToMsg = 0
nanInFromMsg = 0
nanInFromMsgToPoi = 0
nanInSalary = 0
nanInBonus = 0
for key,val in data_dict.iteritems():
    if isNanString(val['to_messages']):
        nanInToMsg+=1
        data_dict[key]['to_messages']=0
    if isNanString(val['from_poi_to_this_person']):
        nanInFromPoiToMsg+=1
        data_dict[key]['from_poi_to_this_person']=0
    if isNanString(val['from_messages']):
        nanInFromMsg+=1
        data_dict[key]['from_messages']=0
    if isNanString(val['from_this_person_to_poi']):
        nanInFromMsgToPoi+=1
        data_dict[key]['from_this_person_to_poi']=0
    if isNanString(val['bonus']):
        nanInSalary+=1
    if isNanString(val['salary']):
        nanInBonus+=1

print "nan in to msg:",nanInToMsg
print "nan in from poi to this person:",nanInFromPoiToMsg
print "nan in from msg:",nanInFromMsg
print "nan in from this person to poi:",nanInFromMsgToPoi
print "nan in salary:",nanInSalary
print "nan in bonus:",nanInBonus

# Feature selection
import numpy
import matplotlib.pyplot as plt
# Checking for unusual salary bonus outliers. 
features=['poi','salary','bonus']
salary_bonus_check = featureFormat(data_dict, features)
salary_bonus_check = numpy.array(salary_bonus_check).T
plt.scatter( salary_bonus_check[1], salary_bonus_check[2], c=salary_bonus_check[0])
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

# Checking for outliers in from message
features=['poi','from_this_person_to_poi','from_messages']
poi_non_poi_message_check = featureFormat(data_dict, features)
poi_non_poi_message_check = numpy.array(poi_non_poi_message_check).T
plt.scatter( poi_non_poi_message_check[2], poi_non_poi_message_check[1], c=poi_non_poi_message_check[0])
plt.xlabel("From him to others")
plt.ylabel("From him to POI")
plt.show()

# Checking for outliers in to message
features=['poi','from_poi_to_this_person','to_messages']
poi_non_poi_message_check = featureFormat(data_dict, features)
poi_non_poi_message_check = numpy.array(poi_non_poi_message_check).T
plt.scatter( poi_non_poi_message_check[2], poi_non_poi_message_check[1], c=poi_non_poi_message_check[0])
plt.xlabel("To him from others")
plt.ylabel("To him from POI")
plt.show()



### Task 3: Create new feature(s)
print "Based on the above findings, 3 New features are created"
for k,d in data_dict.iteritems():
    if not isNanString(d['bonus']) and not isNanString(d['salary']):
        data_dict[k]['bonus_salary_ratio'] = int(d['bonus'])/int(d['salary'])
    else:
        d['bonus_salary_ratio'] = 0

for key,val in data_dict.iteritems():
    if val['bonus'] not in ['NaN',0] and val['salary'] not in ['NaN',0]:
        data_dict[key]['bonus_by_salary']=int(val['bonus'])/int(val['salary'])
    else:
        data_dict[key]['bonus_by_salary']='NaN'
    if val['to_messages'] not in ['NaN',0] and val['from_poi_to_this_person'] not in ['NaN',0]:
        data_dict[key]['to_poi_by_non_poi']=float(val['from_poi_to_this_person'])/float(val['to_messages'])
    else:
        data_dict[key]['to_poi_by_non_poi']= 'NaN'
    if val['from_messages'] not in ['NaN',0] and val['from_this_person_to_poi'] not in ['NaN',0]:
        data_dict[key]['from_poi_by_non_poi']=float(val['from_this_person_to_poi'])/float(val['from_messages'])
    else:
        data_dict[key]['from_poi_by_non_poi']= 'NaN'
print "New feature added 1. bonus_by_salary, 2. to_poi_by_non_poi, 3. from_poi_by_non_poi"


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.


# Trial 2 Naive Bayes

print "Naive Bayes Classifier Testing"
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf, data_dict, features_list, folds = 1000)

# Trial 3 Decision Tree Testing

print "Decision Tree Testing"
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
test_classifier(clf, data_dict, features_list, folds = 1000)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### 

print "Decision Tree Selected  - Tuning decision tree classifier: max_depth=2, min_samples_leaf=5 "
# clf = DecisionTreeClassifier(min_samples_split=3)

clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5)
test_classifier(clf, data_dict, features_list, folds = 1000)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
