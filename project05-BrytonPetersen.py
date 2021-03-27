# %%
"""All of my imports"""

#Basic imports
import pandas as pd 
import altair as alt
import numpy as np
import seaborn as sns

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


# %%
"""Reading in my data to csv"""

url = 'https://github.com/fivethirtyeight/data/raw/master/star-wars-survey/StarWars.csv'

#I use 'latin1' encoding to accomodate for characters not compatible with UTF-8
dat = pd.read_csv(url, encoding = 'latin1', skiprows = 2, header = None)

# Selecting the 1st row that contains all of the column names so I can tidy them up easier
dat_cols = pd.read_csv(url, encoding = 'latin1', nrows = 1)
dat_cols = dat_cols.melt()

# %%
"""Cleaning up the column names"""

#Here we create a dictionary that has the original column names and the ones we want to switch them with 
variables_replace = {
    'Which of the following Star Wars films have you seen\\? Please select all that apply\\.':'seen',
    'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.':'rank',
    'Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.':'view',
    'Do you consider yourself to be a fan of the Star Trek franchise\\?':'star_trek_fan',
    'Do you consider yourself to be a fan of the Expanded Universe\\?\x8cæ':'expanded_fan',
    'Are you familiar with the Expanded Universe\\?':'know_expanded',
    'Have you seen any of the 6 films in the Star Wars franchise\\?':'seen_any',
    'Do you consider yourself to be a fan of the Star Wars film franchise\\?':'star_wars_fans',
    'Which character shot first\\?':'shot_first',
    'Unnamed: \d{1,2}':np.nan,
    ' ':'_',
}
values_replace = {
    'Response':'',
    'Star Wars: Episode ':'',
    ' ':'_'
}

#Using lambda functions to apply the new names to the old ones using .replace()
dat_cols_use = (dat_cols
    .assign(
        value_replace = lambda x:  x.value.str.strip().replace(values_replace, regex=True),
        variable_replace = lambda x: x.variable.str.strip().replace(variables_replace, regex=True)
    )
    .fillna(method = 'ffill')
    .fillna(value = "")
    .assign(column_names = lambda x: x.variable_replace.str.cat(x.value_replace, sep = "__").str.strip('__').str.lower())
    )
dat_cols_use
dat.columns = dat_cols_use.column_names.to_list()
# %%
"""Starting to clean up all of the data to make it machine learning compatible"""

# Converting all yes/no responses to 1/0 using .map()
seen_map = {'Yes': 1, 'No': 0}
dat['seen_any'] = dat['seen_any'].map(seen_map)
dat['star_wars_fans'] = dat['star_wars_fans'].map(seen_map)
dat['star_trek_fan'] = dat['star_trek_fan'].map(seen_map)


#Chaning the incoms column to show numbers instead of a range
income = pd.get_dummies(data = dat)

learndata= income.dropna()

# %%
"""This is for my pairplot"""
snsdata = learndata.filter(['seen_any', 'star_wars_fans',
       'rank__i__the_phantom_menace', 'rank__ii__attack_of_the_clones',
       'rank__iii__revenge_of_the_sith', 'rank__iv__a_new_hope',
       'rank__v_the_empire_strikes_back', 'rank__vi_return_of_the_jedi',
       'star_trek_fan','household_income_$0 - $24,999',
        'household_income_$25,000 - $49,999',
        'household_income_$50,000 - $99,999',
        'household_income_$100,000 - $149,999',
        'household_income_$150,000+'])




# %%
"""Setting up my machine learning model"""
#X_pred = 
X_pred = learndata.drop(columns = ['household_income_$0 - $24,999',
                                    'household_income_$25,000 - $49,999',
                                    'household_income_$50,000 - $99,999',
                                    'household_income_$100,000 - $149,999',
                                    'household_income_$150,000+'])
#y_pred = 
y_pred = learndata.filter(['household_income_$50,000 - $99,999',
                            'household_income_$100,000 - $149,999',
                            'household_income_$150,000+'])
#X_train =
#y_train = 
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .34, 
    random_state = 70) 


# %%
"""Implementing a classifier"""

#setting my classifier clf as RandomForestClassifier
clf = RandomForestClassifier()
clf_1= clf.fit(np.array(X_train), np.array(y_train))

#Using the classifier to predict my X_test subset
predictions = clf.predict(X_test)

# %%
"""Checking the precision and accuracy of my classifier's predictions"""

# Getting and printing the score for my X_test by comparing it to my y_test
score = clf.score(X_test, y_test)
print(score)

# %%
"""

#Getting my accuracy score by comparing y_test to my predictions
metrics.accuracy_score(np.array(y_test), np.array(predictions))

#Getting my precision score by comparing y_test to my predictions

metrics.precision_score(np.array(y_test), np.array(predictions))


#Printing my accuracy and precision score
print(metrics.classification_report(np.array(y_test), np.array(predictions)))

# %%"""