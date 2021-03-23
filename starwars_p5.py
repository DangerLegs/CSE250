# %%
# Basic imports
import pandas as pd 
import altair as alt
import numpy as np


# %%
# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# %%
# Reading in my data to csv file
url = 'https://github.com/fivethirtyeight/data/raw/master/star-wars-survey/StarWars.csv'
dat = pd.read_csv(url, encoding = 'latin1', skiprows = 2, header = None)
# Selecting the 1st row that contains all of the column names
dat_cols = pd.read_csv(url, encoding = 'latin1', nrows = 1)
dat_cols = dat_cols.melt()


# %%
# Cleaning up the names of the columns
# we want to use this with the .replace() command that accepts a dictionary.
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
#use dat!
dat.columns = dat_cols_use.column_names.to_list()


# %%
# Filtering the dataset to only show those that have seen at least one film
dat_names =(dat_cols_use
   .replace('Unnamed: \d{1,2}', np.nan, regex=True)
   .replace('Response', "")
   .assign(
      clean_variable = lambda x: x.variable.str.strip()
         .replace(
            'Which of the following Star Wars films have you seen? Please select all that apply.','seen'),
      clean_value = lambda x: x.value.str.strip()
      )
   .fillna(method = 'ffill')
   .assign(
      column_name = lambda x: x.clean_variable.str.cat(x.clean_value, sep = "__")
   )
)

dat.columns = dat_names.column_names.to_list()

# %% Converting all yes/no responses to 1/0

seen_map = {'Yes': 1, 'No': 0}
dat['seen_any'] = dat['seen_any'].map(seen_map)


# %%
# Adding a column that shows income range as a number
(dat.household_income
    .str.split(' - ', expand=True)
    .rename(columns={0: "min_income", 1: 'max_income'})
    .min_income.str.replace('\$|,|\+', '', regex = True)
    .astype('float')
)


# %% 
# Adding a column that shows age range as a number

(dat.age
    .str.split('-', expand = True)
    .rename(columns = {0: 'min_age', 1: "max_age"})
    .min_age
    .str.replace('> ', '')
    .astype('float')
)

# %%
#Adding a column that shows school groupings as a number
# %%

seenDF = dat.filter(regex = 'seen__')
DatSeen = pd.get_dummies(seenDF)
SeenMovies = DatSeen.sum()

