## General view of the data
import pandas as pd

# Read in track metadata with genre labels
tracks = pd.read_csv('datasets/fma-rock-vs-hiphop.csv')

# Read in track metrics with the features
echonest_metrics = pd.read_json("datasets/echonest-metrics.json", precise_float = True)

# Merge the relevant columns of tracks and echonest_metrics
echo_tracks = echonest_metrics.merge(tracks[['track_id', 'genre_top']])

# Inspect the resultant dataframe
print(echo_tracks.info())
print(echo_tracks.head)



## Creating a correlation matrix
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()



## Splitting the dataset
# Import train_test_split function and Decision tree classifier
# ... YOUR CODE ...
from sklearn.model_selection import train_test_split


# Create features
features = echo_tracks.drop(['genre_top', 'track_id'], axis=1).values

# Create labels
labels = echo_tracks['genre_top'].values



# Split our data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=10)


## Normalization of the data
# Import the StandardScaler
from sklearn.preprocessing import StandardScaler

# Scale the features and set the values to a new variable
scaler = StandardScaler()

# Scale train_features and test_features
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)



## PCA On DATA
# This is just to make plots appear in the notebook
%matplotlib
inline

# Import our plotting module, and PCA class
# ... YOUR CODE ...

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_
# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(0, len(pca.components_)), exp_variance)
ax.set_xlabel('Principal Component #')

cumulative_freqs = list()
freq = 0
for component in exp_variance:
    freq = freq + component
    cumulative_freqs.append(freq)

# Cummulative frequencies
fig, ax = plt.subplots()
ax.bar(range(0, len(pca.components_)), cumulative_freqs)
ax.set_xlabel('Principal Component #')
ax.set_ylabel("Cumulative explained varience ratio")




## More analysis
# Import numpy
import numpy as np

# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.85.
fig, ax = plt.subplots()
ax.bar(range(0,len(pca.components_)), cum_exp_variance)
ax.set_xlabel('Principal Component #')
ax.set_ylabel("Cumulative explained varience ratio")

ax.axhline(y=0.85, linestyle='--')




## Projecting
# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components=6)

# Fit and transform the scaled training features using pca
train_pca = pca.fit_transform(scaled_train_features)

# Fit and transform the scaled test features using pca
test_pca = pca.transform(scaled_test_features)



## Decision tree classifier
# Import Decision tree classifier
# ... YOUR CODE ...
from sklearn.tree import DecisionTreeClassifier

# Train our decision tree
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca, train_labels )

# Predict the labels for the test data
pred_labels_tree = tree.predict(test_pca)


## Logistic regression

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Train our logistic regression and predict labels for the test set
logreg = LogisticRegression(random_state=10)
logreg.fit(train_features, train_labels)
pred_labels_logit = logreg.predict(test_features)

# Create the classification report for both models
from sklearn.metrics import classification_report
class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)

print("Decision Tree: \n", class_rep_tree)
print("Logistic Regression: \n", class_rep_log)


## Balancing data for better performance

# Subset only the hip-hop tracks, and then only the rock tracks
hop_only = echo_tracks.loc[echo_tracks['genre_top'] == "Hip-Hop"]
rock_only = echo_tracks.loc[echo_tracks['genre_top'] == "Rock"]

# sample the rocks songs to be the same number as there are hip-hop songs
rock_only = rock_only.sample(n=len(hop_only), random_state=10)

# concatenate the dataframes rock_only and hop_only
rock_hop_bal = pd.concat([rock_only, hop_only])

# The features, labels, and pca projection are created for the balanced dataframe
features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1)
labels = rock_hop_bal['genre_top']

# Redefine the train and test set with the pca_projection from the balanced data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=10)

scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)


train_pca = pca.fit_transform(scaler.fit_transform(scaled_train_features))
test_pca = pca.transform(scaler.transform(scaled_test_features))




## Model performance

# Train our decision tree on the balanced data
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca, train_labels)
pred_labels_tree = tree.predict(test_pca)

# Train our logistic regression on the balanced data
logreg = LogisticRegression(random_state=10)
logreg.fit(train_pca, train_labels)
pred_labels_logit = logreg.predict(test_pca)

# Compare the models
print("Decision Tree: \n", classification_report(test_labels,y_pred=pred_labels_tree))
print("Logistic Regression: \n", classification_report(test_labels, y_pred=pred_labels_logit))



## Cross val to evaluate models
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
tree_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=6)), 
                      ("tree", DecisionTreeClassifier(random_state=10))])
logreg_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=6)), 
                        ("logreg", LogisticRegression(random_state=10))])

# Set up our K-fold cross-validation
kf = KFold(10)

# Train our models using KFold cv


tree_score = cross_val_score(tree_pipe, features, labels, cv=kf)
logit_score = cross_val_score(logreg_pipe, features, labels, cv=kf)

# Print the mean of each array of scores
print("Decision Tree:", tree_score, "Logistic Regression:", logit_score)


