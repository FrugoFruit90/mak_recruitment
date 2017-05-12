import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler
import keras
from keras.constraints import maxnorm
from keras.regularizers import l1_l2
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dropout, Dense
from sklearn.svm import SVC
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier

# PART 1 - LOADING THE DATA AND FEATURE EXTRACTION
# Load the data
data = pd.read_csv("loan.csv")

# Check how big is the data that we are working with
print("There are {} rows and {} columns in the data".format(data.shape[0], data.shape[1]))

# While loading the data, we got the following message:
# "sys:1: DtypeWarning: Columns (19,55) have mixed types. Specify dtype option on import or set low_memory=False."
# Let us check those columns:
print(data[[19, 55]].columns)
data[[19, 55]].head()

# Both of them contain NaNs and strings, so pandas could not assert their type. This data can still be useful for us.
# The verification column should be easy to use, as we will have only a few categorical variables:
data.iloc[:, 55].unique()

# We can therefore safely convert NaNs to strings
data['verification_status_joint'].astype(str)

# The description column is more tricky, as it will carry multiple values and thus simply taking all unique values will
# not help us a lot. Instead, we could
# TODO nr 1: extract useful information from the data on description - manually and/or using NLP methods
# As an example of the manual method, I will check if there is "consolidation" in the title:
data['maybe_consolidation'] = data.iloc[:, 19].str.contains('consolidation')

# We could also check if the description is notnan;
data['provided description'] = data.iloc[:, 19].isnull()

# We have to drop object variables that have too many unique value, because:
# a) we will run into memory error and
# b) they won't be too informative
# Here, for simplicity and computational efficiency, I drop all columns with > 51 unique values
# (I chose 51 to preserve the state variable as I expect it to be important due to legal differences)
print("Printing number of categories in each variable to see which one I can make dummies of")
for column in data.select_dtypes(include=['object', 'category']):
    print(column, data[column].unique().shape, data[column].dtype)  # check the number of categories in each field...
    if data[column].unique().shape[0] > 51:  # ...and if there are too many...
        data.drop(column, axis=1, inplace=True)  # ... then drop the column

# Finally make sure to extract the column that we actually want to predict, and drop it from the main data
y = data['loan_status'].isin(
    ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']).astype(
    float)
data.drop('loan_status', axis=1, inplace=True)

# Now, for the remaining non-numerical columns, we can convert them to dummy variables for our use
data = pd.get_dummies(data, columns=None)

# When fitting the model I got an error "ValueError: feature_names may not contain [, ] or <"
# Thus, we need to get rid of those signs beforehand
data.rename(index=str, columns={'emp_length_< 1 year': 'emp_length_ 1 year'}, inplace=True)

# Now let us check the size of the "new" data:
print("There are {} rows and {} columns in the data set after feature extraction".format(data.shape[0], data.shape[1]))

# Note that although here I just automatically convert all the columns, I had to additionally read all column names.
# This is done for several reasons, the main of which are to exclude the possibility of data leakage and to
# ensure that there are no legal issues due to sensitive information.
# The latter does not apply here as we are only making a predictive model, but it would in practical considerations
# Most models won't work with NaN values (with the notable exception of XGBoost)
# Here I just deal with it by substituting a new value, which is one good strategy, but different ones should be checked
# TODO: create a strategy for dealing with NaNs depending on the particular column data
data.fillna(-1, inplace=True)

# Also, some of the models will not work without scaling the data (e.g. Neural Networks usually don't learn without it)
data = StandardScaler().fit_transform(data)

# Lets check the share of unpaid loans
print("The share of unpaid loans is ~{}%".format(round(y.mean() * 100, 1)))

# Apparently, ~6.6% of loans are not repaid. Therefore, we have a problem of unbalanced classes.
# Thus, we need to have a similar share in training and test set, as well as use AUC as the measure to optimize
# With the above in mind, lets divide into training and testing data:
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3, stratify=y)

################################## PART 2: Model estimation: ########################################################
# For highly structured data, such as this, the XGB usually nets the best results (from the class of "single" models)
# It is also the fastest one!
xgb_model = xgb.XGBClassifier()
xgb_model.fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_test)
xgb_proba = xgb_model.predict_proba(x_test)

print("The AUC of the ROC of the XGB model is {}".format(round(roc_auc_score(y_test, xgb_proba[:, 1]), 2)))
print("The precision of the XGB model is {}".format(round(precision_score(y_test, xgb_pred), 2)))
print("The recall of the XGB model is {}".format(round(recall_score(y_test, xgb_pred), 2)))
xgb_roc = roc_curve(y_test, xgb_proba[:, 1])

plt.figure()
lw = 2
plt.plot(xgb_roc[0], xgb_roc[1], color='darkorange',
         lw=lw, label='XGB ROC curve (area = %0.2f)' % roc_auc_score(y_test, xgb_proba[:, 1]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# We can, however, try different ones as well.
# In particular, one typical choice would be to check SVMs (it's super slow though)
# To make it faster (and perhaps more robust) I use a bagging classifier and only use the first 27 features
# (after that, all features have near 0 importance according to XGB)
svm_model = BaggingClassifier(SVC(), n_estimators=10, max_samples=0.1, n_jobs=2)
svm_model.fit(x_train[:, :27], y_train)
svm_pred = svm_model.predict(x_test[:, :27])
svm_proba = svm_model.predict_proba(x_test[:, :27])
print("The AUC of the ROC of the SVM model is {}".format(round(roc_auc_score(y_test, svm_proba[:, 1]), 2)))
print("The precision of the SVM model is {}".format(round(precision_score(y_test, svm_pred), 2)))
print("The recall of the SVM model is {}".format(round(recall_score(y_test, svm_pred), 2)))
svm_roc = roc_curve(y_test, svm_proba[:, 1])

plt.figure()
lw = 2
plt.plot(svm_roc[0], svm_roc[1], color='darkorange',
         lw=lw, label='SVM ROC curve (area = %0.2f)' % roc_auc_score(y_test, svm_proba[:, 1]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# We can also try deep learning (very slow as well):
# multithreading
num_cores = str(multiprocessing.cpu_count())
os.environ['OMP_NUM_THREADS'] = num_cores
os.environ['GOTO_NUM_THREADS'] = num_cores
os.environ['OPENBLAS_NUM_THREADS'] = num_cores

no_regressors = data.shape[1]


def create_spec_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(no_regressors,)))
    model.add(Dense(200, init='normal', activation='relu', W_constraint=maxnorm(2), W_regularizer=l1_l2(l1=0, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(120, init='normal', activation='relu', W_constraint=maxnorm(2), W_regularizer=l1_l2(l1=0, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.005, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# Early stopping makes the process a bit faster
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='auto')
# Model check enables comparison of results across different epochs
modelCheck = keras.callbacks.ModelCheckpoint('model_deepn', monitor='val_accuracy', verbose=1, save_best_only=True,
                                             mode='auto')
deepnc = KerasClassifier(build_fn=create_spec_model, validation_split=0.2, batch_size=100, nb_epoch=40)
deepnc.fit(np.array(x_train), np.array(y_train))
deepn_pred = deepnc.predict(np.array(y_test))
deepn_proba = deepnc.predict_proba(np.array(y_test))
print("The AUC of the ROC of the Deep Learning model is {}".format(round(roc_auc_score(y_test, deepn_proba[:, 1]), 2)))
print("The precision of the Deep Learning model is {}".format(round(precision_score(y_test, deepn_pred), 2)))
print("The recall of the Deep Learning model is {}".format(round(recall_score(y_test, deepn_pred), 2)))

deepn_roc = roc_curve(y_test, deepn_proba[:, 1])

plt.figure()
lw = 2
plt.plot(deepn_roc[0], deepn_roc[1], color='darkorange',
         lw=lw, label='DL ROC curve (area = %0.2f)' % roc_auc_score(y_test, deepn_proba[:, 1]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# As expected, XGB had the best performance

# The results of all models mentioned above could be slightly improved using parameter tuning
# But the real benefit from parameter tuning comes from model stacking - ones we have several models calibrated,
# we can stack several models e.g. using soft voting (soft voting is when three models vote using probabilities,
# e.g. if we have 3 models we take the average of their probabilities and predict 1 if probability > 0.5)
# TODO: calibrate all three models using GridSearchCV (for SVM as it only has one parameter) and RandomSearchCV (for deep learning architecture and XGB parameters)
# TODO: stack the three models using the Pipeline module and Voting Classifierfrom scikit-learn

# An example of such a pipeline from one of my recent projects:
# pipeline = Pipeline(
#     [
#         ('scaler', StandardScaler()),
#         ('feat_sel', SelectKBest(score_func=f_classif, k = 50)),
#         ('estimators', FeatureUnion([
#             ('deepnc', ModelTransformer(deepnc)),
#             ('xgbc', ModelTransformer(xgbc)),
#             ('svm', ModelTransformer(svm))
#         ])),
#         ('vc', vc)
#     ])
# In the pipeline above, I first scale the data, then select 50 best features, then estimate 3 models, then stack them using a voting classifier
# Then I simply fit the data to this pipeline and the whole big model is fitted.

##################### PART 3: Visualization #####################
# This is a rather time consuming part. What I would do is first plot the feature importances (grouping them by origin
# e.g. I would sum all the 51 importances corresponding to states
# After summing I would
# TODO: plot feature importances from XGB, sum them by category, then plot histograms/maps/boxplots of the most important variables to illustrate the data
