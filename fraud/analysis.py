import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import plot_precision_recall_curve, precision_recall_curve
from sklearn.model_selection import train_test_split


cc_data = pd.read_csv('fraud/tmp/creditcard.csv')
# look at high level info on columns
print(cc_data.info())
print('Descriptive statistics of data:')
print(cc_data.describe())

# class imbalance as expected
print('How imbalanced is the classes?')
print(cc_data.Class.value_counts())
print(cc_data.Class.value_counts(normalize=True))

cc_data.hist(figsize=(20,20));

# use a new scale for time
cc_data.loc[:, 'Time'] = cc_data.Time / 3600
cc_data['time_of_day'] = cc_data.Time.mod(24)
cc_data[['Time', 'time_of_day']].hist(bins=24, figsize=(10,5));

# use log scale for amount
print(cc_data.Amount.describe([.01,.05,.25,.5,.75,.95,.99]))
cc_data['log_amount'] = np.log(cc_data.Amount + 1e-9)
print(cc_data.log_amount.describe([.01,.05,.25,.5,.75,.95,.99]))

cc_data[['Amount', 'log_amount']].hist(figsize=(10,10), bins=250);


# use 10% of data to visualize, to make it faster
lsample, sample = train_test_split(cc_data, test_size=.1, random_state=23, stratify=cc_data.Class)
# commenting out the scatter matrix plotting as it is still slow with so many features and samples
# any lower sample would have so few positive class it'd be hard to see any trends
# sample_color = ['c' if label == 0 else 'm' for label in sample.Class]
# pd.plotting.scatter_matrix(sample[[col for col in cc_data.columns if col not in {'Class', 'Amount', 'Time'}]],
#                            figsize=(20,20), c=sample_color);
#
# plt.figure(figsize=(15,7))
# pd.plotting.parallel_coordinates(sample, class_column='Class',
#                                  cols=[col for col in cc_data.columns if col not in {'Class', 'Amount', 'Time'}],
#                                  color=['c', 'm']);

# zero amount data looks different
print('Percent of examples in different classes:',
      cc_data.Class.value_counts(normalize=True))
print('Percent of examples in different classes when Amount=0:',
      cc_data[(cc_data.Amount == 0)].Class.value_counts(normalize=True))
print('Percent of examples in different classes when Amount not 0:',
      cc_data[(cc_data.Amount != 0)].Class.value_counts(normalize=True))
print('Percent of examples where Amount=0:',
      cc_data[cc_data.Amount == 0].Amount.count()/ cc_data.Amount.count())
print('Percent of examples where Amount=0 in positive class:',
      cc_data[(cc_data.Amount == 0) &
              (cc_data.Class == 1)].Amount.count()/ cc_data[cc_data.Class == 1].Amount.count())

f = plt.figure(figsize=(15, 15))
plt.matshow(sample.corr(), fignum=f.number)
plt.xticks(range(sample.shape[1]), sample.columns, fontsize=14, rotation=45)
plt.yticks(range(sample.shape[1]), sample.columns, fontsize=14)

cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.show()

print('Modeling')
X_train, X_test = train_test_split(cc_data, test_size=.1, random_state=23, stratify=cc_data.Class)
# further split the data for tuning and model selection
train, dev = train_test_split(X_train, test_size=.2, random_state=23, stratify=X_train.Class)
print(dev.Class.value_counts())

lr = LogisticRegression(random_state=23)
features = [col for col in train.columns if col not in {'Class', 'Time', 'Amount'}]
lr.fit(train[features], train.Class)
print('Training score', f1_score(train.Class, lr.predict(train[features])))
print('Testing score', f1_score(dev.Class, lr.predict(dev[features])))

y_pred = lr.predict(dev[features])
print('Precision', precision_score(dev.Class, y_pred))
print('Recall', recall_score(dev.Class, y_pred))

y_prob = lr.predict_proba(dev[features])
print('ROC AUC Score', roc_auc_score(dev.Class, y_prob[:, 1]))


prec, rec, thres = precision_recall_curve(dev.Class, y_prob[:, 1])
fscore = 2 * prec * rec / (prec + rec)
max_ind = np.argmax(fscore)
print(f'Best threshold: {thres[max_ind]}, f1 score: {fscore[max_ind]}')
print(f'Recall {rec[max_ind]}, Precision: {prec[max_ind]}')

fig = plot_precision_recall_curve(lr, dev[features], dev.Class, color='c');
fig.ax_.set_title(f'Best f1: {fscore[max_ind]} @ threshold: {thres[max_ind]}');
plt.plot([0, 1], [0, 1], color='navy', linestyle='--');
plt.scatter(rec[max_ind], prec[max_ind], marker='o', color='m', label='Best');

# over_sampling
over_5 = SMOTE(sampling_strategy=.05, random_state=23)
train_os, y_train_os = over_5.fit_resample(train[features], train.Class)
print(y_train_os.value_counts())

_, sample_os, _, sample_os_y = train_test_split(train_os, y_train_os, test_size=.1,
                                                random_state=23, stratify=y_train_os)
sample_os['Class'] = sample_os_y
# sos_color = ['c' if label == 0 else 'm' for label in sample_os.Class]
# pd.plotting.scatter_matrix(sample_os[[col for col in cc_data.columns if col not in {'Class', 'Amount', 'Time'}]],
#                            figsize=(20,20), c=sos_color);

lr_os = LogisticRegression(random_state=23)
lr_os.fit(train_os[features], y_train_os)
print('Training score', f1_score(y_train_os, lr_os.predict(train_os[features])))
print('Testing score', f1_score(dev.Class, lr_os.predict(dev[features])))

y_pred_os = lr_os.predict(dev[features])
print('Precision', precision_score(dev.Class, y_pred_os))
print('Recall', recall_score(dev.Class, y_pred_os))

y_prob_os = lr_os.predict_proba(dev[features])
print('ROC AUC Score', roc_auc_score(dev.Class, y_prob_os[:, 1]))

# what ratio of over-sampling is better?
def over_sample_and_train(data, sampling_ratio, test_data):
    if sampling_ratio == 0:
        train_os, y_train_os = data[features], data.Class
    else:
        over = SMOTE(sampling_strategy=sampling_ratio, random_state=23)
        train_os, y_train_os = over.fit_resample(data[features], data.Class)

    lr_os = LogisticRegression(random_state=23)
    lr_os.fit(train_os[features], y_train_os)

    y_pred_os = lr_os.predict(test_data[features])
    y_prob_os = lr_os.predict_proba(test_data[features])

    result = {'model': lr_os, 'smote_ratio': sampling_ratio,
              'f1_score': f1_score(test_data.Class, y_pred_os),
              'precision': precision_score(test_data.Class, y_pred_os),
              'recall': recall_score(test_data.Class, y_pred_os),
              'roc_auc': roc_auc_score(test_data.Class, y_prob_os[:, 1])
             }

    # best f1 threshold
    prec, rec, thres = precision_recall_curve(test_data.Class, y_prob_os[:, 1])
    fscore_os = 2 * prec * rec / (prec + rec)

    max_ind_os = np.argmax(fscore_os)
    result['best_f1_threshold'] = thres[max_ind_os]
    result['best_f1'] = fscore_os[max_ind_os]
    result['best_f1_recall'] = rec[max_ind_os]
    result['best_f1_precision'] = prec[max_ind_os]

    return result

results = []
for ratio in [0, .01, .02, .03, .04, .05,.1,.15,.2,.25, .3, .4, .5, .75, 1]:
    results.append(over_sample_and_train(train, ratio, dev))

smote_df = pd.DataFrame(results)
smote_df.sort_values(['roc_auc', 'best_f1'])

print('Different over-sample scores:')
print(smote_df[['smote_ratio', 'f1_score', 'roc_auc', 'best_f1_threshold', 'best_f1']].sort_values('f1_score'))
plt.show()
