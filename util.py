import pandas as pd
import numpy as np
import math
import argparse
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix




def overall_scorer(type=0):
    ''' create scorer
    Args:
        type: 0=svc(binary),
    Returns:
        function: scorer(clf, X, y)
    '''
    # def binary_classification_scorer(clf, X, y):
    #     y_pred = clf.predict(X)
    #     cm = confusion_matrix(y, y_pred)
    #     y_score = clf.decision_function(X)
    #     auc = roc_auc_score(y, y_score)
    #     return {'tn': cm[0, 0], 'fp': cm[0, 1],'fn': cm[1, 0], 'tp': cm[1, 1], 'auc': auc}

    def binary_classification_scorer(clf, X, y):
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        precision = tp/(tp+fp)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        PPV = tp / (tp+fp)
        NPV = tn / (fn+tn)
        MCC = ((tp*tn)-(fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

        y_score = clf.decision_function(X)
        auc = roc_auc_score(y, y_score)
        results = {
            'tn':tn, 
            'fp':fp, 
            'fn':fn, 
            'tp':tp,
            'auc':auc, 
            'accuracy':accuracy, 
            'precision':precision, 
            'sensitivity':sensitivity, 
            'specificity':specificity, 
            'MCC':MCC, 
            'PPV':PPV, 
            "NPV":NPV,
        }
        return results

    if type==0: return binary_classification_scorer
    # elif type==1: return binary_classification_scorer



parser = argparse.ArgumentParser(
                    prog = 'util',
                    description = "\
                        改作業用\n\
                        type (which problem):\n\
                        0: 作業一\n\
                    ",
                    epilog = '...',
                    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-i', '--test_data', help="Test Data")
parser.add_argument('-m', '--model', help="Predict model (format=joblib)")
parser.add_argument('-f', '--features', help="Record of features selection (format=csv)")
parser.add_argument('-t', '--type', help="which problem")
parser.add_argument('-o', '--out', help="output file name (format=csv, ex: 310350035_hw01.csv)")


args = parser.parse_args()

problems = [0,1,2,3,4,5,6]

try:
    raw_df = pd.read_csv(args.test_data)
except Exception as e:
    exit("Error-test data")

try:
    model = joblib.load(args.model)
except Exception as e:
    exit("Error-model")

try:
    features = pd.read_csv(args.features)
    features = list(features.columns)
except Exception as e:
    exit("Error-features")

if int(args.type) not in problems: exit("Error-type: Not in problems.")
else: problem_type = int(args.type)

outfile = args.out

selected_df = raw_df[features]
labels = raw_df.label
preds = model.predict(selected_df)

# Features number
features_num = len(features)

# Classification
def eval_classification(df, labels, model):
    ''' Cross Validation (macro average)
    '''
    cv_results = cross_validate(model, df, labels, cv = 5, scoring = overall_scorer(0))
    tn = round(cv_results['test_tn'].mean())
    fp = round(cv_results['test_fp'].mean())
    fn = round(cv_results['test_fn'].mean())
    tp = round(cv_results['test_tp'].mean())
    auc = cv_results['test_auc'].mean()
    accuracy = cv_results['test_accuracy'].mean()
    precision = cv_results['test_precision'].mean()
    sensitivity = cv_results['test_sensitivity'].mean()
    specificity = cv_results['test_specificity'].mean()
    MCC = cv_results['test_MCC'].mean()
    PPV = cv_results['test_PPV'].mean()
    NPV = cv_results['test_NPV'].mean()

    results = {
        'tn':[tn], 
        'fp':[fp], 
        'fn':[fn], 
        'tp':[tp],
        'auc':[auc], 
        'accuracy':[accuracy], 
        'precision':[precision], 
        'sensitivity':[sensitivity], 
        'specificity':[specificity], 
        'MCC':[MCC], 
        'PPV':[PPV], 
        "NPV":[NPV],
    }
    return pd.DataFrame(data=results)

# Regression

# Survival

# Evaluation
results = eval_classification(selected_df, labels, model)
print(results)

if outfile: 
    results.to_csv(outfile)
    print(f"Save results:{outfile}")

