from pandas import read_csv
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from numpy import mean
from numpy import std
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from pandas.plotting import scatter_matrix

def evaluate_model(X,y,model):
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3, random_state=1 )
    scores = cross_val_score(model,X,y,scoring = 'roc_auc', cv = cv, n_jobs=-1)
    return scores

def get_models():
    models, names = list(), list()
# LR
    models.append(LogisticRegression(solver='lbfgs'))
    names.append('LR')
# SVM
    models.append(SVC(gamma='scale'))
    names.append('SVM')
# Bagging
 #   models.append(BaggingClassifier(n_estimators=1000))
  #  names.append('BAG')
    models.append(RandomForestClassifier(n_estimators=1000))
    names.append('RF')
    # GBM
    models.append(GradientBoostingClassifier(n_estimators=1000))
    names.append('GBM')
    return models, names

def get_cost_Sensitive_models():
    models, names = list(), list()
# LR
    models.append(LogisticRegression(solver='lbfgs', class_weight='balanced'))
    names.append('LR')
# SVM
    models.append(SVC(gamma='scale', class_weight='balanced'))
    names.append('SVM')
# RF
    models.append(RandomForestClassifier(n_estimators=1000))
    names.append('RF')
    return models, names

filename = 'mammography.csv'
df = read_csv(filename, header = None)
print(df.shape)
target = df.values[:,-1]
counter = Counter(target)
for k,v in counter.items():
    per = v/len(target)*100
    print('Class = %s, Count = %d, Percentage = %.3f%%' %(k,v,per))
df.hist()
pyplot.show()

#draw a scatterplot
#color_dict= {"'-1'":'red', "'1'":'green'}
#colors = [color_dict[str(x)] for x in df.values[:, -1]]
# pairwise scatter plots of all numerical variables
#scatter_matrix(df, diagonal='kde', color=colors)
#pyplot.show()
df = df.values
X,y = df[:,:-1], df[:,-1]
y = LabelEncoder().fit_transform(y)
print(X.shape, y.shape,Counter(y))
model = DummyClassifier(strategy = 'stratified')
scores = evaluate_model(X,y,model)
print('Dummy classifier Mean ROC AUC: %.3f (%.3f)' %(mean(scores), std(scores)))
models,names = get_cost_Sensitive_models()
results = list()
for i in range(len(models)):
    steps = [('p', PowerTransformer()), ('m', models[i])]
    # define pipeline
    pipeline = Pipeline(steps=steps)
    scores = evaluate_model(X,y,pipeline)
    results.append(scores)
    print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))

pyplot.boxplot(results, labels= names, showmeans=True)
pyplot.show()