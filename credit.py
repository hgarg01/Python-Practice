#load the dataset
from pandas import read_csv
from collections import Counter
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.pipeline import Pipeline



filename = 'german.csv'

#load the file
df = read_csv(filename, header = None)
print(df.shape)

#summarize class distribution
target = df.values[:,-1]
counter = Counter(target)
for k,v in counter.items():
    per = v/len(target) * 100
    print('Class = %d, Count = %d, Percentage = %.3f%%' %(k,v,per))

#create histograms of numeric input variables

#split dataset into X and y
last_ix = len(df.columns) - 1
X, y = df.drop(last_ix, axis=1), df[last_ix]
cat_ix = X.select_dtypes(include = ['object', 'bool']).columns
num_ix = X.select_dtypes(include = ['int64', 'float64']).columns

#label encode the target variable
y = LabelEncoder().fit_transform(y)

def f2_measure(y_true, y_pred):
    return fbeta_score(y_true,y_pred, beta = 2)

def evaluate_model(X,y,model):
    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats=3, random_state=1 )
    metric = make_scorer(f2_measure)
    scores = cross_val_score(model, X,y, scoring= metric, cv = cv, n_jobs=-1)
    return scores

#define models to test
def get_models():
    models,names = list(),list()
    models.append(LogisticRegression(solver = 'liblinear'))
    names.append('LR')
    models.append(LinearDiscriminantAnalysis())
    names.append('LDA')
    models.append(GaussianNB())
    names.append('NB')
    models.append(SVC(gamma = 'scale'))
    names.append('SVM')
    return models,names

def get_under_sample_models():
    models,names = list(),list()
    models.append(TomekLinks())
    names.append('TomesLinks')
    models.append(EditedNearestNeighbours())
    names.append('EditedNearestNeighbors')
    models.append(RepeatedEditedNearestNeighbours())
    names.append('RENN')
    models.append(OneSidedSelection())
    names.append('OneSidedSelection')
    models.append(NeighbourhoodCleaningRule())
    names.append('NCR')
    return models,names

#create dummy model and evaluate
model = DummyClassifier(strategy = 'constant', constant = 1)
scores = evaluate_model(X,y,model)
print(X.shape, y.shape, Counter(y))
print('Dummy Classifier  :  ')
print('Mean F2 :%.3f (%.3f)' %(mean(scores), std(scores)))

models,names = get_models()
results = list()
#evaluate each model and print results

for i in range(len(models)):
    # one hot encode categorical, normalize numerical
    ct = ColumnTransformer([('c', OneHotEncoder(), cat_ix), ('n', MinMaxScaler(), num_ix)])
    # wrap the model in a pipeline
    pipeline = Pipeline(steps=[('t', ct), ('m', models[i])])
    # evaluate the model and store results
    scores = evaluate_model(X, y, pipeline)
    results.append(scores)
    # summarize and store
    print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))


#boxplot of results
pyplot.boxplot(results, labels = names, showmeans=True)
pyplot.show()

results = list()
models,names = get_under_sample_models()
for i in range(len(models)):
# define model to evaluate
    model = LogisticRegression(solver='liblinear', class_weight='balanced')
# one hot encode categorical, normalize numerical
    ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])
# scale, then undersample, then fit model
    pipeline = Pipeline(steps=[('t',ct), ('s', models[i]), ('m',model)])
# evaluate the model and store results
    scores = evaluate_model(X, y, pipeline)
    results.append(scores)
# summarize and store
    print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))

pyplot.boxplot(results, labels = names, showmeans=True)
pyplot.show()

