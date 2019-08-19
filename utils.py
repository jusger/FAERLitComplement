# Import Packages
import numpy as np
import pandas as pd
from bitarray import bitarray
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import SCORERS
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

#Define some useful functions

# Load disproportionality data (from Harpaz et al)
# calculated on FAERS data through Q3 2011 
# I skip LR and ELR here
# The LR and ELR they present are calculated differently
# by running the regressor on more empirical data, and not just
# by running it on calculated ratios (as done here).
def loadharpazdata(inputfile, labelfile, harpazfile):
    tmpvecs = []
    tmpqueries = []
    with open(inputfile, 'r') as infile:
        infile.readline()#skip header row
        for line in infile:
            tmp = line.strip().split('|')
            tmpvecs += [np.asarray(bitarray(tmp[1]).tolist(), dtype=int)]
            tmpqueries += [tmp[0]]
        tmpdf = pd.DataFrame(np.asarray(tmpvecs))
        tmpdf.columns = [str(x) for x in range(1, np.asarray(tmpvecs).shape[1]+1)]
        tmpdf.insert(0, 'Query', tmpqueries)
    labsdict = dict()
    with open(labelfile,'r') as infile:
        for line in infile:
            tmp = line.strip().split('\t')
            labsdict[tmp[1]] = int(tmp[0])
    tmpdf.insert(1, 'Label', [labsdict[x] for x in tmpqueries])
    prrdict = dict()
    with open(harpazfile, 'r') as infile:
        infile.readline() #skip header
        for line in infile:
            tmp = line.strip().split('\t')
            if tmp[0] == 'darunavir' or tmp[0] == 'sitagliptin':
                continue
            if tmp[0] == 'tenofovir':
                prrdict[f'S({tmp[0]})*S({tmp[1]})'] = ['NA']*8
                continue
            prrdict[f'S({tmp[0]})*S({tmp[1]})'] = [tmp[7], tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[21], tmp[22]]
            #['EBGM', 'EB05', 'PRR', 'PRR05', 'ROR', 'ROR05', 'EBGM (none-stratified)', 'EB05 (none-stratified)']
    tmpdf.insert(2, 'EB05NoStrat', [prrdict[x][-1] for x in tmpqueries])
    tmpdf.insert(2, 'EBGMNoStrat', [prrdict[x][-2] for x in tmpqueries])
    tmpdf.insert(2, 'ROR05', [prrdict[x][-3] for x in tmpqueries])
    tmpdf.insert(2, 'ROR', [prrdict[x][-4] for x in tmpqueries])
    tmpdf.insert(2, 'PRR05', [prrdict[x][-5] for x in tmpqueries])
    tmpdf.insert(2, 'PRR', [prrdict[x][-6] for x in tmpqueries])
    tmpdf.insert(2, 'EB05', [prrdict[x][-7] for x in tmpqueries])
    tmpdf.insert(2, 'EBGM', [prrdict[x][-8] for x in tmpqueries])
    return tmpdf


#Load disproportionality data calculated by Banda et al
#from FAERS data through Q2 2015
def loadbandadata(inputfile, labelfile, bandafile):
    tmpvecs = []
    tmpqueries = []
    with open(inputfile, 'r') as infile:
        infile.readline()#skip header row
        for line in infile:
            tmp = line.strip().split('|')
            tmpvecs += [np.asarray(bitarray(tmp[1]).tolist(), dtype=int)]
            tmpqueries += [tmp[0]]
        tmpdf = pd.DataFrame(np.asarray(tmpvecs))
        tmpdf.columns = [str(x) for x in range(1, np.asarray(tmpvecs).shape[1]+1)]
        tmpdf.insert(0, 'Query', tmpqueries)
    labsdict = dict()
    with open(labelfile,'r') as infile:
        for line in infile:
            tmp = line.strip().split('\t')
            labsdict[tmp[1]] = int(tmp[0])
    tmpdf.insert(1, 'Label', [labsdict[x] for x in tmpqueries])
    prrdict = dict()
    with open(bandafile, 'r') as infile:
        for line in infile:
            tmp = line.strip().split('\t')
            prrdict['S(fluvoxamine)*S(diseases_of_mitral_valve)'] = ['NA'] * 10
            prrdict['S(captopril)*S(acute_kidney_insufficiency)'] = ['NA'] * 10
            prrdict['S(carteolol)*S(liver_failure,_acute)'] = ['NA'] * 10
            prrdict[f'S({tmp[0]})*S({tmp[1]})'] = [tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], tmp[8]]
            #Drug    ADE     #Reports        PRR     PRRUB   PRRLB   ROR     RORUB   RORLB 
    tmpdf.insert(2, 'RORLB', [prrdict[x][-1] for x in tmpqueries])
    tmpdf.insert(2, 'RORUB', [prrdict[x][-2] for x in tmpqueries])
    tmpdf.insert(2, 'ROR', [prrdict[x][-3] for x in tmpqueries])
    tmpdf.insert(2, 'PRRLB', [prrdict[x][-4] for x in tmpqueries])
    tmpdf.insert(2, 'PRRUB', [prrdict[x][-5] for x in tmpqueries])
    tmpdf.insert(2, 'PRR', [prrdict[x][-6] for x in tmpqueries])
    tmpdf.insert(2, 'CaseReports', [prrdict[x][-7] for x in tmpqueries])
    return tmpdf 

# Leave one out crossvalidation quick test
def lootest(df):
    preds = []
    predprob = []
    reals = []
    vecs = np.asarray(df.iloc[:,2:])
    labels = np.asarray(df.Label)
    for train,test in LeaveOneOut().split(vecs):
        model = LogisticRegression(penalty='l1', solver='liblinear')
        model.fit(vecs[train], labels[train])
        preds += [model.predict(vecs[test])]
        predprob += [model.predict_proba(vecs[test])[:,1]]
        reals += [labels[test]]
    return(f1_score(reals, preds), roc_auc_score(reals, predprob))
    
# Stratified 5 Fold crossvalidation quick test
# Returns overall F1 and ROC AUC for comparison to other research
# That is, we don't compute fold to fold, but over the whole set
def skftest(df):
    reals = np.asarray([])
    preds = np.asarray([])
    predprob = np.asarray([])
    vecs = np.asarray(df.iloc[:,2:])
    labels = np.asarray(df.Label)
    for train,test in StratifiedKFold(n_splits=5, shuffle=True).split(vecs, labels):
        model = LogisticRegression(penalty='l1', solver='liblinear')
        model.fit(vecs[train], labels[train])
        predprob = np.append(predprob, model.predict_proba(vecs[test])[:,1])
        reals = np.append(reals, labels[test])
        preds = np.append(preds, model.predict(vecs[test]))
    return(f1_score(reals, preds), roc_auc_score(reals, predprob))

# Get the average performance across 100 runs for a given LR training model
def get_average_performance(df, trainfunc):
    aucscores = []
    fscores = []
    for i in range(100):
        tmpf1, tmpauc = trainfunc(df)
        fscores += [tmpf1]
        aucscores += [tmpauc]
    return(np.average(fscores), 1.96*(np.std(fscores)/np.sqrt(100)), np.average(aucscores), 1.96*(np.std(aucscores)/np.sqrt(100)))

# Create an ensemble method which weights the literature at a float between 0 and 1 relative contribution
def ensembleskftest(basevecs, dvecs, labels, litweight=0.1):
    reals = np.asarray([])
    preds = np.asarray([])
    basepredprob = np.asarray([])
    dpredprob = np.asarray([])
    predprob = np.asarray([])
    for train,test in StratifiedKFold(n_splits=5, shuffle=True).split(basevecs, labels):
        basemodel = LogisticRegression(penalty='l1', solver='liblinear', max_iter=200)
        basemodel.fit(basevecs[train], labels[train])
        baseprob = basemodel.predict_proba(basevecs[test])[:,1]
        basepredprob = np.append(basepredprob, baseprob)
        dmodel = LogisticRegression(penalty='l1', solver='liblinear', max_iter=200)
        dmodel.fit(dvecs[train], labels[train])
        dprob = dmodel.predict_proba(dvecs[test])[:,1]
        dpredprob = np.append(dpredprob, dprob)
        ensembleprob = baseprob*litweight+dprob*(1-litweight)
        predprob = np.append(predprob, ensembleprob)
        reals = np.append(reals, labels[test])
        preds = np.append(preds, np.round(ensembleprob))
    return(average_precision_score(reals, preds), roc_auc_score(reals, predprob))

def get_average_performance_ensemble(trainfunc, litvecs, dvecs, labels, weight):
    aucscores = []
    fscores = []
    runs = 100
    for i in range(runs):
        tmpf1, tmpauc = trainfunc(litvecs, dvecs, labels, weight)
        fscores += [tmpf1]
        aucscores += [tmpauc]
    return(np.average(fscores), np.std(fscores)/np.sqrt(runs), np.average(aucscores), np.std(aucscores)/np.sqrt(runs)) #1.96 is multiplied later in graphing

#Define graphing functions

def plot_ensemble(df, disproidx, title='Ensemble Model Performance'):
    plt.figure(figsize=(15,10), dpi=300)
    plt.title(title)
    plt.xlabel("Percent Literature Contribution to Prediction")
    plt.ylabel("Mean ROC AUC")
    litvecs = np.asarray(df.iloc[:,disproidx:])
    dvecs = np.asarray(df.iloc[:,2:disproidx])
    labels = np.asarray(df.Label)
    litweights = np.linspace(0,1,20)
    #test_scores = []
    baselinedscore = []
    
    test_scores_mean = []
    test_scores_std = []
    for lw in litweights:
        '''
        _tmpscores = []
        for train,test in StratifiedKFold(n_splits=5, shuffle=True).split(litvecs, labels):
            #Define models and fit
            litmodel = LogisticRegression(penalty='l1', solver='liblinear')
            litmodel.fit(litvecs[train], labels[train])
            dmodel = LogisticRegression(penalty='l1', solver='liblinear')
            dmodel.fit(dvecs[train], labels[train])
            #Get respective performance
            litprob = litmodel.predict_proba(litvecs[test])[:,1]
            dprob = dmodel.predict_proba(dvecs[test])[:,1]
            ensembleprob = litprob*lw+dprob*(1-lw)
            preds = np.round(ensembleprob)
            _tmpscores += [f1_score(labels[test], preds)]
        test_scores += [_tmpscores]
            
    #print(test_scores)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
        '''
        _tmpresults = get_average_performance_ensemble(ensembleskftest, litvecs, dvecs, labels, lw)
        test_scores_mean += [_tmpresults[2]]
        test_scores_std += [_tmpresults[3]]
     
    plt.grid()
    plt.fill_between(litweights, np.asarray(test_scores_mean) - np.asarray(test_scores_std)*1.96,
                     np.asarray(test_scores_mean) + np.asarray(test_scores_std)*1.96, alpha=0.1, color="g")
    plt.plot(litweights, np.asarray(test_scores_mean), '^-', color="g",
             label="Ensemble Model")
    
    plt.fill_between(litweights, [test_scores_mean[0] - test_scores_std[0]*1.96]*len(litweights),
                     [test_scores_mean[0]+test_scores_std[0]*1.96]*len(litweights), alpha=0.1, color='b')
    plt.plot(litweights, [test_scores_mean[0]]*len(litweights), 's-', color='b', label='DPM Only')

    plt.legend(loc="best")
    plt.ylim(0.5, 1.0)
    return plt

def plot_shuffle_triple_comparison_learning_curve(estimator, title, X1, y1, X2, y2, X3, y3, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 20), score=None, label1=' ', label2=' ', label3=' '):
    """
    Code adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    fig = plt.figure(figsize=(15,10), dpi=300)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if score is None:
        score = 'average_precision'
    ylabscore = ' '.join([x.capitalize() for x in score.split('_')])
    if ylabscore == 'Roc Auc':
        ylabscore = 'ROC AUC'
    plt.xlabel("Training Examples")
    plt.ylabel(f"Mean {ylabscore}")
    train_sizes2 = np.asarray(train_sizes)
    train_sizes3 = np.asarray(train_sizes)

    test_scores_means = []
    test_scores_means2 = []
    test_scores_means3 = []
    for i in range(100): # probably need to refresh each of the models passed with a clone operation, in all likelihood
        #Shuffle rows of data
        '''
        This must be done if we don't want the learning curve to essentially pick the same data points for
        every single run. This is because the learning curve pulls only the first n samples, and StratifiedKFold
        returns the properly stratified splits, but in index order (i.e. they aren't designed to be subsampled).
        So, in order to generally keep the same stratification and subsample appropriately (for Standard Error of
        the Mean, the reporting metric for variances of means), we shuffle the data set on every repetition prior
        to passing to SKF-CV (note that SKF has a shuffle function, but this only shuffles membership not order).
        '''
        idxs = np.random.permutation(X1.shape[0])

        #First Metrics
        train_sizes, train_scores, test_scores = learning_curve(
            clone(estimator), X1[idxs], y1[idxs], cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=score, shuffle=False)

        #Comparison Metrics
        train_sizes2, train_scores2, test_scores2 = learning_curve(
            clone(estimator), X2[idxs], y2[idxs], cv=cv, n_jobs=n_jobs, train_sizes=train_sizes2, scoring=score, shuffle=False)
        
        train_sizes3, train_scores3, test_scores3 = learning_curve(
            clone(estimator), X3[idxs], y3[idxs], cv=cv, n_jobs=n_jobs, train_sizes=train_sizes3, scoring=score, shuffle=False)
        
        #train_scores_mean = np.mean(train_scores, axis=1)
        #train_scores_std = np.std(train_scores, axis=1)
        test_scores_means += [np.mean(test_scores, axis=1)]
        #test_scores_std = np.std(test_scores, axis=1)

        #train_scores_mean2 = np.mean(train_scores2, axis=1)
        #train_scores_std2 = np.std(train_scores2, axis=1)
        test_scores_means2 += [np.mean(test_scores2, axis=1)]
        #test_scores_std2 = np.std(test_scores2, axis=1)
        test_scores_means3 += [np.mean(test_scores3, axis=1)]
    
    test_scores_mean = np.mean(test_scores_means, axis=0)
    test_scores_std = np.std(test_scores_means, axis=0)/np.sqrt(100)
    
    test_scores_mean2 = np.mean(test_scores_means2, axis=0)
    test_scores_std2 = np.std(test_scores_means2, axis=0)/np.sqrt(100)
    
    test_scores_mean3 = np.mean(test_scores_means3, axis=0)
    test_scores_std3 = np.std(test_scores_means3, axis=0)/np.sqrt(100)
    
    
    #print(np.mean(test_scores_means, axis=0))
    #print(np.std(test_scores_means, axis=0))
    #print(ks_2samp(np.ravel(test_scores_means, order='F').reshape(20,3), np.ravel(test_scores_means2, order='F').reshape(20,3))[1])
    #print(np.ravel(test_scores_means, order='F').reshape(20,3))
    
    plt.grid()
    
    #Plot first graph
    #plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     #train_scores_mean + train_scores_std, alpha=0.1,
                     #color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std*1.96,
                     test_scores_mean + test_scores_std*1.96, alpha=0.1, color="g")
    #plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             #label=f"{label1} Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label=f"{label1}")
    
    #Plot comparison graph
    #plt.fill_between(train_sizes2, train_scores_mean2 - train_scores_std2,
                     #train_scores_mean2 + train_scores_std2, alpha=0.1,
                     #color="darkred")
    plt.fill_between(train_sizes2, test_scores_mean2 - test_scores_std2*1.96,
                     test_scores_mean2 + test_scores_std2*1.96, alpha=0.1, color="darkblue")
    #plt.plot(train_sizes2, train_scores_mean2, 'o-', color="darkred",
             #label=f"{label2} Training Score")
    plt.plot(train_sizes2, test_scores_mean2, '^-', color="darkblue",
             label=f"{label2}")
    
    #Plot comparison graph
    #plt.fill_between(train_sizes2, train_scores_mean2 - train_scores_std2,
                     #train_scores_mean2 + train_scores_std2, alpha=0.1,
                     #color="darkred")
    plt.fill_between(train_sizes3, test_scores_mean3 - test_scores_std3*1.96,
                     test_scores_mean3 + test_scores_std3*1.96, alpha=0.1, color="#FF4500")
    #plt.plot(train_sizes2, train_scores_mean2, 'o-', color="darkred",
             #label=f"{label2} Training Score")
    plt.plot(train_sizes3, test_scores_mean3, 's-', color="#FF4500",
             label=f"{label3}")

    plt.legend(loc="lower right")
    return fig


def plot_shuffle_quad_comparison_learning_curve(estimator, title, X1, y1, X2, y2, X3, y3, X4, y4, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 20), score=None, label1=' ', label2=' ', label3=' ', label4=' '):
    """
    Code adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    fig = plt.figure(figsize=(15,10), dpi=300)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if score is None:
        score = 'average_precision'
    ylabscore = ' '.join([x.capitalize() for x in score.split('_')])
    if ylabscore == 'Roc Auc':
        ylabscore = 'ROC AUC'
    plt.xlabel("Training Examples")
    plt.ylabel(f"Mean {ylabscore}")
    train_sizes2 = np.asarray(train_sizes)
    train_sizes3 = np.asarray(train_sizes)
    train_sizes4 = np.asarray(train_sizes)

    test_scores_means = []
    test_scores_means2 = []
    test_scores_means3 = []
    test_scores_means4 = []
    for i in range(100): # probably need to refresh each of the models passed with a clone operation, in all likelihood
        #Shuffle rows of data
        '''
        This must be done if we don't want the learning curve to essentially pick the same data points for
        every single run. This is because the learning curve pulls only the first n samples, and StratifiedKFold
        returns the properly stratified splits, but in index order (i.e. they aren't designed to be subsampled).
        So, in order to generally keep the same stratification and subsample appropriately (for Standard Error of
        the Mean, the reporting metric for variances of means), we shuffle the data set on every repetition prior
        to passing to SKF-CV (note that SKF has a shuffle function, but this only shuffles membership not order).
        '''
        idxs = np.random.permutation(X1.shape[0])
        idxs4 = np.random.permutation(X4.shape[0])

        #First Metrics
        train_sizes, train_scores, test_scores = learning_curve(
            clone(estimator), X1[idxs], y1[idxs], cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=score, shuffle=False)

        #Comparison Metrics
        train_sizes2, train_scores2, test_scores2 = learning_curve(
            clone(estimator), X2[idxs], y2[idxs], cv=cv, n_jobs=n_jobs, train_sizes=train_sizes2, scoring=score, shuffle=False)
        
        train_sizes3, train_scores3, test_scores3 = learning_curve(
            clone(estimator), X3[idxs], y3[idxs], cv=cv, n_jobs=n_jobs, train_sizes=train_sizes3, scoring=score, shuffle=False)
        
        train_sizes4, train_scores4, test_scores4 = learning_curve(
            clone(estimator), X4[idxs4], y4[idxs4], cv=cv, n_jobs=n_jobs, train_sizes=train_sizes4, scoring=score, shuffle=False)
        
        #train_scores_mean = np.mean(train_scores, axis=1)
        #train_scores_std = np.std(train_scores, axis=1)
        test_scores_means += [np.mean(test_scores, axis=1)]
        #test_scores_std = np.std(test_scores, axis=1)

        #train_scores_mean2 = np.mean(train_scores2, axis=1)
        #train_scores_std2 = np.std(train_scores2, axis=1)
        test_scores_means2 += [np.mean(test_scores2, axis=1)]
        #test_scores_std2 = np.std(test_scores2, axis=1)
        test_scores_means3 += [np.mean(test_scores3, axis=1)]
        test_scores_means4 += [np.mean(test_scores4, axis=1)]
    
    test_scores_mean = np.mean(test_scores_means, axis=0)
    test_scores_std = np.std(test_scores_means, axis=0)/np.sqrt(100)
    
    test_scores_mean2 = np.mean(test_scores_means2, axis=0)
    test_scores_std2 = np.std(test_scores_means2, axis=0)/np.sqrt(100)
    
    test_scores_mean3 = np.mean(test_scores_means3, axis=0)
    test_scores_std3 = np.std(test_scores_means3, axis=0)/np.sqrt(100)
    
    test_scores_mean4 = np.mean(test_scores_means4, axis=0)
    test_scores_std4 = np.std(test_scores_means4, axis=0)/np.sqrt(100)
    
    
    #print(np.mean(test_scores_means, axis=0))
    #print(np.std(test_scores_means, axis=0))
    #print(ks_2samp(np.ravel(test_scores_means, order='F').reshape(20,3), np.ravel(test_scores_means2, order='F').reshape(20,3))[1])
    #print(np.ravel(test_scores_means, order='F').reshape(20,3))
    
    plt.grid()
    
    #Plot first graph
    #plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     #train_scores_mean + train_scores_std, alpha=0.1,
                     #color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std*1.96,
                     test_scores_mean + test_scores_std*1.96, alpha=0.1, color="g")
    #plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             #label=f"{label1} Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label=f"{label1}")
    
    #Plot comparison graph
    #plt.fill_between(train_sizes2, train_scores_mean2 - train_scores_std2,
                     #train_scores_mean2 + train_scores_std2, alpha=0.1,
                     #color="darkred")
    plt.fill_between(train_sizes2, test_scores_mean2 - test_scores_std2*1.96,
                     test_scores_mean2 + test_scores_std2*1.96, alpha=0.1, color="darkblue")
    #plt.plot(train_sizes2, train_scores_mean2, 'o-', color="darkred",
             #label=f"{label2} Training Score")
    plt.plot(train_sizes2, test_scores_mean2, 'o-', color="darkblue",
             label=f"{label2}")
    
    #Plot comparison graph
    #plt.fill_between(train_sizes2, train_scores_mean2 - train_scores_std2,
                     #train_scores_mean2 + train_scores_std2, alpha=0.1,
                     #color="darkred")
    plt.fill_between(train_sizes3, test_scores_mean3 - test_scores_std3*1.96,
                     test_scores_mean3 + test_scores_std3*1.96, alpha=0.1, color="#FF4500")
    #plt.plot(train_sizes2, train_scores_mean2, 'o-', color="darkred",
             #label=f"{label2} Training Score")
    plt.plot(train_sizes3, test_scores_mean3, 'o-', color="#FF4500",
             label=f"{label3}")
    
    plt.fill_between(train_sizes4, test_scores_mean4 - test_scores_std4*1.96,
                     test_scores_mean4 + test_scores_std4*1.96, alpha=0.1, color="black")
    #plt.plot(train_sizes2, train_scores_mean2, 'o-', color="darkred",
             #label=f"{label2} Training Score")
    plt.plot(train_sizes4, test_scores_mean4, 'o-', color="black",
             label=f"{label4}")

    plt.legend(loc="lower right")
    return fig
