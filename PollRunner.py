from copy import copy

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

import GetData
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd


import sklearn.model_selection as ms

if __name__ == "__main__":

    alphas = [2., 4., 8.]
    hiddens = [(h,) * l for l in [1, 2, 3] for h in [4, 4 // 2, 4 * 2]]
    hiddens += [(32,), (64,), (16,),(100,100,100),(60,60,60),(30,30,30),(120,120),(100,100),(30,30),(60,60)]

    params_mlp = {
        'MLP__solver':['lbfgs','adam'],
        # 'MLP__solver': ['lbfgs'],
        'MLP__alpha': alphas,
        # 'MLP__alpha': [2],
        'MLP__hidden_layer_sizes': hiddens,
        # 'MLP__hidden_layer_sizes': [(8,)],
        'MLP__random_state': [1],
        'MLP__activation':['identity','relu']
        # 'MLP__activation': ['relu']
    }

    games_list, teams_list = GetData.returnTeams()
    games_list = list(map(lambda x : x[1::],games_list))
    prettyPrinted = []

    rawScore = []
    for row in games_list:
        newRow = copy(row)
        score = (newRow[3] - newRow[6]) * 1.0
        if score < -20:
            score = -20
        if score > 20:
            score = 20

        del(newRow[6])
        del(newRow[3])
        newRow[0] = 0
        newRow[2] += 1.
        newRow[4] += 1.
        newRow.append(score)
        rawScore.append(newRow)
    games_list = rawScore



    # for row in games_list:
    #     row = copy(row)
    #     row[1] = teams_list[row[1]]
    #     row[1] = teams_list[row[4]]
    #
    #
    #     prettyPrinted.append(row)

    games_reversed = []
    for row in games_list:
        newRow = copy(row)
        newRow[1:3] = row[3:5]
        newRow[3:5] = row[1:3]
        newRow[5] =  row[5] * -1.0
        games_reversed.append(newRow)



    enc = OneHotEncoder(categorical_features=[1,3])


    games_total = games_list+games_reversed
    games = pd.DataFrame(games_total,columns=['Useless','TeamA','SiteA','TeamB','SiteB','ScoreDiff'])
    standardScaler = StandardScaler()
    games[['ScoreDiff']] = standardScaler.fit_transform(games[['ScoreDiff']].values)

    games_x = games.iloc[:,:-1]
    games_y = games.iloc[:,-1]
    new_features = enc.fit_transform(games_x).toarray()
    # newTest = enc.transform(games_x.iloc[0:1,:])
    # newTest = games_x.iloc[0:1,:]
    # print(enc.fit_transform(games_total).toarray())
    learner =Pipeline([('Scale', StandardScaler(with_mean=False)),
                                           ('MLP', MLPRegressor(max_iter=1000))])
    # learner =Pipeline([('Scale', StandardScaler(with_mean=False)),
    #                                        ('MLP', MLPRegressor(max_iter=2000,activation='relu'))])

    gs = ms.GridSearchCV(learner, params_mlp, cv=3, n_jobs=4,
                         verbose=10, scoring=None, refit='neg_mean_squared_error',
                         return_train_score=True)

    gs.fit(new_features,games_y)
    params = gs.best_params_
    estimator = gs.best_estimator_
    estimator.set_params(**params)
    print("BEST PARAMS!!!!!!!!!!! {}".format(params))
    estimator.fit(new_features,games_y)
    teamResults = {}
    teamSimulations = {}
    for teamId,teamName in teams_list.items():
        simulations = []
        #home team
        for awayId in teams_list.keys():
            if awayId == teamId:
                continue
            #Home
            simulations.append([0,teamId,2.0,awayId,0.0])
            # simulations.append([20180918,awayId,0.0,awayId,2.0])
            #neutral
            simulations.append([0, teamId, 1.0, awayId, 1.0])
            # simulations.append([20180918, awayId, 1.0, awayId, 1.0])
            #away
            simulations.append([0, teamId, 0.0, awayId, 2.0])
            # simulations.append([20180918, awayId, 2.0, awayId, 0.0])

        simulations = pd.DataFrame(simulations)
        t = enc.transform(simulations).toarray()
        predictions = estimator.predict(t)

        # predictions[predictions > 0] = 1
        # predictions[predictions < 0] = -1

        teamResults[teamId] = predictions.sum()
    teamCleanResults = []



    for teamId,Score in teamResults.items():
        teamCleanResults.append([teams_list[teamId],Score,teamId])
    sortedRankings = sorted(teamCleanResults,key=lambda x: x[1])

    print(simulations.iloc[0,:])
    print(games_x.iloc[0,:])
    for s  in sortedRankings:
        print(s)
    sortedRankings.reverse()



    i = 0
    for s  in sortedRankings[0:25]:
        i += 1
        print("{} {}".format(i,s))
    sortedRankings = sortedRankings[0:50]
    teamResults2 = {}
    for s in sortedRankings:
        simulations = []
        # home team
        for awayId in sortedRankings:
            if awayId[2] == s[2]:
                continue
            # Home
            simulations.append([0, s[2], 2.0, awayId[2], 0.0])
            # simulations.append([20180918,awayId[2],0.0,awayId[2],2.0])
            # neutral
            simulations.append([0, s[2], 1.0, awayId[2], 1.0])
            # simulations.append([20180918, awayId[2], 1.0, awayId[2], 1.0])
            # away
            simulations.append([0, s[2], 0.0, awayId[2], 2.0])
            # simulations.append([20180918, awayId[2], 2.0, awayId[2], 0.0])

        simulations = pd.DataFrame(simulations)
        t = enc.transform(simulations).toarray()
        predictions = estimator.predict(t)
        # predictions[predictions > 0] = 1
        # predictions[predictions < 0] = -1
        teamResults2[s[2]] = predictions.sum()

    teamCleanResults = []
    for teamId,Score in teamResults2.items():
        teamCleanResults.append([teams_list[teamId],Score,teamId])
    sortedRankings = sorted(teamCleanResults,key=lambda x: x[1])

    print(simulations.iloc[0,:])
    print(games_x.iloc[0,:])
    for s  in sortedRankings:
        print(s)
    sortedRankings.reverse()

    i = 0
    for s in sortedRankings[0:50]:
        i += 1
        print("{} {}".format(i, s))

    5+5# print(learner.predict(newTest))





