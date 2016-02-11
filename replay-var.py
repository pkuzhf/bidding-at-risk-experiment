import sys
import os
import datetime
from scipy import integrate
from scipy.special import psi,polygamma
from scipy import special
from math import e,log,copysign,pi,sqrt,floor,log10
from numpy import arange
from multiprocessing import Process, Pool
from math import exp, sqrt, pi
from sklearn.metrics import roc_auc_score
from numpy.random import normal
import numpy as np
import pickle

isTrans = True
# transMethod: 'int' or 'mc'
transMethod = 'mc' 
isXGmean = False
varScale = 0.1
downrate = 1
hasPrior = True
testDataLimit = 10000000

transMap = {}
mapSize = 0
outputfilename = '../exp/var.replay.budgeted.txt'


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def logit(x):
    return log(x / (1 - x))

def ints(s):
    res = []
    for ss in s:
        if ss != 'null':
            res.append(int(ss))
    return res
def calibrate(x):
    return x / (x + (1.0 - x) / downrate)

def round_to_n(x, n):
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))

def getBudget0(camp):
    budget0 = 0
    filename = str(camp) + '/test.yzx.txt.half1'
    fi = open(filename, 'r')
    for line in fi:
        mp = float(line.split()[1])
        budget0 += mp
    return budget0

def readWeightFromXG(camp):
    fi = open(str(camp) + '/xg.weight.txt', 'r')
    fi.readline()
    fi.readline()
    fi.readline()
    fi.readline()
    xgWeight = {}
    count = 0
    for line in fi:
        xgWeight[count] = float(line)
        count += 1
    fi.close()

    return xgWeight

    y = []
    yp = []
    fi = open(str(camp) + '/test.yzx.txt', 'r')
    for line in fi:
        data = ints(line.replace(":1", "").split())
        clk = data[0]
        pred = 0.0
        for i in range(2, len(data)):
            if data[i] in xgWeight:
                pred += xgWeight[data[i]]
        pred = sigmoid(pred)
        y.append(clk)
        yp.append(pred)
    fi.close()
    print(str(roc_auc_score(y, yp)))

def trans_int(mean, var):
    global transMap
    global mapSize
    #print([mean, var])
    mean = round_to_n(mean, 3)
    var = round_to_n(var, 3)
    #print([mean, var])
    if mean not in transMap:
        transMap[mean] = {}
    if var not in transMap[mean]:
        pctr1 = lambda x: x / sqrt(2 * pi * var) * exp(- (logit(x) - mean)**2 / (2 * var)) / (x - x**2)
        pctr2 = lambda x: x**2 / sqrt(2 * pi * var) * exp(- (logit(x) - mean)**2 / (2 * var)) / (x - x**2)
        meanTrans = integrate.quad(pctr1, 0.0, 1.0)[0]
        varTrans = integrate.quad(pctr2, 0.0, 1.0)[0] - meanTrans**2
        transMap[mean][var] = [meanTrans, varTrans]
        mapSize += 1
        if mapSize % 10000 == 0:
            print(str(['mapsize', mapSize]))
    return transMap[mean][var]

def trans_mc(mean, var):
    global transMap
    global mapSize
    mean = round_to_n(mean, 3)
    var = round_to_n(var, 3)
    if mean not in transMap:
        transMap[mean] = {}
    if var not in transMap[mean]:
        s = normal(mean, sqrt(var), 1000)
        for i in range(len(s)):
            s[i] = sigmoid(s[i])
        transMap[mean][var] = [np.mean(s), np.std(s)**2]
        mapSize += 1
        if mapSize % 10000 == 0:
            print(str(['mapsize', mapSize]))
    return transMap[mean][var]

def trans(mean, var):
    if transMethod == 'int':
        return trans_int(mean, var)
    if transMethod == 'mc':
        return trans_mc(mean, var)

def dataStat(camp, featMean, featVar, bias): 
    sumMP = 0
    sumClick = 0
    fi = open(str(camp) + '/train.yzx.txt', 'r')
    sumMean = 0.0
    sumStd = 0.0
    #mp = []
    count = 0
    for line in fi:
        data = ints(line.replace(":1", "").split())
        sumClick += data[0]
        sumMP += data[1]
        #mp.append(data[1])
        mean = bias
        var = 0.0
        for i in range(2, len(data)):
            feat = data[i]
            if feat in featMean:
                mean += featMean[feat]
            if feat in featVar:
                var += featVar[feat]
        if count % 100 == 0:
            if isTrans:
                [mean, var] = trans(mean, var)
            else:
                mean = sigmoid(mean)
            std = sqrt(var)
            sumStd += std
            sumMean += mean
        count += 1
    fi.close()
    v = 1.0 * sumMP / (sumClick * downrate)
    #mp.sort()
    adjust = abs(sumStd / sumMean)
    return [v, adjust]

def getFeatWeight(camp):
    featMean = {}
    featVar = {}
    if hasPrior:
        weightfilename = 'output/' + str(camp) + '.lap.bayes.hasPrior.weight'
    else:
        weightfilename = 'output/' + str(camp) + '.lap.bayes.weight'
    fi = open(weightfilename, 'r')
    bias = float(fi.readline())
    i = 0
    for line in fi:
        if line != '0\t0\n':
            featMean[i] = float(line.split()[0])
            featVar[i] = 1.0 / (float(line.split()[1]) - 9) * varScale
        i += 1
    fi.close()
    return [featMean, featVar, bias]

def replay(camp, lambs, bidScales, featMean, featVar, bias, v, adjust, budget_scale):
    budget0 = getBudget0(camp)
    budget = budget0 * budget_scale
    costSums = {}
    clkSums = {}
    winCounts = {}
    for lamb in lambs:
        costSums[lamb] = {}
        clkSums[lamb] = {}
        winCounts[lamb] = {}
        for bidScale in bidScales:
            costSums[lamb][bidScale] = 0
            clkSums[lamb][bidScale] = 0
            winCounts[lamb][bidScale] = 0
    budgetOfLastPoint = [budget for i in range(len(lambs))]
    idOfLastPoint = [len(bidScales) - 1 for i in range(len(lambs))]
    count = 0
    fi = open(str(camp) + '/test.yzx.txt.half1', 'r')
    fo = open(str(camp) + '/bid.price.txt', 'w')
    for line in fi:
        if count % 10000 == 0:
            print([camp, budget_scale, str(count), str(datetime.datetime.now())])
            # print(budgetOfLastPoint)
            # print(idOfLastPoint)
            if count >= testDataLimit:
                break
        count += 1
        data = ints(line.replace(":1", "").split())
        clk = data[0]
        mp = data[1]
        mean = bias
        var = 0.0
        for i in range(2, len(data)):
            feat = data[i]
            if feat in featMean:
                mean += featMean[feat]
            if feat in featVar:
                var += featVar[feat]
        #print([mean,var])
        if isTrans:
            [mean, var] = trans(mean, var)
        else:
            mean = sigmoid(mean)
        std = sqrt(var)
        #print([mean, std])
        #fo.write('mp ' + str(int(mp/1000)) + '\tbid ' + str(int(max(0, 0.29 * v *(mean -0.9 * std / adjust)) / 1000)) + '\n')
        #fo.write('mp ' + str(int(mp/1000)) + '\tbid ' + str(int(20000000 * mean / 1000)) + '\n')
        i = len(lambs) - 1
        j = 0
        while i >= 0:
            lamb = lambs[i]
            while j < len(bidScales):
                bidScale = bidScales[j]
                ctr = mean + lamb * std / adjust
                ctr = min(1, ctr)
                ctr = max(0, ctr)
                bid = bidScale * v * calibrate(ctr)
                #print([mean, std, lamb, adjust, bid])
                if bid >= mp:
                    if j <= idOfLastPoint[i]:
                        winCounts[lamb][bidScale] += 1
                        costSums[lamb][bidScale] += mp
                        clkSums[lamb][bidScale] += clk
                        budgetOfLastPoint[i] -= mp
                    break
                j += 1
            while idOfLastPoint[i] >= 0 and budgetOfLastPoint[i] <= 0:
                bidScale = bidScales[idOfLastPoint[i]]
                budgetOfLastPoint[i] += costSums[lamb][bidScale]
                for j in range(idOfLastPoint[i]):
                    costSums[lamb][bidScale] += costSums[lamb][bidScales[j]]
                    winCounts[lamb][bidScale] += winCounts[lamb][bidScales[j]]
                    clkSums[lamb][bidScale] += clkSums[lamb][bidScales[j]]
                idOfLastPoint[i] -= 1
            i -= 1
    for i in range(len(lambs)):
        for j in range(1, idOfLastPoint[i] + 1):
            winCounts[lambs[i]][bidScales[j]] += winCounts[lambs[i]][bidScales[j - 1]]
            costSums[lambs[i]][bidScales[j]] += costSums[lambs[i]][bidScales[j - 1]]
            clkSums[lambs[i]][bidScales[j]] += clkSums[lambs[i]][bidScales[j - 1]]
            
    fi.close()
    fo.close()

    for lamb in lambs:
        for bidScale in bidScales:
            clkSum = clkSums[lamb][bidScale]
            costSum = costSums[lamb][bidScale]
            winCount = winCounts[lamb][bidScale]
            if clkSum != 0:
                ecpc = 1.0 * costSum / clkSum
            else:
                ecpc = 'inf'
            if winCount != 0:
                ctr = 1.0 * clkSum / winCount
                cpm = 1.0 * costSum / winCount
            else:
                ctr = 'inf'
                cpm = 'inf'
            #print(str(camp) + '\t' + str(lamb) + '\t' + str(bidScale) + '\tcost ' + str(costSum) + '\tclk ' + str(clkSum) + '\tecpc ' + str(ecpc) + '\trevenue ' + str(v * clkSum - costSum) + '\twinRate ' + str(1.0 * winCount / count))
            fo = open(outputfilename, 'a')
            fo.write('\t'.join([str(camp), str(lamb), str(bidScale), str(costSum), str(ctr), str(clkSum), str(ecpc), str(v * clkSum - costSum), \
                str((v * clkSum - costSum) / (costSum+1)), str(1.0 * winCount / count), str(cpm), str(budget0), str(budget_scale)]) + '\n')
            fo.close()
    return 

def getTransMapFileName():
    return 'transMap.mean-sigma.obj'
    
def loadTransMap():
    global transMap
    global mapSize
    if os.path.exists(getTransMapFileName()):
        print('loading transMap...')
        fi = open(getTransMapFileName(), 'rb')
        transMap = pickle.load(fi)
        for meanIdx in transMap:
            mapSize += len(transMap[meanIdx])
        fi.close()
        print(str(mapSize) + ' records loaded.')
    return

def saveTransMap():
    print('saving transMap...')
    if os.path.exists(getTransMapFileName()):
        print('loading transMap...')
        fi = open(getTransMapFileName(), 'rb')
        oldTransMap = pickle.load(fi)
        fi.close()
        for mean in oldTransMap:
            for var in oldTransMap[mean]:
                if mean not in transMap:
                    transMap[mean] = {}
                if var not in transMap[mean]:
                    transMap[mean][var] = oldTransMap[mean][var]
    fo = open(getTransMapFileName(), 'wb')
    pickle.dump(transMap, fo)
    fo.close()
    return

def run(camp, budget_scale):
    if isTrans:
        lambs = np.concatenate((arange(-1, -0.05, 0.05), arange(-0.05, 0.05, 0.005), arange(0.05, 1, 0.05)))
        loadTransMap()
    else:
        lambs = arange(-0.0002, 0.0002, 0.00001)
    bidScales = arange(0.02, 2, 0.01)
    

    print(str(camp) + '\t' + str(datetime.datetime.now()))
    [featMean, featVar, bias] = getFeatWeight(camp)
    print([np.mean(list(featMean.values())), np.mean(list(featVar.values()))])
    if isXGmean:
        featMean = readWeightFromXG(camp)
    print('data statistic...')
    [v, adjust] = dataStat(camp, featMean, featVar, bias)
    print(str([v, adjust]))
    print('replay...')
    replay(camp, lambs, bidScales, featMean, featVar, bias, v, adjust, budget_scale)
    # if isTrans:
    #     saveTransMap()
def main():
    fo = open(outputfilename, 'w')
    fo.write('\t'.join(['camp', 'lamb', 'bidScale', 'cost', 'ctr', 'click', 'ecpc', 'revenue', 'roi', 'winRate', 'cpm', 'budget0', 'budgetScale']) + '\n')
    fo.close()
    budget_scales = [1. / 2**n for n in range(7)]
    if len(sys.argv) == 2:
        run(sys.argv[1], budget_scales[0])
        return
    camps = ['2997n', '2261n', '2259n', '1458n', '3427n', '3476n', '2821n', '3358n', '3386n']
    #camps = ['0113']
    processNum = 20
    p = Pool(processNum)
    for budget_scale in budget_scales:
        for c in camps:
            p.apply_async(run, [c, budget_scale])
    p.close()
    p.join()

if __name__ == '__main__':
    main()