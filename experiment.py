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

#pinyou
varScale = 0.1
preOffset = 9
#yoyi
#varScale = 1
#preOffset = 0

hasPrior = True
transMap = {}
mapSize = 0


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

def round_to_n(x, n):
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))

def trans(mean, var):
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
        # if mapSize % 10000 == 0:
        #     print(str(['mapsize', mapSize]))
    return transMap[mean][var]

def dataStat(camp, featMean, featVar, bias): 
    print('dataStat... ' + str(camp))
    sumMP = 0
    sumClick = 0
    fi = open(str(camp) + '/train.yzx.txt', 'r')
    sumMean = 0.0
    sumStd = 0.0
    mpDis = []
    count = 0
    for line in fi:
        data = ints(line.replace(":1", "").split())
        sumClick += data[0]
        sumMP += data[1]
        mpDis.append(data[1])
        mean = bias
        var = 0.0
        for i in range(2, len(data)):
            feat = data[i]
            if feat in featMean:
                mean += featMean[feat]
            if feat in featVar:
                var += featVar[feat]
        if count % 100 == 0:
            [mean, var] = trans(mean, var)
            std = sqrt(var)
            sumStd += std
            sumMean += mean
        count += 1
    fi.close()
    v = 1.0 * sumMP / sumClick
    mpDis.sort()
    l = mpDis[int((len(mpDis)-1) * 0.999)]
    adjust = abs(sumStd / sumMean)
    return [l, v, adjust, mpDis]

def replayVaR(camp, parameters, featMean, featVar, bias, v, adjust, budget):
    print(['replayVaR', camp, parameters, budget])
    impSums = [0 for i in range(len(parameters))]
    clkSums = [0 for i in range(len(parameters))]
    costSums = [0 for i in range(len(parameters))]
    bidPrices = [[] for i in range(len(parameters))]
    count = 0
    fi = open(str(camp) + '/test.yzx.txt.half2', 'r')
    for line in fi:
        if count % 10000 == 0:
            print(['replayVaR', camp, budget, count, str(datetime.datetime.now())])
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
        [mean, var] = trans(mean, var)
        std = sqrt(var)
        for i in range(len(parameters)):
            if budget > 0 and costSums[i] > budget:
                continue
            bidScale = parameters[i][0]
            alpha = parameters[i][1]
            ctr = mean + alpha * std / adjust
            ctr = min(1, ctr)
            ctr = max(0, ctr)
            bid = bidScale * v * ctr
            bidPrices[i].append(bid)
            if bid >= mp:
                impSums[i] += 1
                costSums[i] += mp
                clkSums[i] += clk
    fi.close()
    return [impSums, clkSums, costSums, count, bidPrices]

def laplaceSmoothing(mpDis, l, extendedL):
    n = len(mpDis) * 0.1
    for i in range(extendedL):
        for j in range(int(n / extendedL)):
            mpDis.append(i)
    return mpDis

def getFeatWeight(camp):
    print('reading feat weight...')
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
            featVar[i] = 1.0 / (float(line.split()[1])) * varScale
        i += 1
    fi.close()
    return [featMean, featVar, bias]

def findb(mean, var, v, mpDis, alphas, left, right, step):
    max = {}
    choice = {}
    for alpha in alphas:
        choice[alpha] = -1
        max[alpha] = -1000000.0

    n = 1000
    r = [0 for i in range(n)]
    r_mp = [[0,0] for i in range(n)]
    realR = [0 for i in range(n)]
    meanCTR = 0.0
    for i in range(n):
        mp = mpDis[int(np.random.random() * len(mpDis))]
        ctr = sigmoid(np.random.normal(mean, var))
        r_mp[i] = [v*ctr-mp, mp]
        meanCTR += ctr
    meanCTR /= n
    r_mp = sorted(r_mp, key=lambda iter: iter[1])

    i = 0
    sum = 0.0
    sum2 = 0.0
    for b in arange(left, right, step):
        while i < n and b >= r_mp[i][1]:
            realR[i] = r_mp[i][0]
            sum += realR[i]
            sum2 += realR[i]**2
            i += 1

        [rMean, rStd] = [sum / n, sqrt(sum2 / n - (sum / n)**2)]
        for alpha in alphas:
            u = rMean + alpha * rStd
            if u > max[alpha]:
                max[alpha] = u
                choice[alpha] = b
    return [choice, (v * meanCTR - choice[alphas[int(len(alphas) / 2)]])]

def replayRMR(camp, parameters, featMean, featVar, mpDis, v, l, bias, budget):
    print(['replayRMR', camp, parameters, budget])
    impSums = [0 for i in range(len(parameters))]
    clkSums = [0 for i in range(len(parameters))]
    costSums = [0 for i in range(len(parameters))]
    bidPrices = [[] for i in range(len(parameters))]
    count = 0
    fi = open(str(camp) + '/test.yzx.txt.half2', 'r')
    bidMap = {}
    mapSize = 0
    for line in fi:
        count += 1
        if count % 1000 == 0:
            print(['replayRMR', camp, budget, count, str(datetime.datetime.now())])    
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
        mean = round_to_n(mean, 3)
        var = round_to_n(var, 3)
        if mean not in bidMap:
            bidMap[mean] = {}
        if var not in bidMap[mean]:
            left = 0
            right = l
            step = (left + right) / 300.0
            [bidMap[mean][var], diff] = findb(mean, var, v, mpDis, [alpha for [bidScale, alpha, lamb] in parameters], left, right, step)
            mapSize += 1
            # if mapSize % 1000 == 0:
            #     print([camp, 'bidMap size', mapSize])

        for i in range(len(parameters)):
            if budget > 0 and costSums[i] > budget:
                continue
            bidScale = parameters[i][0]
            alpha = parameters[i][1]
            bid = bidScale * bidMap[mean][var][alpha]
            bidPrices[i].append(bid)
            if bid >= mp:
                impSums[i] += 1
                costSums[i] += mp
                clkSums[i] += clk
    fi.close()
    return [impSums, clkSums, costSums, count, bidPrices]

def getOutputFilename(budget):
    filename = '../exp/'
    if budget < 0:
        filename = filename + 'non-'
    filename = filename + 'budget-test-half2.txt'
    return filename
    
def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def writeResult(camp, strategy, budget0, budget_scale, parameters, impSums, clkSums, costSums, count, v, bidPrices):
    budget = budget0 * budget_scale
    filename = getOutputFilename(budget)
    for i in range(len(parameters)):
        bidScale = parameters[i][0]
        alpha = parameters[i][1]
        lamb = parameters[i][2]
        impSum = impSums[i]
        clkSum = clkSums[i]
        costSum = costSums[i]
        if clkSum != 0:
            ecpc = 1.0 * costSum / clkSum
        else:
            ecpc = 'inf'
        fo = open(filename, 'a')
        ctr = 1. * clkSum / impSum
        cpm = 1000. * costSum / impSum
        fo.write('\t'.join([str(camp), str(strategy), str(lamb), str(alpha), str(bidScale), str(costSum), str(clkSum), str(ecpc), str(v * clkSum - costSum), \
            str(1.0 * impSum / count), str((v * clkSum - costSum) / (costSum + 1)), str(ctr), str(cpm), str(budget0), str(budget_scale)]) + '\n')
        fo.close()

        dir = '../exp/bid-price/'
        mkdir(dir)
        if budget0 < 0 :
            fo = open(dir + '-'.join(['bid-price', camp, strategy, 'non', str(lamb)]) + '.txt', 'w')
        else:
            fo = open(dir + '-'.join(['bid-price', camp, strategy, str(int(1 / budget_scale)), str(lamb)]) + '.txt', 'w')
        fo.write('\t'.join([str(bidScale), str(alpha), str(budget0), str(budget_scale)]) + '\n')
        sample_size = 10000
        samples = np.random.randint(len(bidPrices[i]), size=sample_size)
        for sample in samples:
            fo.write(str(bidPrices[i][sample]) + '\n')
        fo.close()

def runRMR(camp, parameters, budget0, budget_scale, featMean, featVar, bias, l, v, mpDis):
    print(['runRMR', camp, budget_scale, str(datetime.datetime.now())])
    extendedL = int(l / 0.1)
    mpDis = laplaceSmoothing(mpDis, l, extendedL)
    [impSums, clkSums, costSums, count, bidPrices] = replayRMR(camp, parameters, featMean, featVar, mpDis, v, extendedL, bias, budget0 * budget_scale)
    writeResult(camp, 'rmr', budget0, budget_scale, parameters, impSums, clkSums, costSums, count, v, bidPrices)

def runVaR(camp, parameters, budget0, budget_scale, featMean, featVar, bias, v, adjust):
    print(['runVaR', camp, budget_scale, str(datetime.datetime.now())])
    [impSums, clkSums, costSums, count, bidPrices] = replayVaR(camp, parameters, featMean, featVar, bias, v, adjust, budget0 * budget_scale)
    writeResult(camp, 'var', budget0, budget_scale, parameters, impSums, clkSums, costSums, count, v, bidPrices)

def runLR(camp, parameters, budget0, budget_scale, v):
    print(['runLR', camp, budget_scale, str(datetime.datetime.now())])
    budget = budget0 * budget_scale
    pred = []
    fi = open(str(camp) + '/test.yzx.txt.lr.pred.half2', 'r')
    for line in fi:
        pred.append(float(line))
    fi.close()
    fi = open(str(camp) + '/test.yzx.txt.half2', 'r')
    count = 0
    impSums = [0 for i in range(len(parameters))]
    clkSums = [0 for i in range(len(parameters))]
    costSums = [0 for i in range(len(parameters))]
    bidPrices = [[] for i in range(len(parameters))]
    for line in fi:
        data = ints(line.replace(":1", "").split())
        clk = data[0]
        mp = data[1]
        for i in range(len(parameters)):
            if budget > 0 and costSums[i] > budget:
                continue
            bidScale = parameters[i][0]
            bid = bidScale * v * pred[count]
            bidPrices[i].append(bid)
            if bid >= mp:
                impSums[i] += 1
                costSums[i] += mp
                clkSums[i] += clk
        count += 1
    for i in range(len(parameters)):    
        clkSum = clkSums[i]
        costSum = costSums[i]
        impSum = impSums[i]
        if clkSum != 0:
            ecpc = costSum / clkSum
        else:
            ecpc = 'inf'
        if impSum != 0:
            ctr = clkSum / float(impSum)
        else:
            ctr = 'inf'
    writeResult(camp, 'lr', budget0, budget_scale, parameters, impSums, clkSums, costSums, count, v, bidPrices)

def buildDict(head, data):
    head = head.split()
    data = data.split()
    ret = {}
    for i in range(len(head)):
        ret[head[i]] = data[i]
    return ret

def readData(filename):
    data_bid = []
    fi = open(filename, 'r')
    head = fi.readline()
    for line in fi:
        data_bid.append(buildDict(head, line))
    return data_bid

def writeSelectedParameters(lamb, optBids1, optBids2, optLR):
    fo = open('../exp/tex.txt', 'a')
    fo.write('\\begin{table}\n')
    fo.write('\\centering\n')
    fo.write('\\caption{Optimaze net profit - ' + str(lamb) + ' cost}\\label{tab:optimize-lamb-' + str(lamb) + '}\n')
    fo.write('\\begin{tabular}{c|ccc}\n')
    fo.write('camp & LR & VaR & RM\\\\\n')
    fo.write('\\hline\n')
    for camp in optBids1:
        fo.write(str(camp) + ' & ' + str("{:.2E}".format(optLR[camp][0])) + ' & ' + \
            str("{:.2E}".format(optBids1[camp][0])) + ' & ' + str("{:.2E}".format(optBids2[camp][0])) + '\\\\\n')
    fo.write('\\end{tabular}\n')
    fo.write('\\end{table}\n')
    fo.write('\n')
    fo.close()

def nonBudgetRun(camp, var_parameters, rmr_parameters):
    [featMean, featVar, bias] = getFeatWeight(camp)
    [l, v, adjust, mpDis] = dataStat(camp, featMean, featVar, bias)
    budget0 = -1
    budget_scale = 1
    runVaR(camp, var_parameters[camp], budget0, budget_scale, featMean, featVar, bias, v, adjust)
    runRMR(camp, rmr_parameters[camp], budget0, budget_scale, featMean, featVar, bias, l, v, mpDis)
    runLR(camp, [[1.0, '-', '-']], budget0, budget_scale, v)

def budgetRun(camp, var_parameters, rmr_parameters, lr_parameters, budget0, budget_scale):
    [featMean, featVar, bias] = getFeatWeight(camp)
    [l, v, adjust, mpDis] = dataStat(camp, featMean, featVar, bias)
    runVaR(camp, var_parameters, budget0, budget_scale, featMean, featVar, bias, v, adjust)
    runRMR(camp, rmr_parameters, budget0, budget_scale, featMean, featVar, bias, l, v, mpDis)
    runLR(camp, lr_parameters, budget0, budget_scale, v)

def nonBudgetExperiment():
    inputfilename = '../exp/select-non-budget-parameters.txt'
    data = readData(inputfilename)
    var_parameters = {}
    rmr_parameters = {}
    for d in data:
        camp = d['camp']
        alpha = float(d['alpha'])
        bidScale = float(d['bidScale'])
        lamb = float(d['lambda'])
        if d['strategy'] == 'var':
            if camp not in var_parameters:
                var_parameters[camp] = []
            var_parameters[camp].append([bidScale, alpha, lamb])
        elif d['strategy'] == 'rmr':
            if camp not in rmr_parameters:
                rmr_parameters[camp] = []
            rmr_parameters[camp].append([bidScale, alpha, lamb])

    print(['var', var_parameters])
    print(['rmr', rmr_parameters])

    budget = -1
    fo = open(getOutputFilename(budget), 'w')
    fo.write('\t'.join(['camp', 'strategy', 'lambda', 'alpha', 'bidScale', 'cost', 'click', 'ecpc', 'revenue', \
            'winRate', 'roi', 'ctr', 'cpm', 'budget0', 'budgetScale']) + '\n')
    fo.close()

    if len(sys.argv) == 2:
        nonBudgetRun(sys.argv[1], var_parameters, rmr_parameters)
        return

    processNum = len(var_parameters)
    p = Pool(processNum)
    for camp in var_parameters:
        p.apply_async(nonBudgetRun, [camp, var_parameters, rmr_parameters])
    p.close()
    p.join()

def budgetExperiment():
    inputfilename = '../exp/select-budget-parameters.txt'
    data = readData(inputfilename)
    var_parameters = {}
    rmr_parameters = {}
    lr_parameters = {}
    budget0s = {}
    for d in data:
        camp = d['camp']
        lamb = float(d['lambda'])
        bidScale = float(d['bidScale'])
        budget0 = float(d['budget0'])
        budget_scale = float(d['budgetScale'])
        budget0s[camp] = budget0
        if d['strategy'] == 'var':
            alpha = float(d['alpha'])
            if camp not in var_parameters:
                var_parameters[camp] = {}
            if budget_scale not in var_parameters[camp]:
                var_parameters[camp][budget_scale] = []
            var_parameters[camp][budget_scale].append([bidScale, alpha, lamb])
        elif d['strategy'] == 'rmr':
            alpha = float(d['alpha'])
            if camp not in rmr_parameters:
                rmr_parameters[camp] = {}
            if budget_scale not in rmr_parameters[camp]:
                rmr_parameters[camp][budget_scale] = []
            rmr_parameters[camp][budget_scale].append([bidScale, alpha, lamb])
        elif d['strategy'] == 'lr':
            if camp not in lr_parameters:
                lr_parameters[camp] = {}
            if budget_scale not in lr_parameters[camp]:
                lr_parameters[camp][budget_scale] = []
            lr_parameters[camp][budget_scale].append([bidScale, 0, lamb])

    fo = open(getOutputFilename(budget=1), 'w')
    fo.write('\t'.join(['camp', 'strategy', 'lambda', 'alpha', 'bidScale', 'cost', 'click', 'ecpc', 'revenue', \
            'winRate', 'roi', 'ctr', 'cpm', 'budget0', 'budgetScale']) + '\n')
    fo.close()

    processNum = 20
    p = Pool(processNum)
    if len(sys.argv) == 2:
        camp = sys.argv[1]
        budget_scale = var_parameters[camp].keys()[0]
        budgetRun(camp, var_parameters[camp][budget_scale], rmr_parameters[camp][budget_scale], lr_parameters[camp][budget_scale], budget0s[camp], budget_scale)
        return
    for camp in var_parameters:
        for budget_scale in var_parameters[camp]:
            p.apply_async(budgetRun, [camp, var_parameters[camp][budget_scale], rmr_parameters[camp][budget_scale], lr_parameters[camp][budget_scale], \
                budget0s[camp], budget_scale])
        
    p.close()
    p.join()

def main():
    nonBudgetExperiment()
    #budgetExperiment()

if __name__ == '__main__':
    main()