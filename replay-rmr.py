import sys
import os
import datetime
import random
from math import e,log,copysign,pi,sqrt,floor,log10
from numpy import arange
from multiprocessing import Process, Pool
from math import exp,sqrt,pi
import numpy as np
import pickle

#yoyi
#varScale = 0.1
#pinyou
varScale = 1
hasPrior = True
outputfilename = '../exp/rmr.replay.budgeted.txt'

def sigmoid(p):
    return 1.0 / (1.0 + exp(-p))

def ints(s):
    res = []
    for ss in s:
        res.append(int(ss))
    return res

def round_to_n(x, n):
    return round(x, -int(floor(log10(abs(x)))) + (n - 1))

def laplaceSmoothing(mpDis, l, extendedL):
    n = len(mpDis) * 0.1
    for i in range(extendedL):
        for j in range(int(n / extendedL)):
            mpDis.append(i)
    return mpDis

def getBudget0(camp):
    budget0 = 0
    filename = str(camp) + '/test.yzx.txt.half1'
    fi = open(filename, 'r')
    for line in fi:
        mp = float(line.split()[1])
        budget0 += mp
    return budget0

def getbidMapFileName(camp):
    return 'revenue-dis-lookup-' + str(camp) + '.obj'

def loadBidMap(camp):
    bidMap = {}
    mapSize = 0
    if os.path.exists(getbidMapFileName(camp)):
        print('loading bidMap...')
        fi = open(getbidMapFileName(camp), 'rb')
        bidMap = pickle.load(fi)
        for mean in bidMap:
            mapSize += len(bidMap[mean])
        fi.close()
        print(str(mapSize) + ' records loaded.')
    return [bidMap, mapSize]

def saveBidMap(camp, bidMap):
    print('saving bidMap...')
    fo = open(getbidMapFileName(camp), 'wb')
    pickle.dump(bidMap, fo)
    fo.close()
    return

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

def dataStat(camp, featMean, featVar): 
    sumMP = 0
    sumClick = 0
    fi = open(str(camp) + '/train.yzx.txt', 'r')
    mpDis = []
    for line in fi:
        data = ints(line.replace(":1", "").split())
        sumClick += data[0]
        sumMP += data[1]
        mpDis.append(data[1])
    mpDis.sort()
    v = 1.0 * sumMP / sumClick
    l = mpDis[int((len(mpDis)-1) * 0.999)]
    #print(mpDis[len(mpDis) - 1])
    return [l, v, mpDis]

def findb(mean, var, v, mpDis, lambs, left, right, step):
    max = {}
    choice = {}
    for lamb in lambs:
        choice[lamb] = -1
        max[lamb] = -1000000.0

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
        for lamb in lambs:
            u = rMean + lamb * rStd
            if u > max[lamb]:
                max[lamb] = u
                choice[lamb] = b
    return [choice, (v * meanCTR - choice[lambs[int(len(lambs) / 2)]])]

def replayRisk(camp, lambs, featMean, featVar, mpDis, bidScales, v, l, bias, budget_scale):
    budget0 = getBudget0(camp)
    budget = budget0 * budget_scale
    costSums = {}
    clkSums = {}
    winCounts = {}
    for bidScale in bidScales:
        costSums[bidScale] = {}
        clkSums[bidScale] = {}
        winCounts[bidScale] = {}
        for lamb in lambs:
            costSums[bidScale][lamb] = 0.0
            clkSums[bidScale][lamb] = 0.0
            winCounts[bidScale][lamb] = 0.0
    budgetOfLastPoint = [budget for i in range(len(lambs))]
    idOfLastPoint = [len(bidScales) - 1 for i in range(len(lambs))]
    count = 0
    fi = open(str(camp) + '/test.yzx.txt.half1', 'r')
    #[bidMap, mapSize] = loadBidMap(camp)
    [bidMap, mapSize] = [{}, 0]
    sumdiff = 0.0
    for line in fi:
        #print(line)
        # if count == 5000:
        #     break
        if count % 1000 == 0:
            print([str(camp), budget_scale, str(count), str(datetime.datetime.now())])
            # print(budgetOfLastPoint)
            # print(idOfLastPoint)
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
        mean = round_to_n(mean, 3)
        var = round_to_n(var, 3)
        if mean not in bidMap:
            bidMap[mean] = {}
        if var not in bidMap[mean]:
            left = 0
            right = l
            step = (left + right) / 300.0
            [bidMap[mean][var], diff] = findb(mean, var, v, mpDis, lambs, left, right, step)
            sumdiff += diff
            # if diff > 30:
            #     break
            mapSize += 1
            if mapSize % 1000 == 0:
                print([camp, 'bidMap size', mapSize])
        
        i = len(lambs) - 1
        j = 0
        while i >= 0:
            lamb = lambs[i]
            while j < len(bidScales):
                bidScale = bidScales[j]
                bid = bidScale * bidMap[mean][var][lamb]
                if bid >= mp:
                    if j <= idOfLastPoint[i]:
                        winCounts[bidScale][lamb] += 1
                        costSums[bidScale][lamb] += mp
                        clkSums[bidScale][lamb] += clk
                        budgetOfLastPoint[i] -= mp
                    break
                j += 1
            while idOfLastPoint[i] >= 0 and budgetOfLastPoint[i] <= 0:
                bidScale = bidScales[idOfLastPoint[i]]
                budgetOfLastPoint[i] += costSums[bidScale][lamb]
                for j in range(idOfLastPoint[i]):
                    costSums[bidScale][lamb] += costSums[bidScales[j]][lamb]
                    winCounts[bidScale][lamb] += winCounts[bidScales[j]][lamb]
                    clkSums[bidScale][lamb] += clkSums[bidScales[j]][lamb]
                idOfLastPoint[i] -= 1
            i -= 1
    for i in range(len(lambs)):
        for j in range(1, idOfLastPoint[i] + 1):
            winCounts[bidScales[j]][lambs[i]] += winCounts[bidScales[j - 1]][lambs[i]]
            costSums[bidScales[j]][lambs[i]] += costSums[bidScales[j - 1]][lambs[i]]
            clkSums[bidScales[j]][lambs[i]] += clkSums[bidScales[j - 1]][lambs[i]]
    
    fi.close()

    for lamb in lambs:
        for bidScale in bidScales:
            clkSum = clkSums[bidScale][lamb]
            mpSum = costSums[bidScale][lamb]
            winCount = winCounts[bidScale][lamb]
            if clkSum != 0:
                ecpc = mpSum / clkSum
            else:
                ecpc = 'inf'
            if winCount != 0:
                ctr = 1.0 * clkSum / winCount
                cpm = 1.0 * mpSum / winCount
            else:
                ctr = 'inf'
                cpm = 'inf'
            #print(str(camp) + '\tlamb ' + str(lamb) + '\tbisScale ' + str(bidScale) + '\tcost ' + str(mpSum) + '\tclk ' + str(clkSum) + '\tecpc ' + str(ecpc) + '\trevenue ' + str(v * clkSum - mpSum) + '\twinRate ' + str(winCount / count))
            fo = open(outputfilename, 'a')
            fo.write('\t'.join([str(camp), str(lamb), str(bidScale), str(mpSum), str(ctr), str(clkSum), str(ecpc), str(v * clkSum - mpSum), \
                str((v * clkSum - mpSum) / (mpSum + 1)), str(winCount / count), str(cpm), str(budget0), str(budget_scale)]) + '\n')
            fo.close()
    #saveBidMap(camp, bidMap)
    return

def run(camp, budget_scale):
    random.seed(10)
    #pinyou
    #lambs = arange(-1, 1, 0.05)
    #yoyi
    lambs = arange(-1, 1, 0.05)
    #lambs = [0]
    bidScales = arange(0.1, 2, 0.02)
    
    print(str(camp) + '\t' + str(datetime.datetime.now()))
    [featMean, featVar, bias] = getFeatWeight(camp)
    [l, v, mpDis] = dataStat(camp, featMean, featVar)
    extendedL = int(l / min(bidScales))
    mpDis = laplaceSmoothing(mpDis, l, extendedL)
    replayRisk(camp, lambs, featMean, featVar, mpDis, bidScales, v, extendedL, bias, budget_scale)


def main():
    
    fo = open(outputfilename, 'w')
    fo.write('\t'.join(['camp', 'lamb', 'bidScale', 'cost', 'ctr', 'click', 'ecpc', 'revenue', 'roi', 'winRate', 'cpm', 'budget0', 'budgetScale']) + '\n')
    fo.close()
    budget_scales = [1. / 2**n for n in range(7)]

    if len(sys.argv) == 2:
        run(sys.argv[1], 0.00001)
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
