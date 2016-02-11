from math import exp,sqrt
import datetime
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve,lsqr
import numpy as np
from multiprocessing import Pool
import sys
import random 
import os

initMu = -1e-5
initPre = 1.0
sigmoidScale = 1.0
hasPrior = True
downrate = 1
testDataLimit = 500000

def ints(s):
    res = []
    for ss in s:
        if ss != 'null':
            res.append(int(ss))
    return res

def sigmoid(x):
    return 1.0 / (1.0 + exp(- sigmoidScale * x))

def calibrate(x):
    return x / (x + (1.0 - x) / downrate)

def readPriorWeight(camp):
    mu = {}
    pre = {}
    fi = open(str(camp) + '/train.yzx.txt.lr.weight')
    bias_line = fi.readline()
    [key, value] = bias_line.strip().split('\t')
    if key == 'bias':
        mu['bias'] = float(value)
        pre['bias'] = initPre
    else:
        mu['bias'] = 0
        pre['bias'] = initPre
        mu[int(key)] = float(value)
        pre[int(key)] = initPre
    for line in fi:
        [key, value] = line.strip().split('\t')
        if value == 'null':
            value = initMu
        mu[int(key)] = float(value)
        pre[int(key)] = initPre
    fi.close()
    return [mu, pre]

def calcAUC(mu, testData):
    y = []
    yp = []
    avgPred = 0.0
    for data in testData:
        clk = data[0]
        pred = 0.0
        pred += mu['bias']
        for i in range(2, len(data)):
            feat = data[i]
            if feat in mu:
                pred += mu[feat]
        #pred = calibrate(sigmoid(pred))
        pred = sigmoid(pred)
        avgPred += pred
        y.append(clk)
        yp.append(pred)
    return [roc_auc_score(y, yp), avgPred / len(testData)]

def run(camp, rounds):
    outputfilename = 'output/' + str(camp) + '.lap.bayes'
    if hasPrior:
        outputfilename = outputfilename + '.hasPrior'
    
    print(str(camp) + ' read train data... [' + str(datetime.datetime.now()) + ']')
    trainData = []
    fi = open(str(camp) + '/train.yzx.txt', 'r')
    
    for line in fi:
        data = ints(line.replace(":1", "").split())
        data.append('bias')
        if data[0] == 1 and random.random() > downrate:
            continue
        trainData.append(data)
    fi.close()

    if hasPrior == False:
        print(str(camp) + ' read test data... [' + str(datetime.datetime.now()) + ']')
        fi = open(str(camp) + '/test.yzx.txt', 'r')
        testData = []
        count = 0
        for line in fi:
            if count == testDataLimit:
                break
            count += 1
            data = ints(line.replace(":1", "").split())            
            testData.append(data)
        fi.close()
    
    print(str(camp) + ' prepare prior weight...')
    if hasPrior == True:
        [mu, pre] = readPriorWeight(camp)
    else:
        mu = {}
        mu['bias'] = initMu
        pre = {}
        pre['bias'] = initPre
        for data in trainData:
            for i in range(2, len(data)):
                feat = data[i]
                if feat not in mu:
                    mu[feat] = initMu
                    pre[feat] = initPre
        fi.close()

    lastUpdate = {}
    lastAUC = 0.0
    maxLife = 3
    life = maxLife
    converge = False
    nIter = 3 
    alpha = 0.00001
    eta = 0.00001
    nBatch = 10
    alphaDecreaseRate = 1.0
    alphaDecreaseFactor = 0.9
    nDecreaseAlpha = len(trainData) / 2
    countDecreaseAlpha = nDecreaseAlpha
    nTest = 100000

    if hasPrior == True:
        nIter = 0
        alpha = 0
        nBatch = 1
        nTest = 100000

    for round in range(rounds):
        print(str(camp) + ' round ' + str(round) + ': ' + str(datetime.datetime.now()))
        count = 0
        while count < len(trainData) and converge == False:
            oldMu = {}
            for i in range(count, min(count + nBatch, len(trainData))):
                data = trainData[i]
                for j in range(2, len(data)):
                    if data[j] not in oldMu:
                        oldMu[data[j]] = mu[data[j]]
            for feat in oldMu:
                if feat in lastUpdate:
                    n = (count - lastUpdate[feat] + len(trainData)) % len(trainData)
                    pre[feat] = (1 - eta)**n * (pre[feat] - initPre) + initPre
                lastUpdate[feat] = count


            for iter in range(nIter):
                for i in range(count, min(count + nBatch, len(trainData))):
                    data = trainData[i]
                    clk = data[0]
                    if clk == 0:
                        clk = -1
                    wx = 0.0
                    for j in range(2, len(data)):
                        feat = data[j]
                        if feat in mu:
                            wx += mu[feat]
                    for j in range(2, len(data)):
                        feat = data[j]
                        #print(str(pre[feat]) + ' ' + str(mu[feat]) + ' ' + str(oldMu[feat]) + ' ' + str(wx))
                        mu[feat] -= alpha * (2 * (mu[feat] - oldMu[feat]) * pre[feat] - clk * sigmoidScale / (1 + exp(sigmoidScale * clk * wx)))
            for i in range(count, min(count + nBatch, len(trainData))):
                data = trainData[i]
                wx = 0.0
                for j in range(2, len(data)):
                    feat = data[j]
                    if feat in mu:
                        wx += mu[feat]
                        #print(mu[feat])
                sig = sigmoid(wx)
                for j in range(2, len(data)):
                    feat = data[j]
                    pre[feat] += sigmoidScale**2 * sig * (1 - sig)

            count += nBatch
            if count % nTest < nBatch:
                print(str(count) + '/' + str(len(trainData)))
                print(['avg mu', np.mean(list(mu.values())), 'avg pre', np.mean(list(pre.values()))])
                print(['std mu', np.std(list(mu.values())), 'avg std', np.std(list(pre.values()))])
                if hasPrior == False:
                    [auc, avgPred] = calcAUC(mu, testData)
                    if auc < lastAUC:
                        life -= 1
                    else:
                        life = maxLife
                    if life == maxLife - 2:
                        alpha *= alphaDecreaseFactor
                    if life == 0:
                        converge = True
                    lastAUC = auc
                    print(str(camp) + '\tauc ' + str(auc) + '\tavgPred ' + str(avgPred) + '\t[' + str(datetime.datetime.now()) + ']')
                    fo = open(outputfilename + '.measure', 'a')
                    fo.write('round ' + str(round) + '\tauc ' + str(auc) + '\tavgPred ' + str(avgPred) + '\n')
                    fo.close()
            if countDecreaseAlpha <= 0:
                alpha *= alphaDecreaseRate
                countDecreaseAlpha += nDecreaseAlpha
        
        if converge == True:
            break
    


    fo = open(outputfilename + '.weight', 'w')
    fo.write(str(mu['bias']) + '\n')

    if os.path.exists(str(camp) + '/featindex.txt'):
        nFeat = int(os.popen('wc -l ' + str(camp) + '/featindex.txt').readline().split()[0])
        i = 0    
        while i < nFeat:
            if i in mu:
                fo.write(str(mu[i]) + '\t' + str(pre[i]) + '\n')
            else:
                fo.write('0\t0\n')
            i += 1
    else:
        count = 1
        i = 0    
        while count < len(mu):
            if i in mu:
                fo.write(str(mu[i]) + '\t' + str(pre[i]) + '\n')
                count += 1
            else:
                fo.write('0\t0\n')
            i += 1
    fo.close()

def main():
    if len(sys.argv) >= 2:
        if len(sys.argv) >= 3:
            run(sys.argv[1], int(sys.argv[2]))
        else:
            run(sys.argv[1], 5)
        return
    processNum = 9
    #camps = [1458, 3386, 3427, 3476, 3358, 2821, 2259, 2261, 2997]
    camps = ['2997n', '2261n', '2259n', '1458n', '3427n', '3476n', '2821n', '3358n', '3386n']
    p = Pool(processNum)
    for c in camps:
        p.apply_async(run, [c,5])
    p.close()
    p.join()

if __name__ == '__main__':
    main()
