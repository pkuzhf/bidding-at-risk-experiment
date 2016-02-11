#!/usr/bin/python
import sys
import random
import math
import operator
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from multiprocessing import Process, Pool

bufferCaseNum = 1000000
eta = 0.01
lamb = 1E-6
#eta = 0.001
#lamb = 1e-7
featWeight = {}
trainRounds = 5
random.seed(10)
initWeight = 0.05

def nextInitWeight():
    return (random.random() - 0.5) * initWeight

def ints(s):
    res = []
    for ss in s:
        res.append(int(ss))
    return res

def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))


def run(c):
    trainfilename = str(c) + '/train.yzx.txt'
    testfilename = str(c) + '/test.yzx.txt'
    for round in range(0, trainRounds):
        # train for this round
        fi = open(trainfilename, 'r')
        lineNum = 0
        trainData = []
        count = 0
        for line in fi:
            lineNum = (lineNum + 1) % bufferCaseNum
            trainData.append(ints(line.replace(":1", "").split()))
            if lineNum == 0:
                for data in trainData:
                    clk = data[0]
                    mp = data[1]
                    fsid = 2 # feature start id
                    # predict
                    pred = 0.0
                    for i in range(fsid, len(data)):
                        feat = data[i]
                        if feat not in featWeight:
                            featWeight[feat] = nextInitWeight()
                        pred += featWeight[feat]
                    pred = sigmoid(pred)
                    # start to update weight
                    # w_i = w_i + learning_rate * [ (y - p) * x_i - lamb * w_i ] 
                    for i in range(fsid, len(data)):
                        feat = data[i]
                        featWeight[feat] = featWeight[feat] * (1 - lamb) + eta * (clk - pred)
                trainData = []
                count = count + bufferCaseNum
                print(str(c) + ' ' + str(count))

        if len(trainData) > 0:
            for data in trainData:
                clk = data[0]
                mp = data[1]
                fsid = 2 # feature start id
                # predict
                pred = 0.0
                for i in range(fsid, len(data)):
                    feat = data[i]
                    if feat not in featWeight:
                        featWeight[feat] = nextInitWeight()
                    pred += featWeight[feat]
                pred = sigmoid(pred)
                # start to update weight
                # w_i = w_i + learning_rate * [ (y - p) * x_i - lamb * w_i ]
                for i in range(fsid, len(data)):
                    feat = data[i]
                    featWeight[feat] = featWeight[feat] * (1 - lamb) + eta * (clk - pred)
        fi.close()

        # test for this round
        y = []
        yp = []
        fi = open(testfilename, 'r')
        for line in fi:
            data = ints(line.replace(":1", "").split())
            clk = data[0]
            mp = data[1]
            fsid = 2 # feature start id
            pred = 0.0
            for i in range(fsid, len(data)):
                feat = data[i]
                if feat in featWeight:
                    pred += featWeight[feat]
            pred = sigmoid(pred)
            y.append(clk)
            yp.append(pred)
        fi.close()
        auc = roc_auc_score(y, yp)
        fo = open(testfilename + '.lr.measure', 'a')
        rmse = math.sqrt(mean_squared_error(y, yp))
        fo.write(str(round) + '\t' + str(auc) + '\t' + str(rmse) + '\n')
        print(str(round) + '\t' + str(auc) + '\t' + str(rmse))
        fo.close()
    # output the weights
        fo = open(trainfilename + '.lr.weight', 'w')
        for i in featWeight:
            fo.write(str(i) + '\t' + str(featWeight[i]) + '\n')
        fo.close()


# output the prediction
    fi = open(testfilename, 'r')
    fo = open(testfilename + '.lr.pred', 'w')

    for line in fi:
        data = ints(line.replace(":1", "").split())
        pred = 0.0
        for i in range(1, len(data)):
            feat = data[i]
            if feat in featWeight:
                pred += featWeight[feat]
        pred = sigmoid(pred)
        fo.write(str(pred) + '\n')    
    fo.close()
    fi.close()


def main():
    if len(sys.argv) == 2:
        run(sys.argv[1])
        return
    processNum = 9
    #camps = [1458, 2259, 2261, 2821, 2997, 3358, 3386, 3427, 3476]
    camps = ['2997n', '2261n', '2259n', '1458n', '3427n', '3476n', '2821n', '3358n', '3386n']
    #camps = ['yoyi']
    args = []
    for c in camps:
        args.append([c])
    p = Pool(processNum)
    for arg in args:
        p.apply_async(run, arg)
    p.close()
    p.join()

if __name__ == '__main__':
    main()
