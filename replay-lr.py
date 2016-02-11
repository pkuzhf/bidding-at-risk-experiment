from math import exp
from numpy import arange
import numpy as np
from multiprocessing import Process, Pool
import sys

def ints(s):
    res = []
    for ss in s:
        res.append(int(ss))
    return res

def sigmoid(p):
    return 1.0 / (1.0 + exp(-p))

outputfilename = '../exp/lr.replay.budgeted.txt'

def getV(camp): 
    sumMP = 0
    sumClick = 0
    fi = open(str(camp) + '/train.yzx.txt', 'r')
    for line in fi:
        data = ints(line.replace(":1", "").split())
        sumClick += data[0]
        sumMP += data[1]
    v = 1.0 * sumMP / sumClick
    return v

def getBudget0(camp):
    budget0 = 0
    filename = str(camp) + '/test.yzx.txt.half1'
    fi = open(filename, 'r')
    for line in fi:
        mp = float(line.split()[1])
        budget0 += mp
    return budget0

def run(camp, budget_scales):
    print([camp, budget_scales])
    budget0 = getBudget0(camp)
    budgets = []
    for budget_scale in budget_scales:
        budgets.append(budget0 * budget_scale)
    fo = open(outputfilename, 'a')
    v = getV(camp)
    pred = []
    fi = open(str(camp) + '/test.yzx.txt.lr.pred.half1', 'r')
    for line in fi:
        pred.append(float(line))
    fi.close()
    fi = open(str(camp) + '/test.yzx.txt.half1', 'r')
    count = 0
    bidScales = np.append(arange(0.01, 0.2, 0.01), arange(0.2, 3, 0.1))
    mpSum = [{} for i  in range(len(budget_scales))]
    clkSum = [{} for i  in range(len(budget_scales))]
    impSum = [{} for i  in range(len(budget_scales))]
    for i in range(len(budget_scales)):
        for c in bidScales:
            mpSum[i][c] = 0.
            clkSum[i][c] = 0.
            impSum[i][c] = 0.
    for line in fi:
        data = ints(line.replace(":1", "").split())
        clk = data[0]
        mp = data[1]
        for c in bidScales:
            bid = c * v * pred[count]
            for i in range(len(budget_scales)):
                if mpSum[i][c] + mp <= budgets[i] and bid >= mp:
                    impSum[i][c] += 1
                    mpSum[i][c] += mp
                    clkSum[i][c] += clk
        count += 1
    for i in range(len(budget_scales)):
        for c in bidScales:
            if clkSum[i][c] != 0:
                ecpc = mpSum[i][c] / clkSum[i][c]
            else:
                ecpc = 'inf'
            if impSum[i][c] != 0:
                ctr = clkSum[i][c] / float(impSum[i][c])
                cpm = mpSum[i][c] / float(impSum[i][c])
            else:
                ctr = 'inf'
                cpm = 'inf'
            fo.write('\t'.join([str(camp), str(c), str(mpSum[i][c]), str(ctr), str(clkSum[i][c]), str(ecpc), str(v * clkSum[i][c] - mpSum[i][c]),\
             str((v * clkSum[i][c] - mpSum[i][c]) / (mpSum[i][c]+1)), str(impSum[i][c] / count), str(cpm), str(budget0), str(budget_scales[i])]) + '\n')
            # print('\t'.join([str(camp), str(c), str(mpSum[i][c]), str(ctr), str(clkSum[i][c]), str(ecpc), str(v * clkSum[i][c] - mpSum[i][c]),\
            #  str((v * clkSum[i][c] - mpSum[i][c]) / (mpSum[i][c]+1)), str(impSum[i][c] / count), str(cpm), str(budget0), str(budget_scales[i])]))
    fo.close()
    print([camp, 'finished.'])

def main():
    camps = ['2997n', '2261n', '2259n', '1458n', '3427n', '3476n', '2821n', '3358n', '3386n']
    budget_scales = [1. / 2**n for n in range(7)]
    
    fo = open(outputfilename, 'w')
    fo.write('\t'.join(['camp', 'bidScale', 'cost', 'ctr', 'click', 'ecpc', 'revenue', 'roi', 'winRate', 'cpm', 'budget0', 'budgetScale']) + '\n')
    fo.close()
    if len(sys.argv) == 2:
        camp = sys.argv[1]
        run(camp, budget_scales)
        return
    
    #camps = ['0113']
    processNum = 9
    p = Pool(processNum)
    for c in camps:
        p.apply_async(run, [c, budget_scales])
    p.close()
    p.join()

if __name__ == '__main__':
    main()