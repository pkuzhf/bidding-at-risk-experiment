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

def findNonBudgetOptimalPara(data_bid, lamb):
    optimalBids = {}
    for data in data_bid:
        if float(data['bidScale']) != 1 or float(data['budgetScale']) != 1:
            continue
        value = float(data['revenue']) - lamb * float(data['cost'])
        camp = data['camp']
        if camp not in optimalBids or optimalBids[camp][0] < value:
            optimalBids[camp] = [value, data]
    return optimalBids

def findBudgetOptimalPara(data_bid, lamb):
    optimalBids = {}
    for data in data_bid:
        cost = float(data['cost'])
        revenue = float(data['revenue'])
        budget_scale = float(data['budgetScale'])
        camp = data['camp']
        value = float(data['revenue']) - lamb * cost
        if camp not in optimalBids:
            optimalBids[camp] = {}
        if budget_scale not in optimalBids[camp] or optimalBids[camp][budget_scale][0] < value:
            optimalBids[camp][budget_scale] = [value, data]
    return optimalBids

def writeSelectedParametersInLatex(lamb, optVaR, optRMR, optLR):
    fo = open('../exp/tex.txt', 'a')
    fo.write('\\begin{table}\n')
    fo.write('\\centering\n')
    fo.write('\\caption{Optimaze net profit - ' + str(lamb) + ' cost}\\label{tab:optimize-lamb-' + str(lamb) + '}\n')
    fo.write('\\begin{tabular}{c|ccc}\n')
    fo.write('camp & LR & VaR & RM\\\\\n')
    fo.write('\\hline\n')
    for camp in optVaR:
        fo.write(str(camp) + ' & ' + str("{:.2E}".format(optLR[camp][0])) + ' & ' + \
            str("{:.2E}".format(optVaR[camp][0])) + ' & ' + str("{:.2E}".format(optRMR[camp][0])) + '\\\\\n')
    fo.write('\\end{tabular}\n')
    fo.write('\\end{table}\n')
    fo.write('\n')
    fo.close()

def writeNonBudgetedSelectedParameters(filename, lamb, optVaR, optRMR, optLR):
    fo = open(filename, 'a')
    for camp in optLR:
        fo.write('\t'.join([camp, 'lr', str(lamb), str(optLR[camp][0]), str(optLR[camp][1]['bidScale']), '0', \
            str(optLR[camp][1]['cost']), str(optLR[camp][1]['revenue'])]) + '\n')
        fo.write('\t'.join([camp, 'var', str(lamb), str(optVaR[camp][0]), str(optVaR[camp][1]['bidScale']), str(optVaR[camp][1]['lamb']), \
            str(optVaR[camp][1]['cost']), str(optVaR[camp][1]['revenue'])]) + '\n')
        fo.write('\t'.join([camp, 'rmr', str(lamb), str(optRMR[camp][0]), str(optRMR[camp][1]['bidScale']), str(optRMR[camp][1]['lamb']), \
            str(optRMR[camp][1]['cost']), str(optRMR[camp][1]['revenue'])]) + '\n')

def getBudget0(data_lr):
    camps = []
    for d in data_lr:
        camp = d['camp']
        if camp not in camps:
            camps.append(camp)
    print(camps)
    budget0 = {}
    for camp in camps:
        budget0[camp] = 0
        filename = str(camp) + '/test.yzx.txt.half1'
        fi = open(filename, 'r')
        for line in fi:
            mp = float(line.split()[1])
            budget0[camp] += mp
    return budget0

def main():
    data_var = readData('../exp/var.replay.budgeted.txt')
    data_rmr = readData('../exp/rmr.replay.budgeted.txt')
    data_lr = readData('../exp/lr.replay.budgeted.txt')

    var_parameters = {}
    rmr_parameters = {}
    # lambs = np.arange(0, 1.1, 0.2) # for selected model comparison
    lambs = np.arange(0, 0.6, 0.2) # for tangent plotting
    outputfilename = '../exp/select-non-budget-parameters.txt'
    fo = open(outputfilename, 'w')
    fo.write('\t'.join(['camp', 'strategy', 'lambda', 'value', 'bidScale', 'alpha', 'cost', 'revenue']) + '\n')
    fo.close()
    for lamb in lambs:
        optVaR = findNonBudgetOptimalPara(data_var, lamb)
        optRMR = findNonBudgetOptimalPara(data_rmr, lamb)
        optLR = findNonBudgetOptimalPara(data_lr, lamb)
        writeNonBudgetedSelectedParameters(outputfilename, lamb, optVaR, optRMR, optLR)

    outputfilename = '../exp/select-budget-parameters.txt'
    fo = open(outputfilename, 'w')
    fo.write('\t'.join(['camp', 'strategy', 'lambda', 'bidScale', 'alpha', 'cost', 'revenue', 'budget0', 'budgetScale']) + '\n')
    lambs = [0, 0.2, 0.4]
    selected_parameters = {}
    var_parameters = {}
    rmr_parameters = {}
    lr_parameters = {}
    count = 0
    for lamb in lambs:
        optimal_para = findBudgetOptimalPara(data_var, lamb)
        count += 9 - len(optimal_para)
        for camp in optimal_para:
            for budget_scale in optimal_para[camp]:
                data = optimal_para[camp][budget_scale][1]
                fo.write('\t'.join([str(camp), 'var', str(lamb), data['bidScale'], data['lamb'], data['cost'], data['revenue'], data['budget0'], \
                    data['budgetScale']]) + '\n')
        optimal_para = findBudgetOptimalPara(data_rmr, lamb)
        count += 9 - len(optimal_para)
        for camp in optimal_para:
            for budget_scale in optimal_para[camp]:
                data = optimal_para[camp][budget_scale][1]
                fo.write('\t'.join([str(camp), 'rmr', str(lamb), data['bidScale'], data['lamb'], data['cost'], data['revenue'], data['budget0'], \
                    data['budgetScale']]) + '\n')
        optimal_para = findBudgetOptimalPara(data_lr, lamb)
        count += 9 - len(optimal_para)
        for camp in optimal_para:
            for budget_scale in optimal_para[camp]:
                data = optimal_para[camp][budget_scale][1]
                fo.write('\t'.join([str(camp), 'lr', str(lamb), data['bidScale'], '-', data['cost'], data['revenue'], data['budget0'], \
                    data['budgetScale']]) + '\n')
    print(count / len(lambs))
    fo.close()

if __name__ == '__main__':
    main()