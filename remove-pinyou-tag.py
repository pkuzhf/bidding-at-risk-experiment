from multiprocessing import Process, Pool
import sys

def run(c):
    fi = open(str(c) + '/featindex.txt', 'r')
    tagindex = {}
    for line in fi:
        if line[0:3] == '26:':
            tagindex[line.split()[1]] = 1
    fi.close()
    fi = open(str(c) + '/train.yzx.txt', 'r')
    fo = open(str(c) + 'n/train.yzx.txt', 'w')
    for line in fi:
        line = line.split()
        fo.write(line[0] + ' ' + line[1])
        for i in range(2, len(line)):
            word = line[i]
            if word.replace(':1', '') not in tagindex:
                fo.write(' ' + word)
        fo.write('\n')
    fi = open(str(c) + '/test.yzx.txt', 'r')
    fo = open(str(c) + 'n/test.yzx.txt', 'w')
    for line in fi:
        line = line.split()
        fo.write(line[0] + ' ' + line[1])
        for i in range(2, len(line)):
            word = line[i]
            if word.replace(':1', '') not in tagindex:
                fo.write(' ' + word)
        fo.write('\n')
    print(str(c) + ' done')

def main():
    if len(sys.argv) == 2:
        run(sys.argv[1])
        return
    camps = [1458, 2259, 3358, 2997, 2261, 3386, 3427, 2821, 3476]
    processNum = 9
    p = Pool(processNum)
    for c in camps:
        p.apply_async(run, [c])
    p.close()
    p.join()

if __name__ == '__main__':
    main()