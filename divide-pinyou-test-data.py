camps = ['1458n', '2259n', '3358n', '2997n', '2261n', '3386n', '3427n', '2821n', '3476n']

for c in camps:
    print(c)
    fi = open(str(c) + '/test.yzx.txt.lr.pred', 'r')
    data = []
    for line in fi:
        data.append(line)
    fi.close()
    fo = open(str(c) + '/test.yzx.txt.lr.pred.half1', 'w')
    for i in range(len(data) / 2):
        fo.write(data[i])
    fo.close()
    fo = open(str(c) + '/test.yzx.txt.lr.pred.half2', 'w')
    for i in range(len(data) / 2, len(data), 1):
        fo.write(data[i])
    fo.close()

    fi = open(str(c) + '/test.yzx.txt', 'r')
    data = []
    for line in fi:
        data.append(line)
    fi.close()
    fo = open(str(c) + '/test.yzx.txt.half1', 'w')
    for i in range(len(data) / 2):
        fo.write(data[i])
    fo.close()
    fo = open(str(c) + '/test.yzx.txt.half2', 'w')
    for i in range(len(data) / 2, len(data), 1):
        fo.write(data[i])
    fo.close()