import numpy as np
import sys
sys.setrecursionlimit(100000)

SUB1_FILE = '/storage/submissions/sub7/submission.txt'
SUB2_FILE = '/storage/submissions/sub11/submission.txt'
SUB3_FILE = '/storage/submissions/phanbu/submission.txt'

fi1 = open(SUB1_FILE, 'r')
fi2 = open(SUB2_FILE, 'r')
fo = open(SUB3_FILE, 'w')

lines1 = fi1.read().splitlines()
lines2 = fi2.read().splitlines()
lines1 = [line.split() for line in lines1]
lines2 = [line.split() for line in lines2]
lines1 = lines1[:]
lines2 = lines2[:]

lines1.sort(key=lambda x: int(x[0][-2:]) * 20000 + int(x[1]))
lines2.sort(key=lambda x: int(x[0][-2:]) * 20000 + int(x[1]))

for index, line1 in enumerate(lines1):
    print(index)
    for line2 in lines2:
        if line2[-1] != '-100000000' and line1[1] == line2[1]:
            line1.append('-100000000')
            line2.append('-100000000')
            break

lines3 = []
# [lines3.append(x[:4]) for x in lines1 if x[-1] == '-100000000']
[lines3.append(x[:4]) for x in lines2 if x[-1] == '-100000000']
lines3.sort(key=lambda x: int(x[0][-2:]) * 20000 + int(x[1]))
[fo.write(' '.join(x)+'\n') for x in lines3]
fo.close()