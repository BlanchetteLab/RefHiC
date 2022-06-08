keep = ['chr1','chr2','chr3','chr4','chr5','chr6']
import sys
f = open(sys.argv[1])
for line in f:
    line = line.strip()
    if line.split()[0] in keep:
        print(line)