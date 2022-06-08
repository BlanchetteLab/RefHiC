import cooler
import sys
c = cooler.Cooler(sys.argv[1]+'::/resolutions/5000')
m = c.matrix(balance=True)

f = open(sys.argv[2])
for line in f:
    x=line.strip().split()
    frag1=x[0]+':'+x[1]+'-'+x[2]
    frag2=x[3]+':'+x[4]+'-'+x[5]
    val=float(m.fetch(frag1,frag2))
    print(line.strip()+'\t'+str(val))
