import optparse
import os


def parse_range(astr):
    result=set()
    for part in astr.split(','):
        x=part.split('-')
        result.update(range(int(x[0]),int(x[-1])+1))
    return sorted(result)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-n', dest='n')
    opt,args=parser.parse_args()
    opt.n=parse_range(opt.n)

    for i in opt.n:
        os.system('qdel %d' % i)
