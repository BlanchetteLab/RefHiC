import click
from refhic.trainLoop import train as trainLoop
from refhic.loopPred import pred as loopPred
from refhic.tadPred import pred as tadPred
from refhic.trainTAD import  train as trainTAD
from refhic.config import config
from refhic.cool2bcool import cool2bcool
from refhic.poolLoop import pool
from refhic.util import  pileup
from refhic.traindata import traindata

@click.group()
def cli():
    '''RefHiC

    A reference panel guided topological structure annotation of Hi-C data
    '''
    pass

@cli.group()
def loop():
    '''loop annotation

    \b
    annotation with a trained model:
    1. run pred to detect loop candidates from Hi-C data
    2. run pool to select final loop annotations from loop candidates

    '''
    pass

@cli.group()
def tad():
    '''TAD boundary annotation'''
    pass

@cli.group()
def util():
    '''utilities'''
    pass

util.add_command(cool2bcool)
util.add_command(pileup)
util.add_command(traindata)


loop.add_command(trainLoop)
loop.add_command(loopPred)
loop.add_command(pool)

tad.add_command(trainTAD)
tad.add_command(tadPred)


cli.add_command(config)



if __name__ == '__main__':
    cli()