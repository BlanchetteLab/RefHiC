import click
from refhic.trainLoop import train as trainLoop
from refhic.loopPred import pred as loopPred
from refhic.tadPred import pred as tadPred
from refhic.trainTAD import  train as trainTAD
from refhic.config import config
from refhic.cool2bcool import cool2bcool
from refhic.poolLoop import pool
from refhic.poolLoop2 import pool2

@click.group()
def cli():
    pass

@cli.group()
def loop():
    '''loop annotation'''
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

loop.add_command(trainLoop)
loop.add_command(loopPred)
loop.add_command(pool)
loop.add_command(pool2)

tad.add_command(trainTAD)
tad.add_command(tadPred)

cli.add_command(config)



if __name__ == '__main__':
    cli()