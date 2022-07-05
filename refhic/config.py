import click
import os
from appdirs import user_config_dir,user_data_dir
from importlib_resources import files
import requests, zipfile, io
import configparser,sys,glob
from tqdm import tqdm
import pandas as pd

def checkConfig():
    '''check config'''
    localConfig = user_config_dir('refhic') + '/config.ini'
    if os.path.exists(localConfig):
        return True
    return False

def loadConfig():
    '''read config file'''
    localConfig = user_config_dir('refhic') + '/config.ini'
    config = configparser.ConfigParser()
    config.read(localConfig)
    return config

def referenceMeta(reference=None):
    '''read reference meta data from user provided file or config.ini'''
    if reference is None:
        config = loadConfig()
        meta = pd.read_csv(os.path.join(config['reference']['uri'],config['reference']['metafile']))
        for i in range(len(meta)):
            meta['file'].iloc[i]=os.path.join(config['reference']['uri'],meta['file'].iloc[i])
    else:
        meta = pd.read_csv(reference)
    return meta




@click.command()
def init():
    '''config initialization
    '''

    print('init command will setup reference panel and load trained models into user space, you only need to do it once at the begining in most case. ')
    answer = input("Do you want to continue? Yes/[N]o ").lower().strip()
    if answer == "yes" or answer == "y":
        pass
    else:
        print('Good Bye!')
        sys.exit()

    if os.path.isdir(user_config_dir('refhic')):
        pass
    else:
        os.makedirs(user_config_dir('refhic'))
    if os.path.isdir(user_data_dir('refhic')):
        pass
    else:
        os.makedirs(user_data_dir('refhic'))

    iniConfig=True

    localConfig=user_config_dir('refhic')+'/config.ini'
    if os.path.exists(localConfig):
        answer = input("You already configured refhic, do you want to re-config? Yes/[N]o ").lower().strip()
        if answer == "yes" or answer=="y":
            print("re-config ... let's go")
            oldConfig = configparser.ConfigParser()
            oldConfig.read(localConfig)
            _files = glob.glob(oldConfig['reference']['uri']+'/*')
            for _file in _files:
                os.remove(_file)
            pass
        else:
            iniConfig=False
            print('Good Bye!')
    if iniConfig:
        configfile = files('refhic').joinpath('config.ini')
        config = configparser.ConfigParser()
        config.read(configfile)
        print('What do you want to do?')
        default=input("1. download default reference panel; 2. load your local panel [1] ").strip()
        if default!='2':
            print('We are going to download the reference panel: assembly='+config['reference']['assembly']+', file size='+config['reference']['size']+'.')

        if default!='2':
            refpanelURL=config['reference']['url']
        else:
            localRefFile=input("please specific your local data path: ").strip()
            localRefFile=os.path.expanduser(localRefFile)
            if os.path.isabs(localRefFile):
                pass
            else:
                localRefFile=os.path.abspath(localRefFile)
        refpanelURI = input("Where do you want to save it? [" + user_data_dir('refhic') + '] ').strip()
        if refpanelURI is None or refpanelURI == '':
            refpanelURI = user_data_dir('refhic')
        else:
            refpanelURI = os.path.expanduser(refpanelURI)
            if os.path.isabs(refpanelURI):
                pass
            else:
                refpanelURI = os.path.abspath(refpanelURI)

            config['reference']['url'] = 'load from local file '+localRefFile
            config['reference']['size']=str(os.stat(localRefFile).st_size/(1024*1024*1024))+' GB'
            config['reference']['assembly'] = input("What is the assembly [hg38]: ").strip()
            if config['reference']['assembly'] is None or config['reference']['assembly']=='':
                config['reference']['assembly']='hg38'


        if default !='2':
            refFilename = os.path.basename(refpanelURL)
            chunk_size = 1024
            filesize = int(requests.head(refpanelURL).headers["Content-Length"])

            localRefFile=os.path.join(refpanelURI, refFilename)

            if os.path.exists(localRefFile) and os.path.getsize(localRefFile)==filesize:
                pass
            else:
                print('downloading file from ' + refpanelURL)
                with requests.get(refpanelURL, stream=True) as r, open(localRefFile, "wb") as f, tqdm(
                        unit="B",  # unit string to be displayed.
                        unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
                        unit_divisor=1024,  # is used when unit_scale is true
                        total=filesize,  # the total iteration.
                        file=sys.stdout,  # default goes to stderr, this is the display on console.
                        desc=refFilename  # prefix to be displayed on progress bar.
                ) as progress:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        # download the file chunk by chunk
                        datasize = f.write(chunk)
                        # on each chunk update the progress bar.
                        progress.update(datasize)

        with zipfile.ZipFile(localRefFile, 'r') as zip_ref:
            zip_ref.extractall(refpanelURI)
        if default != '2':
            os.remove(localRefFile)
        print('reference panel saved to',refpanelURI)
        config['reference']['uri'] = refpanelURI
        config['tad']['model'] = str(files('refhic').joinpath('model/TAD.tar'))
        config['loop']['model'] = str(files('refhic').joinpath('model/loop.tar'))
        with open(localConfig, 'w') as configfile:
            config.write(configfile)
        print('TAD model:',config['tad']['model'])
        print('Loop model:',config['loop']['model'])
        print('Done!')

@click.command()
def edit():
    '''modify config file manually [for advanced user]'''
    if checkConfig():
        pass
    else:
        print('Please run refhic config first.')
        print('Good bye!')
        sys.exit()
    localConfig = user_config_dir('refhic') + '/config.ini'
    os.system("vim "+localConfig)

@click.group()
def config():
    '''config RefHiC
    '''

config.add_command(init)
config.add_command(edit)