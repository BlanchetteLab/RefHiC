from cooler.tools import split, partition
from operator import add
import argparse
import cooler
import numpy as np

def fetch_data(chunk):
    return np.copy(chunk["pixels"]["count"])

def zero_diags(chunk, data):
    pixels = chunk["pixels"]
    mask = np.abs(pixels["bin1_id"] - pixels["bin2_id"]) < 1
    data[mask] = 0
    return data

def marginalize(chunk, data):
    n = len(chunk["bins"]["chrom"])
    pixels = chunk["pixels"]
    marg = np.bincount(pixels["bin1_id"], weights=data, minlength=n) + np.bincount(
        pixels["bin2_id"], weights=data, minlength=n
    )
    return marg

def zero_trans(chunk, data):
    chrom_ids = chunk["bins"]["chrom"]
    pixels = chunk["pixels"]
    mask = chrom_ids[pixels["bin1_id"]] != chrom_ids[pixels["bin2_id"]]
    data[mask] = 0
    return data

def get_marg(clr):
    n_bins = clr.info["nbins"]
    base_filters = [zero_trans,zero_diags]
    marg = (
        split(clr, map=map)  # noqa
        .prepare(fetch_data)
        .pipe(base_filters)
        .pipe(marginalize)
        .reduce(add, np.zeros(n_bins))
    )
    return marg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",type=str,help='mcool file')
    parser.add_argument('--resol',type=int,help='resolution',default=5000)
    args = parser.parse_args()
    cfile = args.i+'::/resolutions/'+str(args.resol)
    clr = cooler.Cooler(cfile)
    
    marg=get_marg(clr)
    
    store_name = 'cis_margs'

    stats = {
    "ignore_diags": 0
    }

    with clr.open("r+") as grp:
        if store_name in grp["bins"]:
            del grp["bins"][store_name]
        h5opts = dict(compression="gzip", compression_opts=6)
        grp["bins"].create_dataset(store_name, data=marg, **h5opts)
        grp["bins"][store_name].attrs.update(stats)