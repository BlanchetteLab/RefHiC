import click
import cooler
import h5py
from cooler.create._create import write_pixels,write_indexes,index_bins,index_pixels,prepare_pixels,PIXEL_DTYPES,_set_h5opts,write_info
from cooler.util import get_meta
import posixpath

@click.command()
@click.option('-u', type=int, default=3000000, help='distance upperbund [bp] [default=3000000]')
@click.option('--resol',default=None,help='comma separated resols for output')
@click.argument('cool', type=str,required=True)
@click.argument('bcool', type=str,required=True)
def cool2bcool(cool, bcool,u,resol):
    '''covert a .mcool file to  [b]and cool file'''
    h5opts = _set_h5opts(None)
    copy = ['bins', 'chroms']
    Ofile = h5py.File(bcool, 'w')
    Ifile = h5py.File(cool, 'r')

    if resol is None:
        resols = [r.split('/')[-1] for r in cooler.fileops.list_coolers(cool)]
    else:
        resols = resol.split(',')
    # copy bins and chroms
    for grp in Ifile:
        Ofile.create_group(grp)
        for subgrp in Ifile[grp]:
            if subgrp in resols:
                Ofile[grp].create_group(subgrp)
                for ssubgrp in Ifile[grp][subgrp]:
                    if ssubgrp in copy:
                        Ofile.copy(Ifile[grp + '/' + subgrp + '/' + ssubgrp], grp + '/' + subgrp + '/' + ssubgrp)
    Ofile.flush()
    Ifile.close()

    for group_path in ['/resolutions/'+str(r) for r in resols]:
        c = cooler.Cooler(cool + '::' + group_path)
        nnz_src = c.info['nnz']
        n_bins = c.info['nbins']
        n_chroms = c.info['nchroms']
        bins = c.bins()[:]
        pixels = []
        info = c.info
        info['subformat'] = 'bcool'
        info['max_distance'] = u
        info['full_nnz'] = info['nnz']
        info['full_sum'] = info['sum']

        # collect pixels
        for lo, hi in cooler.util.partition(0, nnz_src, nnz_src // 100):
            pixel = c.pixels(join=False)[lo:hi].reset_index(drop=True)
            bins1 = bins.iloc[pixel['bin1_id']][['chrom', 'start']].reset_index(drop=True)
            bins2 = bins.iloc[pixel['bin2_id']][['chrom', 'start']].reset_index(drop=True)
            pixel = pixel[
                (bins1['chrom'] == bins2['chrom']) & ((bins1['start'] - bins2['start']).abs() < u)].reset_index(
                drop=True)
            pixels.append(pixel)

        columns = list(pixels[0].columns.values)
        meta = get_meta(columns, dict(PIXEL_DTYPES), default_dtype=float)

        # write pixels
        with h5py.File(bcool, "r+") as f:
            h5 = f[group_path]
            grp = h5.create_group("pixels")
            max_size = n_bins * (n_bins - 1) // 2 + n_bins
            prepare_pixels(grp, n_bins, max_size, meta.columns, dict(meta.dtypes), h5opts)

        target = posixpath.join(group_path, 'pixels')
        nnz, ncontacts = write_pixels(bcool, target, columns, pixels, h5opts, lock=None)
        info['nnz'] = nnz
        info['sum'] = ncontacts

        # write indexes
        with h5py.File(bcool, "r+") as f:
            h5 = f[group_path]
            grp = h5.create_group("indexes")
            chrom_offset = index_bins(h5["bins"], n_chroms, n_bins)
            bin1_offset = index_pixels(h5["pixels"], n_bins, nnz)
            write_indexes(grp, chrom_offset, bin1_offset, h5opts)
            write_info(h5, info)


if __name__ == '__main__':
    cool2bcool()