import cooler
import numpy as np
from types import SimpleNamespace


class bandmatrix():
    def __init__(self, pixels, extent, max_distance_bins=None, bins=None, info=None):
        self.extent = extent
        self.max_distance_bins = max_distance_bins
        self.bmatrix = np.zeros((extent[1] - extent[0], max_distance_bins))
        self.offset = extent[0]
        self.bmatrix[pixels['bin1_id'] - self.offset, (pixels['bin2_id'] - pixels['bin1_id']).abs()] = pixels[
            'balanced']
        self.diag_mean = np.nanmean(self.bmatrix, axis=0)
        np.nan_to_num(self.bmatrix, copy=False)

        self.bins = bins
        self.bp2bin = \
            bins['start'].reset_index(drop=False).rename(columns={"start": "bp", "index": "bin"}).set_index(
                'bp').to_dict()[
                'bin']
        self.resol = self.bins.iloc[0]['end'] - self.bins.iloc[0]['start']
        self.info = info
        self.bin2bias = np.zeros(self.extent[1] - self.extent[0] + 1)
        if 'full_sum' in self.info:
            self.totalRC = self.info['full_sum']
        elif 'sum' in info:
            self.totalRC = self.info['sum']
        else:
            self.totalRC = None
        self.bin2bias = np.zeros((extent[1] - extent[0]))
        for k, v in bins.to_dict()['weight'].items():
            self.bin2bias[k - self.offset] = v
        self.bin2bias = np.nan_to_num(self.bin2bias)

        self.continousRows = {'start_bp': np.inf, 'end_bp': -1, 'O_matrix': None, 'OE_matrix': None, 'bias': None,
                              'offset_bin': 0}
        self.continousRows = SimpleNamespace(**self.continousRows)

    def __bandedRows2fullRows(self, x):
        """
        coverting rows in bandedMatrix to upper triangle (+ necessary lower triangle) fullMatrix
        x????        x???x000
        x@@xx        ?x@@xx00
        x#xxx   -->  ?@x#xxx0
        xxxxx        ?@#xxxxx
        """
        b, h, w = x.shape
        output = np.zeros((b, h, h + w))
        output[:b, :h, :w] = x
        output = output.reshape(b, -1)[:, :-h].reshape(b, h, -1)[:, :, :h + w]
        i_lower = np.tril_indices(h, -1)
        for i in range(b):
            output[i][i_lower] = output[i].swapaxes(-1, -2)[i_lower]
        return output

    def __relative_right_shift(self, x):
        """
        .........xxxxxx      xxxxxx0000000000
        ........xxxxxx.      xxxxxx.000000000
        .......xxxxxx.. ---> xxxxxx..00000000
        ......xxxxxx...      xxxxxx...0000000
        .....xxxxxx....      xxxxxx....000000
        """
        b, h, w = x.shape
        output = np.zeros((b, h, 2 * w))
        output[:b, :h, :w] = x
        return output.reshape(b, -1)[:, :-h].reshape(b, h, -1)[:, :, h - 1:]

    def __tril_block(self, top, left, bottom, right, type='o'):
        """
        fetch data in lower triangular part without main diagonal
        Parameters:
        top,left,bottom,right : block coords. left/right < 0
        type                  : o [observe], oe [o/e], b [both]
        """

        if left >= 0 or right >= 0:
            raise Exception("Trying to access data outside lower triangular part with tril_block")

        height = bottom - top
        top, bottom = top + left, bottom + right
        left, right = -right, -left

        if top < 0 or bottom > self.bmatrix.shape[0] - 1:
            raise Exception("Accessing values outside the contact map ... valid region:" +
                            str(10 * self.resol) + '~' + str((self.extent[1] - self.extent[0] - 10) * self.resol))

        O = self.bmatrix[top:bottom + 1, left:right + 1]

        if type == 'o':
            out = self.__relative_right_shift(O[None].swapaxes(-1, 1)).swapaxes(-1, 1)[:, :height + 1, :]
        elif type == 'oe':
            OE = O / self.diag_mean[left:right + 1]
            out = self.__relative_right_shift(OE[None].swapaxes(-1, 1)).swapaxes(-1, 1)[:, :height + 1, :]
        else:
            OE = O / self.diag_mean[left:right + 1]
            out = np.concatenate((O[None], OE[None]))
            out = self.__relative_right_shift(out.swapaxes(-1, 1)).swapaxes(-1, 1)[:, :height + 1, :]

        return out[..., ::-1]

    def rows(self, firstRow, lastRow, type='o', returnBias=False):
        """
        fetch rows [firstRow,lastRow] of contacts
        Parameters
        ----------
        firstRow   : inclusive first row in bp
        lastRow    : inclusive last  row in bp
        type       : o [observe], oe [o/e], b [both]
        returnBias : If true, return bias in an array for bins [first row,last row + max_distance_bins)
        """
        firstRow = firstRow // self.resol * self.resol
        lastRow = lastRow // self.resol * self.resol
        ORows = None
        OERows = None
        if firstRow < 0 or lastRow < 0 or firstRow > (self.extent[1] - self.extent[0]) * self.resol or lastRow > (
                self.extent[1] - self.extent[0]) * self.resol:
            raise Exception("Accessing values outside the contact map ... valid region: 0 ~ "
                            + str((self.extent[1] - self.extent[0]) * self.resol))

        firstRowRelativeBin = self.bp2bin[firstRow] - self.offset
        lastRowRelativeBin = self.bp2bin[lastRow] - self.offset
        ORows = self.bmatrix[firstRowRelativeBin:lastRowRelativeBin + 1, :][None]

        if type == 'o':
            outRows = ORows
        elif type == 'oe':
            OERows = (ORows / self.diag_mean)
            outRows = OERows
        elif type == 'b':
            OERows = (ORows / self.diag_mean)
            outRows = np.concatenate((ORows, OERows), axis=0)

        outRows = self.__bandedRows2fullRows(outRows)

        if returnBias:
            bias = self.bin2bias[firstRowRelativeBin:lastRowRelativeBin + self.max_distance_bins]
            # print('bias.shape',bias.shape)
            # p2ll = self.p2ll(output[-1,:,:],cw=3) # prefer to use obs to compuate p2ll
            return outRows, bias

        return outRows

    def __squareFromContinousRows(self, xCenter, yCenter, w, type='o', meta=True):
        """
        fetch a (2w+1)*(2w+1) square of contacts centered at (xCenter,yCenter) from continousrows efficiently
        Parameters
        ----------
        xCenter : xCenter in bp
        yCenter : yCenter in bp
        w       : block width = 2w+1, in bins
        type    : o [observe], oe [o/e], b [both]
        """

        if xCenter < self.continousRows.start_bp or xCenter > self.continousRows.end_bp:
            print('miss')
            rowStep = 1000
            startRow_bp = np.max([0, xCenter // (rowStep * self.resol) * (rowStep - 2 * w) * self.resol])
            endRow_bp = np.min(
                [startRow_bp + (rowStep + 2 * w) * self.resol, (self.extent[1] - self.offset - 1) * self.resol])
            mat, bias = self.rows(startRow_bp, endRow_bp, type='b', returnBias=True)

            self.continousRows.start_bp = startRow_bp
            self.continousRows.end_bp = endRow_bp
            self.continousRows.O_matrix = mat[0, :, :]
            self.continousRows.OE_matrix = mat[1, :, :]
            self.continousRows.bias = bias
        else:
            print('hit')

        xCenterRelativeBin = (xCenter - self.continousRows.start_bp) // self.resol
        yCenterRelativeBin = (yCenter - self.continousRows.start_bp) // self.resol

        # = {'start_bp': v, 'end_bp': v, 'O_matrix': None, 'OE_matrix': None, 'bias':None, 'offset_bin': 0}
        if type == 'o':
            output = self.continousRows.O_matrix[xCenterRelativeBin - w:xCenterRelativeBin + w + 1,
                     yCenterRelativeBin - w:yCenterRelativeBin + w + 1][None]
        elif type == 'oe':
            output = self.continousRows.OE_matrix[xCenterRelativeBin - w:xCenterRelativeBin + w + 1,
                     yCenterRelativeBin - w:yCenterRelativeBin + w + 1][None]
        else:
            OEsquare = self.continousRows.OE_matrix[xCenterRelativeBin - w:xCenterRelativeBin + w + 1,
                       yCenterRelativeBin - w:yCenterRelativeBin + w + 1][None]
            Osquare = self.continousRows.O_matrix[xCenterRelativeBin - w:xCenterRelativeBin + w + 1,
                      yCenterRelativeBin - w:yCenterRelativeBin + w + 1][None]
            output = np.concatenate((Osquare, OEsquare))

        if meta:
            xBias = self.continousRows.bias[xCenterRelativeBin - w:xCenterRelativeBin + w + 1]
            yBias = self.continousRows.bias[yCenterRelativeBin - w:yCenterRelativeBin + w + 1]
            bias = np.concatenate((xBias, yBias))
            p2ll,crk = self.p2ll(output[-1, :, :], cw=3)  # prefer to use obs to compuate p2ll
            return output, np.concatenate((bias, [self.totalRC, p2ll,yCenterRelativeBin,crk]))
        return output

    def p2ll(self, x, cw=3):
        """
        P2LL for a peak.
        Parameters:
        x      : sqaure matrix, peak and its surrandings
        cw     : lower-left corner width
        """
        c = x.shape[0] // 2
        llcorner = x[-cw:, :cw].flatten()
        if sum(llcorner) == 0:
            return 0,np.sum(x[c,c]>x[c-1:c+2,c-1:c+2])
        return x[c, c] / (sum(llcorner) / len(llcorner)),np.sum(x[c,c]>x[c-1:c+2,c-1:c+2])

    def square(self, xCenter, yCenter, w, type='o', meta=True, cache=False):
        """
        fetch a (2w+1)*(2w+1) square of contacts centered at (xCenter,yCenter)
        Parameters
        ----------
        xCenter : xCenter in bp
        yCenter : yCenter in bp
        w       : block width = 2w+1, in bins
        type    : o [observe], oe [o/e], b [both]
        """
        # print(xCenter,yCenter)
        tril = None
        xCenter = xCenter // self.resol * self.resol
        yCenter = yCenter // self.resol * self.resol
        # if xCenter > yCenter:
        #     tmp = xCenter
        #     xCenter = yCenter
        #     yCenter = tmp

        # if xCenter - w * self.resol < 0 or yCenter - w * self.resol < 0 or \
        #         xCenter + w * self.resol > (
        #         self.extent[1] - self.extent[0] - 1) * self.resol or yCenter + w * self.resol > (
        #         self.extent[1] - self.extent[0] - 1) * self.resol:
        #     raise Exception("Accessing values outside the contact map ... valid region: 0 ~ "
        #                     + str((self.extent[1] - self.extent[0]) * self.resol))

        # if cache:
        #     # print("cache")
        #     return self.__squareFromContinousRows(xCenter, yCenter, w, type, meta)

        xCenterRelativeBin = self.bp2bin[xCenter] - self.offset
        yCenterRelativeBin = self.bp2bin[yCenter] - self.offset - xCenterRelativeBin

        # if yCenterRelativeBin + 2 * w >= self.max_distance_bins:
        #     raise Exception("max distance in this gcool file is ", self.max_distance_bins * self.resol)
        topleft = [xCenterRelativeBin - w, yCenterRelativeBin - 2 * w]
        bottomright = [xCenterRelativeBin + w, yCenterRelativeBin + 2 * w]

        if topleft[1] < 0:
            tril = (topleft[0], topleft[1], bottomright[0], -1)
            topleft[1] = 0
            tril_part = self.__tril_block(tril[0], tril[1], tril[2], tril[3], type)

        Osquare = self.bmatrix[topleft[0]:bottomright[0] + 1, topleft[1]:bottomright[1] + 1]

        if type == 'o':
            Osquare = Osquare[None]
            if tril is not None:
                Osquare = np.concatenate((tril_part, Osquare), axis=-1)
            output = self.__relative_right_shift(Osquare)[:, :, :2 * w + 1]
        elif type == 'oe':
            OEsquare = (Osquare / self.diag_mean[topleft[1]:bottomright[1] + 1])[None]
            if tril is not None:
                OEsquare = np.concatenate((tril_part, OEsquare), axis=-1)
            output = self.__relative_right_shift(OEsquare)[:, :, :2 * w + 1]
        else:
            OEsquare = Osquare / self.diag_mean[topleft[1]:bottomright[1] + 1]
            output = np.concatenate((Osquare[None], OEsquare[None]))
            if tril is not None:
                output = np.concatenate((tril_part, output), axis=-1)
            output = self.__relative_right_shift(output)[:, :, :2 * w + 1]
        if meta:
            xBias = self.bin2bias[self.bp2bin[xCenter] - self.offset - w:self.bp2bin[xCenter] - self.offset + w + 1]
            yBias = self.bin2bias[self.bp2bin[yCenter] - self.offset - w:self.bp2bin[yCenter] - self.offset + w + 1]
            bias = np.concatenate((xBias, yBias))

            p2ll,crk = self.p2ll(output[-1, :, :], cw=3)  # prefer to use obs to compuate p2ll
            return output, np.concatenate((bias, [self.totalRC, p2ll,yCenterRelativeBin,crk]))
        return output


class gcool(cooler.Cooler):
    def __init__(self, store):
        super().__init__(store)

    def bchr(self, chrom, balance=True, max_distance=None, annotate=True):
        '''
        get banded matrix for a given chrom
        '''
        resol = self.info['bin-size']
        if max_distance is not None and max_distance > self.info['max_distance']:
            raise Exception("max distance in this gcool file is ", self.info['max_distance'])
        else:
            max_distance = self.info['max_distance']
        pixels = self.matrix(balance=balance, as_pixels=True).fetch(chrom)
        if annotate:
            bins = self.bins().fetch(chrom)
            info = self.info
        else:
            bins = None
            info = None
        extent = self.extent(chrom)
        bmatrix = bandmatrix(pixels, extent, max_distance // resol, bins, info)
        return bmatrix
