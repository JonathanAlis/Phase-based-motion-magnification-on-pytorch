import torch
import torch.fft as fft
import numpy as np

def reconSCFpyr(pyr, indices, levs='all', bands='all', twidth=1):
    if levs == 'all':
        levs = None
    if bands == 'all':
        bands = None

    if twidth <= 0:
        print("Warning: TWIDTH must be positive. Setting to 1.")
        twidth = 1

    pind = indices
    Nsc = int(np.log2(pind[0, 0] / pind[-1, 0]))
    Nor = int((pind.shape[0] - 2) / Nsc)

    for nsc in range(1, Nsc + 1):
        firstBnum = (nsc - 1) * Nor + 2

        # Re-create analytic subbands
        dims = pind[firstBnum - 1, :]
        ctr = np.ceil((dims + 0.5) / 2).astype(int)
        ang = mkAngle(dims, 0, ctr)
        ang[ctr[0] - 1, ctr[1] - 1] = -np.pi / 2

        for nor in range(1, Nor + 1):
            nband = (nsc - 1) * Nor + nor
            ind = pyrBandIndices(pind, nband)
            ch = pyrBand(pyr, pind, nband)
            ang0 = np.pi * (nor - 1) / Nor
            xang = np.mod(ang - ang0 + np.pi, 2 * np.pi) - np.pi
            amask = 2 * (np.abs(xang) < np.pi / 2) + (np.abs(xang) == np.pi / 2)
            amask[ctr[0] - 1, ctr[1] - 1] = 1
            amask[:, 0] = 1
            amask[0, :] = 1
            amask = np.fft.fftshift(amask)
            ch = torch.ifftn(amask * torch.fftn(ch))  # "Analytic" version
            f = 1
            ch = f * 0.5 * torch.real(ch)  # real part
            pyr[ind] = ch

    return reconSFpyr(pyr, indices, levs, bands, twidth)

# You would need to implement the missing helper functions `mkAngle`, `pyrBandIndices`, and `pyrBand`.
# Also, make sure to include them in your code.

import torch
import torch.fft as fft
import numpy as np

def reconSFpyr(pyr, pind, levs='all', bands='all', twidth=1):
    # Defaults
    if levs == 'all':
        levs = list(range(spyrHt(pind) + 1))
    else:
        levs = list(map(int, levs))

    if bands == 'all':
        bands = list(range(1, spyrNumBands(pind) + 1))
    else:
        bands = list(map(int, bands))

    if twidth <= 0:
        print("Warning: TWIDTH must be positive. Setting to 1.")
        twidth = 1

    # Dimensions and center
    dims = pind[0]
    ctr = torch.ceil((dims + 0.5) / 2).int()

    xramp, yramp = torch.meshgrid([torch.arange(1, dims[1] + 1) - ctr[1]] / (dims[1] / 2),
                                  [torch.arange(1, dims[0] + 1) - ctr[0]] / (dims[0] / 2))
    angle = torch.atan2(yramp, xramp)
    log_rad = torch.sqrt(xramp**2 + yramp**2)
    log_rad[ctr[0] - 1, ctr[1] - 1] = log_rad[ctr[0] - 1, ctr[1] - 2]
    log_rad = torch.log2(log_rad)

    # Radial transition function (a raised cosine in log-frequency)
    Xrcos, Yrcos = rcosFn(twidth, -twidth/2, [0, 1])
    Yrcos = torch.sqrt(Yrcos)
    YIrcos = torch.sqrt(torch.abs(1.0 - Yrcos**2))

    if pind.size(0) == 2:
        if 1 in levs:
            resdft = fft.fftshift(fft.fft2(pyrBand(pyr, pind, 2)))
        else:
            resdft = torch.zeros(pind[1, :])
    else:
        resdft = reconSFpyrLevs(pyr[pind[1, 0]:], pind[1:, :], log_rad, Xrcos, Yrcos, angle, spyrNumBands(pind), levs, bands)

    lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1] - Xrcos[0], 0)
    resdft = resdft * lo0mask

    # Residual highpass subband
    if 0 in levs:
        hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0], 0)
        hidft = fft.fftshift(fft.fft2(subMtx(pyr, pind[0, :])))
        resdft = resdft + hidft * hi0mask

    res = torch.real(fft.ifft2(fft.ifftshift(resdft)))

    return res

# You would need to implement the missing helper functions `spyrHt`, `spyrNumBands`, `rcosFn`, `pyrBand`, `reconSFpyrLevs`, `pointOp`, and `subMtx`.
# Also, make sure to include them in your code.
