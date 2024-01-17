import torch
import torch.fft as fft
import numpy as np
from scipy.signal import gaussian

def build_scf_pyr(im, ht=None, order=3, twidth=1):
    max_ht = int(np.floor(np.log2(np.min(im.shape))) - 2)

    if ht is None:
        ht = max_ht
    elif ht > max_ht:
        raise ValueError(f"Cannot build pyramid higher than {max_ht} levels.")

    if order < 0 or order > 15:
        print("Warning: ORDER must be an integer in the range [0,15]. Truncating.")
        order = max(0, min(order, 15))
    else:
        order = round(order)

    nbands = order + 1

    if twidth <= 0:
        print("Warning: TWIDTH must be positive. Setting to 1.")
        twidth = 1

    harmonics = np.arange(0, nbands // 2) * 2 + 1 if nbands % 2 == 0 else np.arange(0, (nbands - 1) // 2) * 2
    steer_mtx = steer2HarmMtx(harmonics, np.pi * np.arange(nbands) / nbands, 'even')

    dims = im.shape
    ctr = (np.ceil((dims[0] + 0.5) / 2)).astype(int),(np.ceil((dims[1] + 0.5) / 2)).astype(int)


    xramp, yramp = np.meshgrid((np.arange(1, dims[1] + 1) - ctr[1]) / (dims[1] / 2),
                                (np.arange(1, dims[0] + 1) - ctr[0]) / (dims[0] / 2))
    angle = np.arctan2(yramp, xramp)
    log_rad = np.sqrt(xramp**2 + yramp**2)
    log_rad[ctr[0] - 1, ctr[1] - 1] = log_rad[ctr[0] - 1, ctr[1] - 2]
    log_rad = np.log2(log_rad)

    # Radial transition function (a raised cosine in log-frequency)
    Xrcos, Yrcos = rcosFn(twidth, -twidth / 2, [0, 1])
    Yrcos = np.sqrt(Yrcos)
    YIrcos = np.sqrt(1.0 - Yrcos**2)
    lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1] - Xrcos[0], 0)
    imdft = fft.fftshift(fft.fft2(torch.tensor(im).float()))
    lo0dft = imdft * lo0mask

    pyr, pind = build_scf_pyr_levels(lo0dft, log_rad, Xrcos, Yrcos, angle, ht, nbands)

    hi0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1] - Xrcos[0], 0)
    hi0dft = imdft * hi0mask
    hi0 = fft.ifft2(fft.ifftshift(hi0dft)).numpy()

    pyr = np.concatenate([np.real(hi0.flatten()), pyr])
    pind = np.concatenate([np.array(hi0.shape).reshape(1, -1), pind])

    return pyr, pind, steer_mtx, harmonics


def build_scf_pyr_levels(lo0dft, log_rad, Xrcos, Yrcos, angle, ht, nbands):
    pyr = []
    pind = []

    for i in range(ht):
        banddft, pind = buildSCFpyrLevs(lo0dft, log_rad, Xrcos, Yrcos, angle, i, nbands)
        pyr.append(banddft)
        if i == 0:
            pind = np.array(pind)
        else:
            pind = np.vstack([pind, np.array(pind)])

    return np.concatenate(pyr), pind


def rcosFn(twidth, position, extent):
    """Generate a raised cosine window."""
    X = np.arange(np.floor(position - extent / 2), np.floor(position + extent / 2) + 1)
    Y = np.cos((np.pi / twidth) * (X - position))**2 * (X >= position - extent / 2) * (X <= position + extent / 2)
    return X, Y


def steer2HarmMtx(harmonics, angles, symmetry):
    """Generate a steering matrix."""
    nbands = len(harmonics)
    nangles = len(angles)

    if symmetry == 'even':
        m = np.arange(1, nbands + 1).reshape(-1, 1)
        n = 2 * np.arange(1, nangles + 1) - 1
    elif symmetry == 'odd':
        m = np.arange(1, nbands + 1).reshape(-1, 1)
        n = 2 * np.arange(1, nangles + 1)
    else:
        raise ValueError("Invalid symmetry type. Use 'even' or 'odd'.")

    steer_mtx = np.cos(np.pi * m @ angles.reshape(1, -1) - harmonics.reshape(-1, 1) @ n.reshape(1, -1))

    return steer_mtx


def pointOp(log_rad, YIrcos, Xrcos0, Xrcos1_Xrcos0, angle):
    Xrcos = Xrcos0 + Xrcos1_Xrcos0 * np.cos(angle)
    Yrcos = YIrcos * np.sin(angle)
    Yrcos = np.maximum(Yrcos, 0)

    log_rad = np.array(log_rad)
    Xrcos = np.array(Xrcos)

    Xrcos = np.minimum(Xrcos, np.max(log_rad))
    Xrcos = np.maximum(Xrcos, np.min(log_rad))
    Yrcos = np.minimum(Yrcos, 1)

    return np.interp(log_rad, Xrcos, Yrcos)

def buildSCFpyrLevs(lo0dft, log_rad, Xrcos, Yrcos, angle, ht, nbands):
    pyr = []

    for level in range(ht):
        log_rad_lowpass = log_rad - level
        Xrcos_lowpass = Xrcos - level
        Yrcos_lowpass = Yrcos - level

        lo0mask = pointOp(log_rad_lowpass, Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0], 0)
        hi0mask = pointOp(log_rad_lowpass, Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0], np.pi/2)

        imdft = fft.fftshift(fft.fft2(lo0dft))
        lo0dft = imdft * lo0mask

        hi0dft = imdft * hi0mask
        hi0 = fft.ifft2(fft.ifftshift(hi0dft)).numpy()

        pyr.append(hi0)

    return torch.cat(pyr), []

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

# Example usage:
# im = ...  # Your input image
# pyr, pind, steermtx, harmonics = build_scf_pyr(im)

def pyrBand(pyr, pind, band):
    indices = pyrBandIndices(pind, band)
    res = pyr[indices[0]:indices[1]].reshape(pind[band - 1, 0], pind[band - 1, 1])
    return res

def pyrBandIndices(pind, band):
    if band > pind.shape[0] or band < 1:
        raise ValueError(f"BAND_NUM must be between 1 and the number of pyramid bands ({pind.shape[0]}).")

    if pind.shape[1] != 2:
        raise ValueError("INDICES must be an Nx2 matrix indicating the size of the pyramid subbands")

    ind = 0
    for l in range(band - 1):
        ind += np.prod(pind[l, :])

    indices = np.arange(ind, ind + np.prod(pind[band - 1, :]))
    return indices


def subMtx(vec, sz, offset=1):
    vec = vec.flatten()
    sz = sz.flatten()

    if sz.size != 2:
        raise ValueError("DIMENSIONS must be a 2-vector.")

    mtx = vec[offset - 1:offset + np.prod(sz) - 1].reshape(sz[0], sz[1])
    return mtx

import matplotlib.pyplot as plt
def test_complex_steerable_pyr():
    # Generate a random image
    np.random.seed(42)
    image_size = 256
    random_image = np.random.rand(image_size, image_size)

    # Apply complex steerable pyramid
    pyr, pind, steermtx, harmonics = build_scf_pyr(random_image)

    # Reconstruct the image
    reconstructed_image = reconSCFpyr(pyr, pind, steermtx)

    # Display the original and reconstructed images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(random_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(np.real(reconstructed_image), cmap='gray')
    plt.title('Reconstructed Image')

    plt.show()

if __name__ == "__main__":
    test_complex_steerable_pyr()
