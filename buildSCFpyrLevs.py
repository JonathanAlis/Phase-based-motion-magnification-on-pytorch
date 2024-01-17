import torch
import torch.fft as fft
import numpy as np

def pointOp(log_rad, YIrcos, Xrcos0, Xrcos1_Xrcos0, angle):
    Xrcos = Xrcos0 + Xrcos1_Xrcos0 * torch.cos(angle)
    Yrcos = YIrcos * torch.sin(angle)
    Yrcos = torch.max(Yrcos, torch.zeros_like(Yrcos))

    log_rad = log_rad.clone()
    Xrcos = Xrcos.clone()

    Xrcos = torch.min(Xrcos, torch.max(log_rad))
    Xrcos = torch.max(Xrcos, torch.min(log_rad))
    Yrcos = torch.min(Yrcos, torch.ones_like(Yrcos))

    return torch.nn.functional.grid_sample(Yrcos.unsqueeze(0).unsqueeze(0),
                                            torch.stack([log_rad, Xrcos], dim=-1).unsqueeze(0)).squeeze(0).squeeze(0)

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

# Example usage:
# pyr, pind = buildSCFpyrLevs(lo0dft, log_rad, Xrcos, Yrcos, angle, ht, nbands)
