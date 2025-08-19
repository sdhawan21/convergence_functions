import numpy as np

# ---- Option A: starting from psi (lensing potential) ----
def deflection_from_psi(psi, dx):
    """
    Compute deflection angles alpha = (alpha_x, alpha_y)
    from lensing potential psi on a 2D grid.

    psi : 2D numpy array
        Lensing potential values.
    dx : float
        Pixel size (same in x and y), in angular units (e.g. arcsec).
    """
    # central finite differences
    grad_y, grad_x = np.gradient(psi, dx)  # numpy's gradient: axis order (y,x)
    return grad_x, grad_y


# ---- Option B: starting from kappa (convergence) ----
def deflection_from_kappa(kappa, dx):
    """
    Compute deflection angles alpha from convergence kappa via FFT.

    kappa : 2D numpy array
        Convergence field.
    dx : float
        Pixel size (same in x and y), in angular units.
    """
    ny, nx = kappa.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(ny, d=dx) * 2*np.pi
    kx, ky = np.meshgrid(kx, ky)
    k2 = kx**2 + ky**2

    kappa_ft = np.fft.fft2(kappa)

    # Avoid division by zero at k=0
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha_x_ft = -2j * kx / k2 * kappa_ft
        alpha_y_ft = -2j * ky / k2 * kappa_ft
        alpha_x_ft[k2 == 0] = 0
        alpha_y_ft[k2 == 0] = 0

    alpha_x = np.fft.ifft2(alpha_x_ft).real
    alpha_y = np.fft.ifft2(alpha_y_ft).real
    return alpha_x, alpha_y


# ---- Option B (enhanced): starting from kappa with boundary control ----
def deflection_from_kappa(kappa, dx, *, mode="periodic", pad_factor=2, taper_frac=0.05):
    """
    Compute deflection angles alpha from convergence kappa.

    Parameters
    ----------
    kappa : 2D array
        Convergence field.
    dx : float
        Pixel size (same in x and y), in angular units.
    mode : {"periodic", "isolated"}
        - "periodic": fast FFT-based Poisson inversion assuming periodic BCs.
        - "isolated": approximate isolated BCs via zero-padding + FFT convolution
          with the Green-function kernel.
    pad_factor : int
        Factor by which to enlarge the grid (isolated mode only).
    taper_frac : float
        Fractional cosine taper applied at the borders to reduce ringing.
    """
    import numpy as _np
    ny, nx = kappa.shape

    # Optional cosine taper to reduce edge artefacts
    if taper_frac and taper_frac > 0:
        tx = int(max(1, round(taper_frac * nx)))
        ty = int(max(1, round(taper_frac * ny)))
        wx = _np.ones(nx)
        wy = _np.ones(ny)
        rampx = 0.5 * (1 - _np.cos(_np.linspace(0, _np.pi, tx)))
        rampy = 0.5 * (1 - _np.cos(_np.linspace(0, _np.pi, ty)))
        wx[:tx] = rampx
        wx[-tx:] = rampx[::-1]
        wy[:ty] = rampy
        wy[-ty:] = rampy[::-1]
        window = _np.outer(wy, wx)
        kappa = kappa * window

    if mode == "periodic":
        kx = _np.fft.fftfreq(nx, d=dx) * 2*_np.pi
        ky = _np.fft.fftfreq(ny, d=dx) * 2*_np.pi
        kx, ky = _np.meshgrid(kx, ky)
        k2 = kx**2 + ky**2
        kappa_ft = _np.fft.fft2(kappa)
        with _np.errstate(divide='ignore', invalid='ignore'):
            alpha_x_ft = -2j * kx / k2 * kappa_ft
            alpha_y_ft = -2j * ky / k2 * kappa_ft
            alpha_x_ft[k2 == 0] = 0  # mass-sheet degenerate k=0 mode
            alpha_y_ft[k2 == 0] = 0
        alpha_x = _np.fft.ifft2(alpha_x_ft).real
        alpha_y = _np.fft.ifft2(alpha_y_ft).real
        return alpha_x, alpha_y

    elif mode == "isolated":
        # Zero-pad and convolve with the analytic kernel via FFT
        py = pad_factor * ny
        px = pad_factor * nx
        K = _np.zeros((py, px))
        K[:ny, :nx] = kappa

        y = (_np.arange(py) - py//2) * dx
        x = (_np.arange(px) - px//2) * dx
        X, Y = _np.meshgrid(x, y)
        R2 = X**2 + Y**2
        eps = (dx*0.5)**2  # small core to avoid the r=0 singularity
        kernel_x = (1/_np.pi) * X / (R2 + eps)
        kernel_y = (1/_np.pi) * Y / (R2 + eps)
        kernel_x[py//2, px//2] = 0.0
        kernel_y[py//2, px//2] = 0.0

        # Shift so that the kernel center is at index (0,0) before FFT
        kernel_x = _np.fft.ifftshift(kernel_x)
        kernel_y = _np.fft.ifftshift(kernel_y)

        K_ft = _np.fft.fft2(K)
        ax = _np.fft.ifft2(K_ft * _np.fft.fft2(kernel_x)).real
        ay = _np.fft.ifft2(K_ft * _np.fft.fft2(kernel_y)).real

        # Crop back to original field of view (center)
        sy = (py - ny)//2
        sx = (px - nx)//2
        ax = ax[sy:sy+ny, sx:sx+nx]
        ay = ay[sy:sy+ny, sx:sx+nx]
        return ax, ay

    else:
        raise ValueError("mode must be 'periodic' or 'isolated'")

# ---- Hessian (second derivatives of psi) ----

def hessian_from_kappa(kappa, dx, *, mode="periodic"):
    """
    Compute the Hessian components of the lensing potential from a kappa map.

    Returns psi_xx, psi_yy, psi_xy on the same grid.
    For periodic boundaries, this is an exact spectral inversion:
      	ilde{psi}_{ij}(k) = 2 [k_i k_j / k^2] 	ilde{kappa}(k),  (k != 0)
    The k=0 mode (mass-sheet) is set to zero.
    """
    import numpy as _np
    ny, nx = kappa.shape
    if mode != "periodic":
        raise NotImplementedError("Use periodic mode or reconstruct psi via deflection_from_kappa(..., mode='isolated') then finite-difference.")

    kx = _np.fft.fftfreq(nx, d=dx) * 2*_np.pi
    ky = _np.fft.fftfreq(ny, d=dx) * 2*_np.pi
    kx, ky = _np.meshgrid(kx, ky)
    k2 = kx**2 + ky**2
    K = _np.fft.fft2(kappa)

    with _np.errstate(divide='ignore', invalid='ignore'):
        Pxx = 2.0 * (kx*kx / k2) * K
        Pyy = 2.0 * (ky*ky / k2) * K
        Pxy = 2.0 * (kx*ky / k2) * K
        Pxx[k2 == 0] = 0
        Pyy[k2 == 0] = 0
        Pxy[k2 == 0] = 0

    psi_xx = _np.fft.ifft2(Pxx).real
    psi_yy = _np.fft.ifft2(Pyy).real
    psi_xy = _np.fft.ifft2(Pxy).real
    return psi_xx, psi_yy, psi_xy


def hessian_from_psi(psi, dx):
    """
    Finite-difference Hessian from a psi map (no FFT).
    Returns psi_xx, psi_yy, psi_xy using centered differences.
    """
    import numpy as _np
    dpsi_dy, dpsi_dx = _np.gradient(psi, dx)
    psi_xx = _np.gradient(dpsi_dx, dx, axis=1)
    psi_yy = _np.gradient(dpsi_dy, dx, axis=0)
    # Symmetrized cross derivative
    psi_xy = 0.5*(_np.gradient(dpsi_dx, dx, axis=0) + _np.gradient(dpsi_dy, dx, axis=1))
    return psi_xx, psi_yy, psi_xy

# ---- Example usage ----
if __name__ == "__main__":
    # Example: Gaussian kappa field
    nx = ny = 128
    dx = 0.2  # arcsec per pixel
    x = (np.arange(nx) - nx//2) * dx
    y = (np.arange(ny) - ny//2) * dx
    X, Y = np.meshgrid(x, y)
    sigma = 3.0
    kappa = np.exp(-(X**2+Y**2)/(2*sigma**2))

    alpha_x, alpha_y = deflection_from_kappa(kappa, dx)
    print(alpha_x.shape, alpha_y.shape)
