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
