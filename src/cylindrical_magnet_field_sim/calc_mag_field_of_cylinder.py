'''
Marcel van den Broek 
26-11-2026

implementation of the paper: 
Cylindrical Magnets and Ideal Solenoids, 2010, Norman Derby and Stanislaw Olbert.

This code models the magnetic field from a cylindrical ferromagnet. 
It was verified by the spec sheet of a commonly available magnet with specifications:
    Magnetic_remanence (Br) = 14800 Gauss = 1.48 Tesla 
    radius = 9.525 mm 
    height = 12.7 mm
    Surface_field = 5920 Gaus = 0.592 Telsa
Given the first three parameters we found 0.592 Tesla using equations 2 and 4 of the paper independantly.'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def calc_cel(kc, p, c, s, errtol=1e-6):
    '''this is the algorithm in the paper on the generalized complete elliptic integral.'''
    if kc == 0:
        return float("nan")

    k = abs(kc)
    pp = p
    cc = c
    ss = s
    em = 1.0    

    if p > 0:
        pp = np.sqrt(p)
        ss = s/pp
    else:
        f = kc * kc            
        q = 1.0 - f            # k² = 1 - kc²
        g = 1.0 - pp           # 1 - p   (> 0 here because p <= 0)
        f = f - pp             # kc² - p (> 0)
        q = q * (ss - c * pp)  # k² (s - c p)
        pp = np.sqrt(f / g)  # new sqrt(p) after the transformation
        cc = (c - ss) / g      # new c
        ss = -q / (g * g * pp) + cc * pp   # new s (already divided by sqrt(p))

    f = cc
    cc = cc + ss / pp
    g = k / pp
    ss = 2.0 * (ss + f * g)
    pp = g + pp
    g = em          # remember the previous arithmetic term
    em = k + em     # new arithmetic term (sum = 2 x arithmetic mean)
    kk = k          # kk holds the product used for the geometric mean

    while abs(g - k) > g * errtol:
        k = 2.0 * np.sqrt(kk)
        kk = k * em
        f = cc
        cc = cc + ss / pp
        g = kk / pp
        ss = 2.0 * (ss + f * g)
        pp = g + pp
        g = em
        em = k + em

    return (np.pi / 2.0) * (ss + cc * em) / (em * (em + pp))


def Br_to_B0(Br):
    '''from the paper eq.5: B0 = mu0*n*I/pi. 
    Normally B = mu0(H+M), but with Br the auxilary field H=0 and Br=mu0*M.
    In the paper the derivation relies on the likeness between a solenoid and the magnetization of a ferromagnet and n*I = M. This is also covered in griffith intro to EM which explains surface and volume currents in a material with a net magnetization.
    Thus Br = mu0*n*I. Plugging this into equation 5 yields:
    B0 = Br/pi
    '''
    return Br/np.pi


def field(rho, z, b, a, Br):
    '''This is are the equations 3 to 10 of the paper.
    rho: radius in cylindrical coordinates from the z-axis.
    z: the height in cylindrical coordinates
    b: the height of the cylinder from the origin (half of the total cylinder height)
    a: radius of the cylinder
    Br: magnetic remanence (magnetic field B without external field H, see Br_to_B0 function)
    '''

    B0 = Br_to_B0(Br)

    z_p, z_m = z+b, z-b

    alpha_p = a / np.sqrt(z_p*z_p + (rho+a)**2)
    alpha_m = a / np.sqrt(z_m*z_m + (rho+a)**2)

    beta_p = z_p / np.sqrt(z_p*z_p +(rho+a)**2)
    beta_m = z_m / np.sqrt(z_m*z_m +(rho+a)**2)

    gamma = (a - rho) / (a + rho)

    kc_p = np.sqrt((z_p*z_p + (a-rho)**2) / (z_p*z_p + (a+rho)**2))
    kc_m = np.sqrt((z_m*z_m + (a-rho)**2) / (z_m*z_m + (a+rho)**2))

    B_rho = B0 * (alpha_p * calc_cel(kc_p, 1.0, 1.0, -1.0) - alpha_m * calc_cel(kc_m, 1.0, 1.0, -1.0))
    B_z = B0 * a / (a+rho) * (beta_p * calc_cel(kc_p, gamma**2, 1.0, gamma) - beta_m * calc_cel(kc_m, gamma**2, 1.0, gamma))

    return B_rho, B_z


def plot_field(a, b, Br, extent=3.0, n=200):
    """Heatmap of |B| in the x-z plane plus field lines. Lengths in the units of a and b.
    This part was made with the help of AI (it takes me too long to make fancy pictures like this :P)"""
    x = np.linspace(-extent * a, extent * a, n)
    z = np.linspace(-extent * a, extent * a, n)
    X, Z = np.meshgrid(x, z)
    Bx = np.empty_like(X)
    Bz = np.empty_like(X)

    for i in range(n):
        for j in range(n):
            B_rho, B_z = field(abs(X[i, j]), Z[i, j], b, a, Br)
            Bx[i, j] = B_rho * np.sign(X[i, j])  # B_rho points away from the axis
            Bz[i, j] = B_z

    Bmag = np.hypot(Bx, Bz)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(X, Z, Bmag, norm=LogNorm(vmin=1e-3, vmax=Bmag[np.isfinite(Bmag)].max()),
                       cmap="magma", shading="auto")
    fig.colorbar(im, ax=ax, label="|B| [T]")
    ax.streamplot(x, z, Bx, Bz, color="white", linewidth=0.6, density=1.2, arrowsize=0.7)
    ax.add_patch(Rectangle((-a, -b), 2 * a, 2 * b, fill=False, edgecolor="cyan", linewidth=1.5))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("|B| of cylindrical magnet")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def field_along_z(z, b, a, Br):
    '''This calculates the field along the z-axis and is given as equation 2 of the paper. This is uses as a test assert.'''

    # We see that the constant term is: mu0*n*I/2 (let's call it B0z), whereas in equation 5 we see B0=mu0*n*I/pi. See the derivation of the function Br_to_B0 and you see that along the same arguments we get in this case that B0z is as follows:
    B0z = Br / 2 
    B_z = B0z * ( (z + b) / np.sqrt((z+b)**2 + a**2) - (z - b) / np.sqrt((z-b)**2 + a**2) )
    return B_z

def main():
    Br = 1.48    # [Tesla] magnetic remanence
    a = 0.009525 # [m] radius
    b = 0.00635  # [m] half height (origin is at center of cylinder)

    plot_field(a, b, Br)

    # confirm surface field from specsheet
    rho, z = 0, b
    B_rho, B_z = field(rho, z, b, a, Br)
    B_z_simple = field_along_z(z, b, a, Br)
    print(B_rho, B_z)
    print(B_z_simple)



if __name__ == "__main__":
    main()