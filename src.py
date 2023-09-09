"""
Use Galerkins method to solve the 1D Poisson equation in the sense that 
I need to solve for the shape coefficient C for Poiseuille flow.
"""
import numpy as np
import matplotlib.pyplot as plt


class GalerkinObject():
    def __init__(self) -> None:
        pass
    
    def basis_function(self, x, phi, m, n):
        """
        Basis function for the Galerkin method. We are using
        cylindrical coordinates, so the basis function is given by
        Bessel functions.
        """
        values = x**(2*m+1) * (1-x)**n * np.sin((2*m+1)*phi)
        return values / np.max(values)
    
    def plot_flow(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_thetalim(0, np.pi)
        phi = np.linspace(0, np.pi, 1000)
        x = np.linspace(0, 1, 1000)
        X, PHI = np.meshgrid(x, phi)
        Z = self.basis_function(X, PHI, 0, 0)
        cnt = ax.contourf(PHI, X, Z, cmap='magma')
        fig.colorbar(ax=ax, mappable=cnt)
        plt.title("Basis function")
        plt.grid(False)
        plt.show()

GO = GalerkinObject()
GO.plot_flow()
