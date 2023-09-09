"""
Use Galerkins method to solve the 1D Poisson equation in the sense that 
I need to solve for the shape coefficient C for Poiseuille flow.
"""
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from scipy.special import beta


plt.rcParams["font.family"] = "BigBlueTerm437 Nerd Font Mono"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'BigBlueTerm437 Nerd Font Mono'
plt.rcParams['mathtext.it'] = 'BigBlueTerm437 Nerd Font Mono:italic'
plt.rcParams['mathtext.bf'] = 'BigBlueTerm437 Nerd Font Mono:bold'
plt.rcParams['mathtext.sf'] = 'BigBlueTerm437 Nerd Font Mono'

# Text color #97FC1A
# ['#000000', '#11140B', '#23301C', '#2F4A2A', '#356636', '#368440', '#30A245', '#32C141', '#5EE032', '#97FC1A']

class GalerkinObject():
    def __init__(self,
                 M_dim: int,
                 N_dim: int,
                 ):
        self.phi = np.linspace(0, np.pi, 1000)
        self.x = np.linspace(0, 1, 1000)
        self.X, self.PHI = np.meshgrid(self.x, self.phi)
        self.m = 0
        self.n = 1
        self.M_dim = M_dim
        self.N_dim = N_dim
        # Sub colormap for better contrast on our quasi-green screen
        self.cmap = cmr.get_sub_cmap('cmr.nuclear', 0.2, 0.8)
    
    def basis_function(self, x, phi, m, n):
        """
        Basis function for the Galerkin method. We are using
        cylindrical coordinates, so the basis function is given by
        Bessel functions.
        """
        values = x**(2*m+1) * (1-x)**n * np.sin((2*m+1)*phi)
        return values / np.max(values)
    
    def delta_function(self, m, m_prime):
        """
        Delta function for the Galerkin method.
        """
        if m == m_prime:
            return 1
        else:
            return 0
    
    def create_basis(self, x_dim: int, phi_dim: int):
        """
        Create the basis functions for the Galerkin method.
        """
        basis_vec = []
        for m in range(0, x_dim):
            for n in range(1, phi_dim + 1):
                Z = self.basis_function(self.X, self.PHI, m, n)
                basis_vec.append(Z)
        
        self.basis = np.array(basis_vec)

        return self.basis
    
    def _create_matrix(self, M_dim: int, N_dim: int):
        """
        Create the matrix for the Galerkin method.
        Calculating A_mn,m'n' = (\Delta \psi_mn, \psi_m'n')

        """
        self.A = np.zeros((M_dim*N_dim, M_dim*N_dim))
        for m in range(M_dim):
            for n in range(1, N_dim + 1):
                for m_prime in range(M_dim):
                    for n_prime in range(1, N_dim + 1):
                        self.A[m * N_dim + n - 1, m_prime * N_dim + n_prime - 1] = -(np.pi/2) * self.delta_function(m, m_prime) * \
                                                                                    (n*n_prime*(3+4*m))/(2+4*m+n+n_prime) * \
                                                                                    beta(n+n_prime-1, 3+4*m)

        return self.A
    
    def _create_vector(self, M_dim: int, N_dim: int):
        """
        Create the vector for the Galerkin method.
        Calculating b_mn = (f, \psi_mn)
        In our case f = -1.
        """
        self.b = np.zeros(M_dim*N_dim)
        for m in range(M_dim):
            for n in range(1, N_dim + 1):
                self.b[m*N_dim + n - 1] = - 2/(2*m+1) * beta(2*m+3, n+1)

        return self.b
        
    def solve(self):
        """
        Solve the Galerkin method.
        """
        self.create_basis(self.M_dim, self.N_dim)
        self._create_matrix(self.M_dim, self.N_dim)
        self._create_vector(self.M_dim, self.N_dim)
        self.C = 0
        A_inv = np.linalg.inv(self.A)
        for m in range(self.M_dim):
            for n in range(1, self.N_dim + 1):
                for m_prime in range(self.M_dim):
                    for n_prime in range(1, self.N_dim + 1):
                        #  b_mn * A_mn,m'n' * b_m'n' or b_i * A_i,j * b_j
                        self.C += self.b[m*self.N_dim + n - 1] * \
                                  A_inv[m * self.N_dim + n - 1, m_prime * self.N_dim + n_prime - 1] * \
                                  self.b[m_prime*self.N_dim + n_prime - 1]

        self.C *= -32/np.pi 

        return self.C             
    
    def plot_flow(self, Z: np.ndarray | None):
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), facecolor="#000000")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_thetalim(0, np.pi)

        m = self.m
        n = self.n

        if Z is None:
            Z = self.basis_function(self.X, self.PHI, self.m, self.n)

        # Plot the contourf with the colormap
        cnt = ax.contourf(self.PHI, self.X, Z, cmap=self.cmap)
        cb = fig.colorbar(ax=ax, mappable=cnt)

        # Spines for cb are called ['left', 'right', 'bottom', 'top', 'outline']
        # Got by calling cb.ax.spines.keys() and iterating over them
        cb.set_label(r"Flow Rate", color="#5EE032")
        cb.ax.xaxis.set_tick_params(color="#5EE032")
        cb.ax.yaxis.set_tick_params(color="#5EE032")
        cb.ax.tick_params(axis="x", colors="#5EE032")
        cb.ax.tick_params(axis="y", colors="#5EE032")
        cb.ax.spines['outline'].set_color("#97FC1A")
        cb.ax.spines['outline'].set_linewidth(3)
        cb.ax.spines['outline'].set_alpha(0.8)
        
        # Spines are called ['polar', 'start', 'end', 'inner']
        # Polar is outside, start is origin radial line, end is the 
        # end of the radial line and inner is the inner circle.
        # Got by calling ax.spines.keys() and iterating over them
        ax.xaxis.label.set_color("#5EE032")
        ax.yaxis.label.set_color("#5EE032")
        ax.tick_params(axis="x", colors="#5EE032")
        ax.tick_params(axis="y", colors="#5EE032")
        ax.spines['polar'].set_color("#97FC1A")
        ax.spines['polar'].set_linewidth(3)
        ax.spines['polar'].set_alpha(0.8)
        ax.spines['polar'].set_linestyle("--")
        ax.spines['end'].set_color("#97FC1A")
        ax.spines['end'].set_linewidth(3)
        ax.spines['end'].set_alpha(0.8)
        ax.spines['end'].set_linestyle("--")
        ax.spines['start'].set_color("#97FC1A")
        ax.spines['start'].set_linewidth(3)
        ax.spines['start'].set_alpha(0.8)
        ax.spines['start'].set_linestyle("--")

        plt.title(f"Basis function $m={m}$, $n={n}$", color="#5EE032")
        plt.grid(False)
        plt.savefig('filename.png', dpi=1200, bbox_inches="tight")
        plt.show()

GO = GalerkinObject(2, 2)

mat = GO.solve()
print(mat)
