"""
Use Galerkins method to solve the 1D Poisson equation in the sense that 
I need to solve for the shape coefficient C for Poiseuille flow.
"""
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from scipy.special import beta
from time import perf_counter_ns


plt.rcParams["font.family"] = "BigBlueTerm437 Nerd Font Mono"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'BigBlueTerm437 Nerd Font Mono'
plt.rcParams['mathtext.it'] = 'BigBlueTerm437 Nerd Font Mono:italic'
plt.rcParams['mathtext.bf'] = 'BigBlueTerm437 Nerd Font Mono:bold'
plt.rcParams['mathtext.sf'] = 'BigBlueTerm437 Nerd Font Mono'
plt.rcParams.update({'axes.unicode_minus' : False})

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
        # Init empty variables
        self.basis = None
        self.A = None
        self.b = None
        self.a = None
        self.Z = None
    
    def basis_function(self, x, phi, m, n):
        """
        Basis function for the Galerkin method. We are using
        cylindrical coordinates, so the basis function is given by
        Bessel functions.
        """
        values = x**(2*m+1) * (1-x)**n * np.sin((2*m+1)*phi)
        return values
    
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

    def _solve_a(self):
        """
        Solve for flow field coefficients a_mn.
        """
        if self.basis is None:
            self.create_basis(self.M_dim, self.N_dim)
        if self.A is None:
            self._create_matrix(self.M_dim, self.N_dim)
        if self.b is None:
            self._create_vector(self.M_dim, self.N_dim)
        
        self.a = np.zeros(self.M_dim*self.N_dim)
        A_inv = np.linalg.inv(self.A)
        for m in range(self.M_dim):
            for n in range(1, self.N_dim + 1):
                for m_prime in range(self.M_dim):
                    for n_prime in range(1, self.N_dim + 1):
                        self.a[m_prime*self.N_dim + n_prime - 1] += A_inv[m * self.N_dim + n - 1, m_prime * self.N_dim + n_prime - 1] * \
                                                                    self.b[m*self.N_dim + n - 1]

        return self.a
    
    def solve_flow(self):
        """
        Solve for the flow field.
        """
        if self.a is None:
            self._solve_a()
        
        # self.Z = np.zeros((self.M_dim, self.N_dim, self.X.shape[0], self.X.shape[1]))
        self.Z = 0
        for m in range(self.M_dim):
            for n in range(1, self.N_dim + 1):
                self.Z += self.a[m*self.N_dim + n - 1] * self.basis_function(self.X, self.PHI, m, n)
        
        return self.Z
    
    def plot_flow(self, Z: np.ndarray | None, m: int = None, n: int = None, title: str = None, save: bool = False, filename: str = None):
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), facecolor="#000000")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_thetalim(0, np.pi)

        if Z is None:
            Z = self.basis_function(self.X, self.PHI, m, n)

        # Plot the contourf with the colormap
        cnt = ax.contourf(self.PHI, self.X, Z, cmap=self.cmap)
        cb = fig.colorbar(ax=ax, mappable=cnt)

        # Spines for cb are called ['left', 'right', 'bottom', 'top', 'outline']
        # Got by calling cb.ax.spines.keys() and iterating over them
        cb.set_label(r"Flow Rate $[\mathrm{arb. units}]$", color="#5EE032", labelpad=10, size=12)
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

        plt.title(f"{title}", color="#5EE032", size=14)
        plt.grid(False)
        plt.subplots_adjust(right=0.86)
        if save:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()


    def plot_gallery(self, Z: np.ndarray | None, m: int = None, n: int = None, title: str = None, save: bool = False, filename: str = None):
        """
        Here Z is an array of multiple basis functions.
        """
        if Z is None:
            Z = np.empty((self.M_dim, self.N_dim, self.X.shape[0], self.X.shape[1]))
            for m in range(self.M_dim):
                for n in range(1, self.N_dim + 1):
                    Z[m, n - 1] = self.basis_function(self.X, self.PHI, m, n)
        
        norm = plt.Normalize(vmin=np.min(Z), vmax=np.max(Z))


        fig, axes = plt.subplots(self.M_dim, self.N_dim, subplot_kw=dict(projection='polar'), facecolor="#000000")
        cnt = axes[0, 0].contourf(self.PHI, self.X, self.basis_function(self.X, self.PHI, 0, 1), cmap=self.cmap, norm=norm)
        cnt = axes[0, 1].contourf(self.PHI, self.X, self.basis_function(self.X, self.PHI, 1, 1), cmap=self.cmap, norm=norm)
        #cnt = axes[0, 2].contourf(self.PHI, self.X, self.basis_function(self.X, self.PHI, 2, 1), cmap=self.cmap, norm=norm)
        cnt = axes[1, 0].contourf(self.PHI, self.X, self.basis_function(self.X, self.PHI, 0, 2), cmap=self.cmap, norm=norm)
        cnt = axes[1, 1].contourf(self.PHI, self.X, self.basis_function(self.X, self.PHI, 1, 2), cmap=self.cmap, norm=norm)
        #cnt = axes[1, 2].contourf(self.PHI, self.X, self.basis_function(self.X, self.PHI, 2, 2), cmap=self.cmap, norm=norm)
        #cnt = axes[2, 0].contourf(self.PHI, self.X, self.basis_function(self.X, self.PHI, 0, 3), cmap=self.cmap, norm=norm)
        #cnt = axes[2, 1].contourf(self.PHI, self.X, self.basis_function(self.X, self.PHI, 2, 3), cmap=self.cmap, norm=norm)
        #cnt = axes[2, 2].contourf(self.PHI, self.X, self.basis_function(self.X, self.PHI, 3, 3), cmap=self.cmap, norm=norm)
        axes[0, 0].set_title("$m=0$ $n=1$", color="#5EE032", size=14)
        axes[0, 1].set_title("$m=1$ $n=1$", color="#5EE032", size=14)
        #axes[0, 2].set_title("$m=2$ $n=1$", color="#5EE032", size=14)
        axes[1, 0].set_title("$m=0$ $n=2$", color="#5EE032", size=14)
        axes[1, 1].set_title("$m=1$ $n=2$", color="#5EE032", size=14)
        #axes[1, 2].set_title("$m=2$ $n=2$", color="#5EE032", size=14)
        #axes[2, 0].set_title("$m=0$ $n=3$", color="#5EE032", size=14)
        #axes[2, 1].set_title("$m=1$ $n=3$", color="#5EE032", size=14)
        #axes[2, 2].set_title("$m=2$ $n=3$", color="#5EE032", size=14)
        # Set colormap norm

       
        # Plot the contourf with the colormap
        for m in range(axes.shape[0]):
             for n in range(1, axes.shape[1] + 1):
                ax = axes[m-1, n-2]  # -1 because of indexing -1 because of no n=0
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_thetalim(0, np.pi)
                

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
                ax.grid(False)

        # Spines for cb are called ['left', 'right', 'bottom', 'top', 'outline']
        # Got by calling cb.ax.spines.keys() and iterating over them
        cb = plt.colorbar(ax=axes, mappable=cnt)
        cb.set_label(r"Flow Rate $[\mathrm{arb. units}]$", color="#5EE032", labelpad=10, size=12)
        cb.ax.xaxis.set_tick_params(color="#5EE032")
        cb.ax.yaxis.set_tick_params(color="#5EE032")
        cb.ax.tick_params(axis="x", colors="#5EE032")
        cb.ax.tick_params(axis="y", colors="#5EE032")
        cb.ax.spines['outline'].set_color("#97FC1A")
        cb.ax.spines['outline'].set_linewidth(3)
        cb.ax.spines['outline'].set_alpha(0.8)

        plt.suptitle(f"Basis Functions", color="#5EE032", size=14)
        plt.grid(False)
        plt.subplots_adjust(right=0.70)
        if save:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()


def scale_it(N_range, M_range):
    """
    Scale the basis functions to see how they change with the 
    number of basis functions.
    """
    solutions = {}
    for M in range(1, M_range + 1):
        for N in range(1, N_range + 1):
            GO = GalerkinObject(M, N)
            print(f"Now solving for basis of size m={M}, n={N}")
            time = perf_counter_ns()
            C = GO.solve()
            time = (perf_counter_ns() - time)
            solutions[f"m={M}, n={N}"] = (C, time)

    return solutions

GO = GalerkinObject(2, 2)
m = 0
n = 3
GO.plot_flow(None, m=m, n=n, title=f"Basis Function $m={m}$, $n={n}$" )
quit()
# print(scale_it(5, 5))
GO = GalerkinObject(20, 20)
val = GO.solve()
Z = GO.solve_flow()
GO.plot_flow(Z, title=f"Flow Field $C = {round(val, 3)}$ M={GO.M_dim}, N={GO.N_dim}")

