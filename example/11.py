import numpy as np
import matplotlib
from time import time as timeit
import matplotlib.pyplot as plt
import matplotlib.markers as markers
from matplotlib import rc
from matplotlib import animation
from scipy import integrate
import time
from scipy.optimize import curve_fit
from scipy.linalg import solve
from scipy import special
from time import (
    process_time,
    perf_counter,
    sleep,
)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({ 'savefig.dpi':300, "axes.labelweight" :"normal"})
matplotlib.rcParams['axes.linewidth'] = 0.8
sns.set_style("white")

reds = list(map(matplotlib.colors.rgb2hex, sns.color_palette('Reds', 200)))
viridis = list(map(matplotlib.colors.rgb2hex, sns.color_palette('viridis', 500)))

def psi(x, fi, m, n):
    return x**(2*m + 1) * (1-x)**n * np.sin((2*m + 1)*fi)

def scalar_product(m, n, m_, n_):
    if m != m_:
        return 0
    return np.pi/2 * special.beta(4 + 4*m, 1 + n + n_)

def scalar_product2(m, n, m_, n_):
    if m != m_:
        return 0
    return -np.pi/2 * n * n_ * (3+4*m)/(2+4*m + n + n_) * special.beta(n + n_ - 1, 3 + 4*m)

def scalar_product3(m, n):
    return -2/(2*m + 1) * special.beta(2*m + 3, n + 1)

def annotate_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_thetagrids(range(0, 210, 30))
        ax.set_rgrids([0, 0.5, 1.0])

fi = np.radians(np.linspace(0, 180, 1000 ))
r = np.arange(0, 1.0001, 0.001)
rs, fis = np.meshgrid(r, fi)


#slika1
fig = plt.figure(figsize=(10,6))
ax1 = plt.subplot2grid((8,12), (0, 0), rowspan=4, colspan=4, polar=True)
ax2 = plt.subplot2grid((8,12), (0, 4), rowspan=4, colspan=4, polar=True)
ax3 = plt.subplot2grid((8,12), (0, 8), rowspan=4, colspan=4, polar=True)
ax4 = plt.subplot2grid((8,12), (4, 0), rowspan=4, colspan=4, polar=True)
ax5 = plt.subplot2grid((8,12), (4, 4), rowspan=4, colspan=4, polar=True)
ax6 = plt.subplot2grid((8,12), (4, 8), rowspan=4, colspan=4, polar=True)

annotate_axes(fig)

contourf_ = ax1.contourf(fis, rs, psi(rs, fis, 0, 1), cmap='RdYlBu')
cbar = fig.colorbar(contourf_, ax=ax1, orientation='horizontal', shrink=0.8, ticks=[0, 0.08, 0.16, 0.24])

contourf_ = ax2.contourf(fis, rs, psi(rs, fis, 0, 3), cmap='RdYlBu')
cbar = fig.colorbar(contourf_, ax=ax2, orientation='horizontal', shrink=0.8, ticks=[0, 0.04, 0.08, 0.12])

contourf_ = ax3.contourf(fis, rs, psi(rs, fis, 0, 6), cmap='RdYlBu')
cbar = fig.colorbar(contourf_, ax=ax3, orientation='horizontal', shrink=0.8, ticks=[0, 0.02, 0.04, 0.06])

contourf_ = ax4.contourf(fis, rs, psi(rs, fis, 1, 1), cmap='RdYlBu')
cbar = fig.colorbar(contourf_, ax=ax4, orientation='horizontal', shrink=0.8, ticks=[-0.12, -0.06, 0, 0.06, 0.12])

contourf_ = ax5.contourf(fis, rs, psi(rs, fis, 2, 1), cmap='RdYlBu')
cbar = fig.colorbar(contourf_, ax=ax5, orientation='horizontal', shrink=0.8, ticks=[-0.08, -0.04, 0, 0.04, 0.08])

contourf_ = ax6.contourf(fis, rs, psi(rs, fis, 3, 1), cmap='RdYlGn')
cbar = fig.colorbar(contourf_, ax=ax6, orientation='horizontal', shrink=0.8, ticks=[-0.06, -0.03, 0, 0.03, 0.06])

ax1.set_title('$a) \, m=0, n=1$', fontsize=15, y=-0.01)
ax2.set_title('$b) \, m=0, n=3$', fontsize=15, y=-0.01)
ax3.set_title('$c) \, m=0, n=6$', fontsize=15, y=-0.01)
ax4.set_title('$d) \, m=1, n=1$', fontsize=15, y=-0.01)
ax5.set_title('$e) \, m=2, n=1$', fontsize=15, y=-0.01)
ax6.set_title('$f) \, m=3, n=1$', fontsize=15, y=-0.01)

fig.suptitle(r'$\psi_{mn}(\xi, \phi)= \xi^{2m+1}(1-\xi)^n \sin((2m+1)\phi)$', fontsize=18)
plt.tight_layout(pad = 1)
plt.savefig('lastnef.png')

# Poskusna funkcija $f=1-\xi$

ans = solve(A, b,  assume_a='sym')

fi = np.radians(np.linspace(0, 180, 1000 ))
r = np.arange(0, 1.0001, 0.001)
rs, fis = np.meshgrid(r, fi)

Ans = 0
for i in range(N):
    m = ms[i]
    for j in range(N):
        n = ns[j]
        
        Ans += ans[i*N + j] * psi(rs, fis, m, n)
        
'done'

N = 10
b = np.zeros(N**2)
ms = ns = range(N)

for i in range(N):
    m = ms[i]
    for j in range(N):
        n = ns[j]
        b[i*N + j] = 2/(2*m+1) * (special.beta(2*m+3, n+1) - special.beta(2*m+4, n+1))
        
Aa = np.zeros((N**2, N**2))
for i in range(N):
    m = ms[i]
    for j in range(N):
        n = ns[j]
        for k in range(n, N):
            n_ = ns[k]
            Aa[i*N + n_][i*N + n] = scalar_product(m, n, m, n_)
Aa = np.maximum(Aa, Aa.transpose())
'done'

anss = solve(Aa, b,  assume_a='sym')

fi = np.radians(np.linspace(0, 180, 1000 ))
r = np.arange(0, 1.0001, 0.001)
rs, fis = np.meshgrid(r, fi)

Anss = 0
for i in range(N):
    m = ms[i]
    for j in range(N):
        n = ns[j]
        
        Anss += anss[i*N + j] * psi(rs, fis, m, n)
        
'done'


fi = np.radians(np.linspace(0, 180, 1000 ))
r = np.arange(0, 1.0001, 0.001)
rs, fis = np.meshgrid(r, fi)
#slika2
fig = plt.figure(figsize=(6,4))
ax1 = plt.subplot2grid((8,8), (0, 2), rowspan=4, colspan=4, polar=True)
ax2 = plt.subplot2grid((8,8), (4, 0), rowspan=4, colspan=4, polar=True)
ax3 = plt.subplot2grid((8,8), (4, 4), rowspan=4, colspan=4, polar=True)

annotate_axes(fig)
levels = np.linspace(0, 1.1, 10)
contourf_ = ax3.contourf(fis, rs, solution100[0], cmap='PiYG', levels=levels)
contourf_ = ax2.contourf(fis, rs, solution10[0], cmap='PiYG', levels=levels)
contourf_ = ax1.contourf(fis, rs, 1-rs, cmap='PiYG', levels=levels)
cbar = fig.colorbar(contourf_, ax=ax1, shrink=1, ticks=[0, 0.3, 0.6, 0.9])

ax3.set_title(r'c) razvoj po $N=100\times 100$ funkcijah', fontsize=11, y=-0.01)
ax2.set_title(r'b) razvoj po $N=10\times 10$ funkcijah', fontsize=11, y=-0.01)
ax1.set_title(r'$ a) f(\xi)=1-\xi$', fontsize=13, y=-0.01)

fig.suptitle(r'Razvoj $f(\xi)$ po $\psi_{mn}(\xi, \phi)$', fontsize=13)

plt.tight_layout(pad = 1)
plt.savefig('razvoj.png')

def C(a, b):
    return -32/np.pi * np.sum(a*b)

def solver(M, N):
    start = time.time_ns()
    b = np.zeros(M*N)
    ms = range(M)
    ns = range(1, N+1)
    
    for i in range(M):
        m = ms[i]
        for j in range(N):
            n = ns[j]
            b[i*N + j] = scalar_product3(m, n)

    A = np.zeros((M*N, M*N))
    for i in range(M):
        m = ms[i]
        for j in range(N):
            n = ns[j]
            for k in range(0, N):
                n_ = ns[k]
                A[i*N + n_ - 1][i*N + n - 1] = scalar_product2(m, n, m, n_)
    A = np.maximum(A, A.transpose())

    ans = solve(A, b,  assume_a='gen')
    fi = np.radians(np.linspace(0, 180, 1000 ))
    r = np.arange(0, 1.0001, 0.001)
    rs, fis = np.meshgrid(r, fi)
    
    Ans = 0
    for i in range(M):
        m = ms[i]
        for j in range(N):
            n = ns[j]
            Ans += ans[i*N + j] * psi(rs, fis, m, n)
    
    return A, b, ans, Ans, C(ans, b), time.time_ns() - start


#slika3
fig = plt.figure(figsize=(10,8))
sol = solver(5,5)
ax1 = plt.subplot2grid((8,8), (0, 0), rowspan=4, colspan=4)
ax2 = plt.subplot2grid((8,8), (0, 4), rowspan=4, colspan=4)
ax3 = plt.subplot2grid((8,8), (4, 4), rowspan=4, colspan=4, polar=True)
ax4 = plt.subplot2grid((8,8), (4, 0), rowspan=4, colspan=4)

for i in range(len(sol[1])):
    ax1.errorbar(range(len(sol[1]))[i], np.abs(sol[1])[i], fmt='o', color=plasma[i//5 * 2], markersize=9)
for i in range(5):
    ax1.plot(range(len(sol[1]))[(5*i):(5*i+5)], np.abs(sol[1])[(5*i):(5*i+5)], color=plasma[i* 2])
ax1.plot([], [], color=plasma[0], label='$m=0$', lw=3)
ax1.plot([], [], color=plasma[2], label='$m=1$', lw=3)
ax1.plot([], [], color=plasma[4], label='$m=2$', lw=3)
ax1.plot([], [], color=plasma[6], label='$m=3$', lw=3)
ax1.plot([], [], color=plasma[8], label='$m=4$', lw=3)

ax1.grid(alpha=0.7)
ax1.set_title('$a) \, |b_i|$', fontsize=18)
ax1.legend(fontsize=12, ncol=2)
ax1.set_xlabel('$i$', fontsize=15)
ax1.set_yscale('log')

from matplotlib.colors import LogNorm
ht = sns.heatmap(np.abs(sol[0]), ax=ax2, cmap = 'PuRd',square=True,linestyle='-', norm=LogNorm())

ax2.set_title('$b) \, |A_{ij}|$', fontsize=18)

for i in range(5):
    ax2.axvline(x=i*5, color='k',linewidth=1.1)
    ax2.axhline(y=i*5, color='k',linewidth=1.1)
    
ax2.axvline(x=25, color='k',linewidth=2)
ax2.axhline(y=25, color='k',linewidth=2)
    
ax3.set_thetamin(0)
ax3.set_thetamax(180)
ax3.set_thetagrids(range(0, 210, 30))
ax3.set_rgrids([0, 0.5, 1.0])

contourf_ = ax3.contourf(fis, rs, sol[3], cmap='plasma', levels=np.linspace(0, 0.10, 11))
cbar = fig.colorbar(contourf_, ax=ax3, shrink=1, ticks=np.linspace(0, 0.10, 11))
ax3.set_title(r'$d)\, \tilde u(\xi, \phi)$', fontsize=18)

ax4.errorbar(range(len(sol[2])), sol[2], markersize=9, fmt='o', color=mako[150])
ax4.plot(range(len(sol[2])), sol[2], color=mako[-250])
ax4.grid(alpha=0.7)

ax4.set_title(r'$c)\, |a_i|$', fontsize=18)
ax4.set_xlabel('$i$', fontsize=15)

fig.suptitle('Reševanje po metodi Galerkina', fontsize=20)
plt.tight_layout()
plt.savefig('faze.png')

#slika4
fig = plt.figure(figsize=(8,6))
ax1 = plt.subplot2grid((8,8), (0, 0), rowspan=4, colspan=4, polar=True)
ax2 = plt.subplot2grid((8,8), (0, 4), rowspan=4, colspan=4, polar=True)
ax3 = plt.subplot2grid((8,8), (4, 0), rowspan=4, colspan=4, polar=True)
ax4 = plt.subplot2grid((8,8), (4, 4), rowspan=4, colspan=4, polar=True)
annotate_axes(fig)

contourf_ = ax1.contourf(fis, rs, solver(1,1)[3], cmap='plasma', levels=levels)
contourf_ = ax2.contourf(fis, rs, solver(3,1)[3], cmap='plasma', levels=levels)
contourf_ = ax3.contourf(fis, rs, solver(1,3)[3], cmap='plasma', levels=levels)
contourf_ = ax4.contourf(fis, rs, solver(10,10)[3], cmap='plasma', levels=levels)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.45, 0.15, 0.03, 0.7])
fig.colorbar(contourf_, cax=cbar_ax)

ax1.set_title('$a)\,M=N=1$', fontsize=15)
ax2.set_title('$b)\,M=3, N=1$', fontsize=15)
ax3.set_title('$c)\,M=1, N=3$', fontsize=15)
ax4.set_title('$d)\,M=N=10$', fontsize=15)

fig.suptitle(r'Hitrostni profil $\tilde u$ za različno število testnih funkcij', fontsize=18)

plt.tight_layout()
plt.savefig('hitrostprofil.png')


#animacija
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

levels = np.linspace(0.0, 0.12, 13)
fig.suptitle(r'Hitrostni profil $\tilde u$ za različno število testnih funkcij', fontsize=15)
def init():
    contourf_ = ax.contourf(fis, rs, solver(1, 1)[3], levels=levels,  cmap='plasma')
    cbar = fig.colorbar(contourf_, ax=ax, shrink=0.5, ticks=levels, orientation='horizontal')
    cbar.ax.set_xticklabels(levels, rotation=90)
    
combs = [(1, 1), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 4), (5, 5), (6,6), (7,7)]
def animate(i):
    ax.clear()
    print(i)
    sol = solver(combs[i][0], combs[i][1])
    contourf_ = ax.contourf(fis, rs, sol[3], levels=levels,  cmap='plasma')
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_thetagrids(range(0, 210, 30))
    ax.set_rgrids([0, 0.5, 1.0])
    ax.set_title(r'$M={}, N={}$'.format(combs[i][0], combs[i][1]))
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(combs), repeat=False)
plt.tight_layout()
anim.save('orbita5.gif', fps=2)
plt.show()

Ms = np.arange(1, 10)
Ns = np.arange(1, 10)

Cs = np.zeros((len(Ms), len(Ns)))
ts = np.zeros((len(Ms), len(Ns)))
for i in range(len(Ms)):
    print(i)
    for j in range(len(Ns)):
        koef, t = solver(Ms[i], Ns[j])[-2:]
        Cs[i][j] = koef
        ts[i][j] = t
        
'done'

#slika5
fig = plt.figure(figsize=(10,6))
ax1 = plt.subplot2grid((8,8), (0, 0), rowspan=4, colspan=4)
ax2 = plt.subplot2grid((8,8), (0, 4), rowspan=4, colspan=4)
ax3 = plt.subplot2grid((8,8), (4, 0), rowspan=3, colspan=4)
ax4 = plt.subplot2grid((8,8), (4, 4), rowspan=3, colspan=4)

ax1 = sns.heatmap(ts/1e9, ax=ax1, cmap="PuBuGn", cbar_kws={'label': r'$t\,\,\mathrm{[s]}$'}, xticklabels=Ms, yticklabels=Ns, linewidths=0.1, linecolor='black')
ax2 = sns.heatmap(np.abs(Cs - sol10[-2]), ax=ax2, cmap="viridis", cbar_kws={'label': r'$\Delta C$'}, xticklabels=Ms, yticklabels=Ns, linewidths=0.1, linecolor='black', norm=LogNorm())

ax3.errorbar(range(100), -32/np.pi* sol10[1] * sol10[2], fmt='o', markersize=5, alpha=0.8, color=viridis[50])
ax3.plot(range(100), -32/np.pi* sol10[1] * sol10[2], alpha=0.8, lw=0.5, color=viridis[50])

ax4.errorbar(range(100), -32/np.pi* np.cumsum(sol10[1]* sol10[2]), fmt='o', markersize=5, color=reds[150])
ax4.plot(range(100), -32/np.pi* np.cumsum(sol10[1]* sol10[2]), lw=0.5, color=reds[150])

ax4.set_xlabel('$k$', fontsize=15)
ax3.set_xlabel('$i$', fontsize=15)
ax3.grid(alpha=0.7)
ax4.grid(alpha=0.7)

ax4.hlines(sol10[-2], 0, 110, zorder=10, color=viridis[-30], lw=3)

ax4.set_title(r'$d)\, C_k = -32/\pi \cdot \sum_{i=0}^k a_i b_i$', fontsize=15)
ax3.set_title(r'$c)\, -32/\pi \cdot a_i b_i$', fontsize=18)

ax1.set_title('$a)\,$ čas za izračun kot funkcija $M$, $N$', fontsize=18)
ax2.set_title('$b)\, C_{MN} - C_{10, 10}$', fontsize=18)

ax4.text(0.7, 0.7, r'$C\approx 0.75759$', transform=ax4.transAxes, fontsize=14, backgroundcolor=viridis[-30])

plt.tight_layout()
plt.savefig('casC.png')


# dodatna naloga

xi = np.linspace(0, 2*np.pi, 100)
t = np.linspace(0, 3, 100)

def psik(k, xi):
    return 1/(2*np.pi)**0.5 * np.exp(1j*k*xi)

def ak0(k):
    func_r = lambda x: np.sin(np.pi*np.cos(x)) * np.conjugate(psik(k, x)).real
    func_i = lambda x: np.sin(np.pi*np.cos(x)) * np.conjugate(psik(k, x)).imag
    return integrate.quad(func_r, 0, 2*np.pi)[0] + 1j* integrate.quad(func_i, 0, 2*np.pi)[0]

def ak0_analytic(k):
    return np.sin(k*np.pi/2)*special.jv(k, np.pi)* (2*np.pi)**0.5

def ak(k, t):
    def f(x, t):
        return 1j*k*x
    n = len(t)
    x = np.zeros(n, dtype=np.complex128)
    x[0] = ak0(k)
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = h * f( x[i], t[i] )
        k2 = h * f( x[i] + 0.5 * k1, t[i] + 0.5 * h )
        k3 = h * f( x[i] + 0.5 * k2, t[i] + 0.5 * h )
        k4 = h * f( x[i] + k3, t[i+1] )
        x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0
    return x

def ak_analytic(k, t):
    return np.sin(k*np.pi/2)*special.jv(k, np.pi)*np.exp(1j*k*t) * (2*np.pi)**0.5

def res_analytic(xi, t): 
    res = np.zeros((len(t), len(xi)))
    for i in range(len(t)):
        res[i, :] = np.sin(np.pi*np.cos(xi + t[i]))
    return res

def initial(xi):
    return np.sin(np.cos(xi)*np.pi)

T = np.zeros(10)
for g in range(1):

    start = time.time_ns()

    N = 100
    aks = np.zeros((len(t), N), dtype=np.complex128)
    for n in range(-N//2, N//2):
        aks[:, n] = ak(n, t)

    psis = np.zeros((N, len(xi)), dtype=np.complex128)
    for n in range(-N//2, N//2):
        psis[n, :] = psik(n, xi)

    res = np.zeros((len(t), len(xi)), dtype=np.complex128)

    for i in range(len(t)):
        y = 0
        for j in range(N):
            y += aks[i][j] * psis[j]
        res[i, :] = y

    T[g] = (time.time_ns() - start )/1e9

'done'

res_a = res_analytic(xi, t)

#slika6
fig = plt.figure(figsize=(12,4))

ax1 = plt.subplot2grid((4,12), (0, 0), rowspan=4, colspan=4)
ax2 = plt.subplot2grid((4,12), (0, 8), rowspan=4, colspan=4)
ax3 = plt.subplot2grid((4,12), (0, 4), rowspan=4, colspan=4)

cmap1 = ax1.pcolormesh(xi, t, res.real, cmap = plt.get_cmap('mako'), shading='auto')
fig.colorbar(cmap1, ax=ax1)

cmap3 = ax3.pcolormesh(xi, t, res.imag, cmap = plt.get_cmap('BuGn'), shading='auto')
fig.colorbar(cmap3, ax=ax3)

cmap2 = ax2.pcolormesh(xi, t, np.abs(res.real - res_a), cmap = plt.get_cmap('bone_r'), shading='auto')
fig.colorbar(cmap2, ax=ax2)

ax1.set_xlabel(r'$\xi$', fontsize=15)
ax1.set_ylabel('$t$', fontsize=15)
ax2.set_xlabel(r'$\xi$', fontsize=15)
ax2.set_ylabel('$t$', fontsize=15)
ax3.set_xlabel(r'$\xi$', fontsize=15)
ax3.set_ylabel('$t$', fontsize=15)

ax1.set_xticks(np.arange(0, np.pi*2 + np.pi/4, np.pi/4))
labels = [r'$0$', r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$']
ax1.set_xticklabels(labels, fontsize=12)
ax2.set_xticks(np.arange(0, np.pi*2 + np.pi/4, np.pi/4))
ax2.set_xticklabels(labels, fontsize=12)
ax3.set_xticks(np.arange(0, np.pi*2 + np.pi/4, np.pi/4))
ax3.set_xticklabels(labels, fontsize=12)

ax1.set_title(r'$a)\, Re [\tilde u(\xi, t)]$', fontsize=15)
ax3.set_title(r'$b)\, Im [\tilde u(\xi, t)]$', fontsize=15)
ax2.set_title(r'$c)\, |Re[\tilde u(\xi, t)] - u(\xi, t)|$', fontsize=15)

fig.suptitle(r'Primerjava rešitve po metodi Galerkina $\tilde u$ z analitično rešitvijo $u$', fontsize=18)
plt.tight_layout()

plt.savefig('dod1.png')


# dodatna z diferencami
t = np.linspace(0, 3, 100)
x = np.linspace(0, np.pi*2, 100)
res_d_a = res_analytic(x, t)
dx = x[1] - x[0]
x = np.arange(0, np.pi*2 + 2*dx, dx)

h = t[1] - t[0]
k = x[1] - x[0]

T = np.zeros(15)
for g in range(15):
    start = time.time_ns()
    res_d = np.zeros((len(t), len(x)))
    res_d[0, :] = initial(x)

    for i in range(len(t)-1):
        res_d[i + 1, :] = res_d[i, :] + h/k * ( np.roll(res_d[i, :], -1) - res_d[i, :])
    
    T[g] = (time.time_ns() - start)/1e9
    
'done'

res_d = res_d[:, :-1]

#slika7
fig = plt.figure(figsize=(9,4))

ax1 = plt.subplot2grid((4,8), (0, 0), rowspan=4, colspan=4)
ax2 = plt.subplot2grid((4,8), (0, 4), rowspan=4, colspan=4)

cmap1 = ax1.pcolormesh(xi, t, res_d, cmap = plt.get_cmap('viridis'), shading='auto')
fig.colorbar(cmap1, ax=ax1)


cmap2 = ax2.pcolormesh(xi, t, np.abs(res_d - res_d_a), cmap = plt.get_cmap('PuBu'), shading='auto')
fig.colorbar(cmap2, ax=ax2)

ax1.set_xlabel(r'$\xi$', fontsize=15)
ax1.set_ylabel('$t$', fontsize=15)
ax2.set_xlabel(r'$\xi$', fontsize=15)
ax2.set_ylabel('$t$', fontsize=15)

ax1.set_xticks(np.arange(0, np.pi*2 + np.pi/4, np.pi/4))
labels = [r'$0$', r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$', r'$2\pi$']
ax1.set_xticklabels(labels, fontsize=12)
ax2.set_xticks(np.arange(0, np.pi*2 + np.pi/4, np.pi/4))
ax2.set_xticklabels(labels, fontsize=12)

ax1.set_title(r'$a)\, \tilde u_d(\xi, t)$', fontsize=15)
ax2.set_title(r'$c)\, |\tilde u_d(\xi, t) - u(\xi, t)|$', fontsize=15)


fig.suptitle(r'Primerjava rešitve po diferenčni metodi $\tilde u_d$ z analitično rešitvijo $u$', fontsize=18)
plt.tight_layout()
plt.savefig('dod2.png')

