#%%
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

b = sp.Symbol("b",positive=True)
c = sp.Symbol("c",positive=True)
T = sp.Symbol("T",positive=True)
w = sp.Symbol("w",positive=True)

Av = sp.Abs((-sp.I *c +w*(-T*c+b))/(b*w-sp.I*c+sp.I*w**2))



f = sp.lambdify((c,w),(Av**2).subs({b:c*T}).subs({T:1}))
# f = sp.lambdify((c,w),(c**2/(c**2*w**2 + (w**2-c)*(w**2-c))))


N = 50
min_val, max_val =  1e-1,5
_a_s = np.linspace(min_val,max_val,N)
_b_s = np.linspace(min_val,2,N)

aa_s,bb_s = np.meshgrid(_a_s,_b_s)

a_s = aa_s.ravel()
b_s = bb_s.ravel()

bot = np.zeros_like(a_s)



A = f(1,2)

#%%

posses = np.zeros((N,N))

for i,ai in enumerate(_a_s):
    for j,bi in enumerate(_b_s):
        poss = f(ai,bi)
        posses[i,j] = poss
        
#%%

sum(sum(posses < 1)), sum(sum(posses > 1))

#%%



# Plotting setup
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(18, 12))

# 3D surface plot
surf = ax.plot_surface(aa_s, bb_s, posses.T, cmap='viridis', edgecolor='none')
surf = ax.plot_surface(aa_s, bb_s, np.ones_like(bb_s), color="red")
ax.set_title(f"fisk")
ax.set_xlabel("c")
ax.set_ylabel("w")
ax.set_zlabel("Re(eigenvalue)")
ax.set_zlim([0,1])
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()
# %%
