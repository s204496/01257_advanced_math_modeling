#%%
import sympy as sp
a = sp.Symbol("a", positive=True, real=True)
b = sp.Symbol("b", positive=True, real=True)

A_b = sp.Matrix([
    [0,-1],
    [a,-1]
])

B_b = sp.Matrix([
    [0,1],
    [0,1-b]
])

O_b = sp.Matrix([
    [0,0],
    [0,0]
])


A3 = sp.BlockMatrix([
    [A_b,B_b,O_b],
    [O_b,A_b,B_b],
    [B_b,O_b,A_b]
    ]).as_explicit()

A2 = sp.BlockMatrix([
    [A_b,B_b],
    [B_b,A_b]
    ]).as_explicit()


#%%
import matplotlib.pyplot as plt
import numpy as np
eigenvals = [i for i in list(A3.eigenvals().keys())]
print(eigenvals[0])


var_vals = {a:1,b:1}
for eig in eigenvals:
    display(eig.subs(var_vals).simplify().evalf())

#%%

N = 70
min_val, max_val =  1e-3,2
_a_s = np.linspace(min_val,max_val,N)
_b_s = np.linspace(min_val,max_val,N)

aa_s,bb_s = np.meshgrid(_a_s,_b_s)

a_s = aa_s.ravel()
b_s = bb_s.ravel()

bot = np.zeros_like(a_s)


#%%

def get_eigv_from_M(ai,bi):
    np_A3 = np.array(A3.subs({a:ai,b:bi}).tolist(),dtype=np.float64)
    eigenvals = np.linalg.eigvals(np_A3)
    poss = sum([val > -1e-8 for val in np.real(eigenvals)])
    return np.sort(eigenvals), poss

eigs, poss = get_eigv_from_M(1,2)

#%%

eigs_dict = {f"eig{i}":[] for i in range(len(eigs))}
posses = np.zeros((N,N))

for i,ai in enumerate(_a_s):
    for j,bi in enumerate(_b_s):
        eigenvals, poss = get_eigv_from_M(ai,bi)
        posses[i,j] = poss
        for key,val in zip(eigs_dict.keys(),eigenvals):
            eigs_dict[key] += [val]
        

#%%

eig_real = {key: np.real(val).reshape(aa_s.shape) for key, val in eigs_dict.items()}

# Plotting setup
fig, axes = plt.subplots(2, 3, subplot_kw={"projection": "3d"}, figsize=(18, 12))
axes = axes.flatten()

for idx, (key, eig_values) in enumerate(eig_real.items()):
    ax = axes[idx]
    
    # 3D surface plot
    surf = ax.plot_surface(aa_s, bb_s, eig_values.T, cmap='viridis', edgecolor='none')
    ax.set_title(f"{key}")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("Re(eigenvalue)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()

#%%

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(18, 12))

# 3D surface plot
surf = ax.plot_surface(aa_s, bb_s, posses.T, cmap='viridis', edgecolor='none')
ax.set_title("num pos eigenvalues for different alpha and betas")
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_zlabel("Num pos eignevalues")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()

# %%
