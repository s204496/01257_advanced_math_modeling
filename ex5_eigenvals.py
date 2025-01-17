#%%
import sympy as sp
import numpy as np

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

def get_AN(N, get_block = False):
    if get_block == False:
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

    elif get_block:
        A_b = sp.Symbol("A")
        B_b = sp.Symbol("B")
        O_b = sp.Symbol("O")

    AN = []
    for ni in range(N):
        An = []
        for nj in range(N):
            if nj == 0 and ni == N-1:
                An.append(B_b)
            elif ni == nj:
                An.append(A_b)
            elif ni == nj-1:
                An.append(B_b)
            else:
                An.append(O_b)
        AN.append(An)
    
    if get_block:
        return sp.Matrix(AN)
    else:
        return sp.BlockMatrix(AN).as_explicit()

#%%
import matplotlib.pyplot as plt
import numpy as np
eigenvals = [i for i in list(A3.eigenvals().keys())]
print(eigenvals[0])


# var_vals = {a:1,b:1}
# eigenvals[1].subs(var_vals).simplify().evalf()

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

def get_P(N,get_block=False):
    if get_block:
        P = np.zeros((N,N),dtype=np.int16)
        for i in range(len(P)):
            P[-i-1][i]=1
    else:
        P = np.zeros((2*N,2*N),dtype=np.int16)
        for i in range(len(P)):
            P[-i-1][i]=1
    return sp.Matrix(P)

N = 4
get_block = False

AN = get_AN(N, get_block=get_block)

P = get_P(N,get_block)

ANP = AN @ P
ANP

#%%
ANP.eigenvals()