import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
def surface_plot(surface,title, surface1=None):
    M,N = surface.shape

    ax_rows = np.arange(M)
    ax_cols = np.arange(N)

    [X,Y] = np.meshgrid(ax_cols, ax_rows)

    fig = plt.figure()
    if surface1 is not None:
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title)

        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.plot_surface(X,Y,surface1,cmap=cm.viridis,linewidth=0)
        fig.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\plot.png')
        
        plt.title(title)
    else:
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title)
def MSE_R2(z,z_s,n):
    mse= (np.sum((z-z_s)**2, axis=0))/n
    r=1-(mse/(sum((z-np.mean(z))**2)))
    return mse, r
def Bias_var(z,z_p):
    var = np.sum(z_p**2)/len(z_p)-(np.sum(z_p)/len(z_p))**2
    bias = (z-np.sum(z_p)/len(z_p))**2
    bias = (np.sum(bias)/len(bias))**2
    return bias,var
def X_matrise(x,y,j,n):
    
    v=( j*2+1,n)
    X=np.zeros(v)
    
  
    
    
    for i in range(0,j+1):
        
        
        if i == 0:
            X[0]=np.ones(n)
            
            X[2*i+1]=(x.T)**(i+1)
            
        elif i!=j:
            X[2*i+1]=(x.T)**(i+1)
            
            X[2*i]=(y.T)**(i)
        else:
            X[2*i]=(y.T)**(i)
       
      
            
  
    return X.T
def X_matrise2(x,y,G,n,b):
    out = np.zeros((np.size(y), np.size(x)))
    v=(G*2+1)
    
    
  
    
    
    for i,y_ in enumerate(y):
        
        for j,x_ in enumerate(x):
            X=np.zeros(v)
            for g in range(0,G+1):
               
                if g == 0:
                    X[g]=1
                    
                    X[2*g+1]=(x_)**(g+1)
                    
                elif g!=G:
                    X[2*g+1]=(x_)**(g+1)
                    
                    X[2*g]=(y_)**(g)
                    
                else:
                    X[2*g]=(y_)**(g)
                   
            out[i,j] = X@b
      
            
  
    return out
# Load the terrain
    


# Load the terrain
terrain1 = imread('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\SRTM_data_Norway_1.tif')
[n,m] = terrain1.shape

## Find some random patches within the dataset and perform a fit

patch_size_row = 300
patch_size_col = 150
# Define their axes
rows = np.linspace(0,1,patch_size_row)
cols = np.linspace(0,1,patch_size_col)
[C,R] = np.meshgrid(cols,rows)
x = C.reshape(-1,1)
y = R.reshape(-1,1)
num_data = patch_size_row*patch_size_col
# Find the start indices of each patch
num_patches = 1
np.random.seed(4155)

row_starts = np.random.randint(0,n-patch_size_row,num_patches)
col_starts = np.random.randint(0,m-patch_size_col,num_patches)


Grad =7
d=(8,1)
M_mse = np.zeros(d)
M_r2 = np.zeros(d)
M_beta = np.zeros(d)
M_var = np.zeros(d)
M_bias = np.zeros(d)

for g in range(11,12):
    
    for i,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
           
            row_end = row_start + patch_size_row
            col_end = col_start + patch_size_col

            patch = terrain1[row_start:row_end, col_start:col_end]

            z = patch.reshape(-1,1)
            X_data=X_matrise(x,y, g,num_data)
        
            beta_ols = np.linalg.inv(X_data.T@X_data)@X_data.T@z
     
            X_pred=X_matrise2(cols, rows,g,100,beta_ols)
       


       
            z_p = X_pred
        
        
            mse = np.sum( (z_p- patch)**2 )/num_data
        
            var = np.sum( (z_p - np.mean(z_p))**2 )/num_data
            bias = (np.sum( (patch - np.mean(z_p))**2 )/num_data)**2
        
            print("patch %d, from (%d, %d) to (%d, %d)"%(i+1, row_start, col_start, row_end,col_end))
            R2 = 1 - np.sum( (z_p - patch)**2 )/np.sum( (patch - np.mean(patch))**2 )
            print("variance: %g"%var)
            print("bias: %g\n"%bias)
            print("mse: %g\nR2: %g"%(mse, R2))
            
            surface_plot(z_p,'Fitted terrain surface',patch)
            
            
            M_mse[g-4][i] = mse
            M_r2[g-4][i] = R2
            
            M_var[g-4][i] = var
            M_bias[g-4][i] = bias

        
            plt.show() 
            
            
            
            
            
M_r2T= M_r2.T        
M_mseT= M_mse.T
M_varT= M_var.T
M_biasT= M_bias.T           
g=[4,5,6,7,8,9, 10,11]   
"""        
fig1 = plt.figure()


for i in range(5):
    
    plt.plot(g,M_mseT[i],'-', label="test") 
#plt.plot(g,A_mse,'*', label="test")     
plt.xlabel(r'$Polynom grad$')
plt.ylabel(r'$MSE$')
plt.title('')



fig1.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\MSEE_OLS.png')
plt.show()


fig2 = plt.figure()
for i in range(5):
    
    plt.plot(g,M_r2T[i],'-', label="test") 
#plt.plot(g,A_r2,'*', label="test")     
plt.xlabel(r'$Polynom grad$')
plt.ylabel(r'$R2$')
plt.title('')



fig2.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\R2E_OLS.png')
plt.show()



fig3 = plt.figure()
for i in range(5):
    
    plt.plot(g,M_varT[i],'-', label="test") 
#plt.plot(g,A_var,'*', label="test")     
plt.xlabel(r'$Polynom grad$')
plt.ylabel(r'$var$')
plt.title('')



fig3.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\varE_OLS.png')
plt.show()

fig4 = plt.figure()
for i in range(5):
    
    plt.plot(g,M_biasT[i],'-', label="test") 
#plt.plot(g,A_bias,'*', label="test")     
plt.xlabel(r'$Polynom grad$')
plt.ylabel(r'$bias^2$')
plt.title('')



fig4.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\biasE_OLS.png')
plt.show()       
"""
