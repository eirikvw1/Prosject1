import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

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
        plt.title(title)
        fig.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\PlotL.png')
    else:
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title)
def MSE_R2(z,z_s,n):
    mse= (np.sum((z-z_s)**2))/n
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
def X_matrise2(x,y,G,b,z,l):
   
    lasso=Lasso(alpha= lam_values[l])
    lasso.fit(b,z)
    out = np.zeros((np.size(y), np.size(x)))
     
    
    v=(G*2+1)
    
    
  
    
    
    for i,y_ in enumerate(y):
        X1=np.zeros((np.size(x),v))
        
        for j,x_ in enumerate(x):
          
            for g in range(0,G+1):
               
                if g == 0:
                    X1[j][g]=1
                    
                    X1[j][2*g+1]=(x_)**(g+1)
                    
                elif g!=G:
                    
                    X1[j][2*g+1]=(x_)**(g+1)
                    
                    X1[j][2*g]=(y_)**(g)
                    
                else:
                    X1[j][2*g]=(y_)**(g)
      
        z_s=lasso.predict(X1)
        print(z_s)
        print('..')
        
        for a in range(len(z_s)):
            out[i][a]=z_s[a]
            
  
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
d1=(1,1)
MA_mse = np.zeros(d1)
MA_r2 = np.zeros(d1)
MA_beta = np.zeros(d1)
MA_var = np.zeros(d1)
MA_bias = np.zeros(d1)

Grad =7
d=(1,1)


for g in range(10,11):
    M_mse = np.zeros(d)
    M_r2 = np.zeros(d)
    M_beta = np.zeros(d)
    M_var = np.zeros(d)
    M_bias = np.zeros(d)
    teller= 0
    
    for i,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
            
            row_end = row_start + patch_size_row
            col_end = col_start + patch_size_col

            patch = terrain1[row_start:row_end, col_start:col_end]

            z = patch.reshape(-1,1)
            X_data=X_matrise(x,y, g,num_data)
            lam_values = [1e-4]
            num_values = len(lam_values)
            beta_r = np.zeros((1+g*2,num_values))
            I = np.eye(1+g*2)
            
            for l in range(num_values):
                    
        
    
                    X_pred=X_matrise2(cols, rows,g,X_data,z,l)
       


       
                    z_p = X_pred
                    
                    mse = np.sum( (z_p- patch)**2 )/num_data
                    var = np.sum( (z_p - np.mean(z_p))**2 )/num_data
                    bias = (np.sum( (patch - np.mean(z_p))**2 )/num_data)**2
                    R2 = 1 - np.sum( (z_p - patch)**2 )/np.sum( (patch - np.mean(patch))**2 )
        
                    M_mse[i][l]=mse  
                    M_r2[i][l]=R2  
                    M_bias[i][l]=bias  
                    M_var[i][l]=var 
                    
        
        
                    
            surface_plot(z_p,'Fitted terrain surface',patch)
            
    M_mseT = M_mse.T
    M_r2T = M_r2.T
    M_varT = M_var.T
    M_biasT = M_bias.T  
    """
    for a in range(7):
       
       MA_mse[g-4][a]=np.sum(M_mseT[a])/2.0
       MA_r2[g-4][a]=np.sum(M_r2T[a])/2.0
       MA_var[g-4][a]=np.sum(M_varT[a])/2.0
       MA_bias[g-4][a]=np.sum(M_biasT[a])/2.0    
    """
"""    
plt.show() 
for i in range(8):
    
    plt.plot(lam_values,MA_mse[i],'-', label=("Grad",i+1))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
    
plt.xlabel(r'$Lamda$')
plt.ylabel(r'$MSE$')
plt.xscale('log')
plt.yscale('log')
plt.title('Polynom grad')
plt.xscale('log')


fig1.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\MSE(ELi).png')
plt.show()

print(MA_r2)
fig2=plt.figure()
for i in range(8):
    
    plt.plot(lam_values,MA_r2[i],'-', label=("Grade", i+1)) 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
plt.xlabel(r'$lamda$')
plt.ylabel(r'$r2$')
plt.xscale('log')
plt.title('Polynom grad')
fig2.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\r2(ELi).png')
plt.show()

fig3=plt.figure()
for i in range(8):
    
    plt.plot(lam_values,MA_var[i],'-', label=("Grad",i+1)) 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
plt.xlabel(r'$lamda$')
plt.ylabel(r'$Varians$')
plt.xscale('log')
plt.yscale('log')
plt.title('Polynom grad')
fig3.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\var(ELi).png')
plt.show()

fig4=plt.figure()

for i in range(8):
    
    plt.plot(lam_values,MA_bias[i],'-', label=("Grad",i+1)) 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)   

plt.xlabel(r'$lamda$')
plt.ylabel(r'$Bias^2$')
plt.xscale('log')
#plt.yscale('log')
plt.title('Polynom grad')
fig4.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\bias(ELi).png')
plt.show()   
"""