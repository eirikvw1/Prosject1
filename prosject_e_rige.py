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
        plt.title(title)
        fig.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\plotR.png')
    else:
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title)
"""       
def MSE_R2(z,z_s,n):
    mse= (np.sum((z-z_s)**2, axis=0))/n
    r=1-(mse/(sum((z-np.mean(z))**2)))
    return mse, r
def Bias_var(z,z_p):
    var = np.sum(z_p**2)/len(z_p)-(np.sum(z_p)/len(z_p))**2
    bias = (z-np.sum(z_p)/len(z_p))**2
    bias = (np.sum(bias)/len(bias))**2
    return bias,var
"""    
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
def MSE_R2(z,z_s,n):
    s=0
    r=0
    z_=MEAN(z)
    for i in range(len(z)):
            
            s+=(z[i]-z_s[i])**2
            r+=(z[i]-z_)**2
            
    
    
    mse=s/len(z) #np.mean( np.mean((z - z_s)**2) ) ...(np.sum((z-z_s)**2, axis=0))/n
    
    r2=1-(s/r)#1-(mse*(len(z))/(sum((z-np.mean(z))**2)))
    return mse, r2

def MEAN(z):
    s=0
    for i in range(len(z)):
        s+=z[i]
    return (s/len(z))
def Bias_var(z,z_p):
    s=0
    z_=MEAN(z_p)
    z_2=MEAN(z_p**2)
    for i in range(len(z)):
        s+=(z_p[i]-z_)**2
    var = z_2-z_**2#s/len(z_p)
    #var = np.sum(z_p**2)/len(z_p)-(np.sum(z_p)/len(z_p))**2
    bias = (z-MEAN(z_p))**2
    bias = (np.sum(bias)/len(bias))**2
    return bias,var
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
rows1=np.random.rand(patch_size_row,1)
rows1=sorted(rows1.reshape(1,-1)[0])
cols1=np.random.rand(patch_size_col,1)
cols1=sorted(cols1.reshape(1,-1)[0])
[C,R] = np.meshgrid(cols,rows)
x = C.reshape(-1,1)
y = R.reshape(-1,1)
num_data = patch_size_row*patch_size_col
# Find the start indices of each patch
num_patches = 5
np.random.seed(4155)

row_starts = np.random.randint(0,n-patch_size_row,num_patches)
col_starts = np.random.randint(0,m-patch_size_col,num_patches)
d1=(8,9)
MA_mse = np.zeros(d1)
MA_r2 = np.zeros(d1)
MA_beta = np.zeros(d1)
MA_var = np.zeros(d1)
MA_bias = np.zeros(d1)

Grad =7
d=(5,9)


for g in range(4,12):
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
            lam_values = [1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]
            num_values = len(lam_values)
            beta_r = np.zeros((1+g*2,num_values))
            I = np.eye(1+g*2)
            
            for j,lam in enumerate(lam_values):
                    beta = (np.linalg.inv(X_data.T@X_data + lam*I)@X_data.T@z)
            
            
     
                    X_pred=X_matrise2(cols1, rows1,g,100,beta)
       


       
                    z_p = np.ravel(X_pred)
                    mse, R2=MSE_R2(np.ravel(patch),z_p,len(z_p))
                    bias, var=Bias_var(np.ravel(patch),z_p)    
                    #mse = np.sum( (z_p- patch)**2 )/num_data
                    #var = np.sum( (z_p - np.mean(z_p))**2 )/num_data
                    #bias = (np.sum( (patch - np.mean(z_p))**2 )/num_data)**2
                    #R2 = 1 - np.sum( (z_p - patch)**2 )/np.sum( (patch - np.mean(patch))**2 )
                    print("patch %d, from (%d, %d) to (%d, %d)"%(i+1, row_start, col_start, row_end,col_end))
                    print("mse: %g\nR2: %g"%(mse, R2))
                    print("variance: %g"%var)
                    print("bias: %g\n"%bias)
                    M_r2[i][j] = R2
                    M_mse[i][j] =mse
                    M_var[i][j] = var
                    M_bias[i][j] = bias
                   
            #surface_plot(z_p,'Fitted terrain surface',patch)
            

        
            plt.show() 
           
    M_mseT = M_mse.T
    M_r2T = M_r2.T
    M_varT = M_var.T
    M_biasT = M_bias.T  
    
    for a in range(9):
       
       MA_mse[g-4][a]=np.sum(M_mseT[a])/5
       MA_r2[g-4][a]=np.sum(M_r2T[a])/5
       MA_var[g-4][a]=np.sum(M_varT[a])/5
       MA_bias[g-4][a]=np.sum(M_biasT[a])/5
#g=[4,5,6,7,8,9,10,11]
fig1=plt.figure()


for i in range(8):
    
    plt.plot(lam_values,MA_mse[i],'-', label=("Grad",i+1))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
    
plt.xlabel(r'$Lamda$')
plt.ylabel(r'$MSE$')
plt.xscale('log')
#plt.yscale('log')
plt.title('Polynom grad')
plt.xscale('log')


#fig1.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\MSE(ERi).png')
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
#fig2.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\r2(ERi).png')
plt.show()

fig3=plt.figure()
for i in range(8):
    
    plt.plot(lam_values,MA_var[i],'-', label=("Grad",i+1)) 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
plt.xlabel(r'$lamda$')
plt.ylabel(r'$Varians$')
plt.xscale('log')
#plt.yscale('log')
plt.title('Polynom grad')
#fig3.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\var(ERi).png')
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
#fig4.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\bias(ERi).png')
plt.show()   

