import numpy as np
from random import random, seed
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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
        s+=(z_p[i] -z_)**2
    var = z_2-z_**2#s/len(z_p)
    #var = np.sum(z_p**2)/len(z_p)-(np.sum(z_p)/len(z_p))**2
    bias = (z-np.sum(z_p)/len(z_p))**2
    bias = (np.sum(bias)/len(bias))**2
    return bias,var
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + 0.1*np.random.randn(100,1)

def X_matrise(x,y,j,n):
    v=(j*2+1,n*n)
    X=np.zeros(v)
    
    
    for i in range(0,j+1):
        
        
        if i == 0:
            X[0]=np.ones(10000)
            X[2*i+1]=x**(i+1)
        elif i!=j:
            X[2*i+1]=x**(i+1)
            X[2*i]=y**(i)
        else:
            X[2*i]=y**(i)
    return X.T       
"""
def Bias_var(z,z_p):
    var = np.sum(z_p**2)/len(z_p)-(np.sum(z_p)/len(z_p))**2
    bias = (z-np.sum(z_p)/len(z_p))**2
    bias = (np.sum(bias)/len(bias))**2
    return bias,var
    
"""    
    
    

n=100
antall_train= 10
data_point=1000

x_data = (np.random.rand(data_point,1))
x_train=np.split(x_data,10)[:9]
x_test=np.split(x_data,10)[9:]

y= (np.random.rand(data_point,1))
y_train=np.split(y,10)[:9]
y_test=np.split(y,10)[9:]
x_test, y_test = np.meshgrid(sorted(x_test[0]),sorted(x_test[0]))
z_test=FrankeFunction(x_test, y_test)

x_test_p=np.ravel(x_test)
y_test_p=np.ravel(y_test)
z_test_p=np.ravel(z_test)



Grad =7
d=(6,9)
M_mse = np.zeros(d)
M_r2 = np.zeros(d)
M_beta = np.zeros(d)
M_var = np.zeros(d)
M_bias = np.zeros(d)

for g in range(1,Grad):

    for j in range(len(x_train)):
        x=x_train[j]
        y=y_train[j]
       
        
        x, y = np.meshgrid(sorted(x),sorted(y))
        z = FrankeFunction(x, y)
    

        x_p=np.ravel(x)

        y_p=np.ravel(y)

        z_p=np.ravel(z)
   


# ruller ut datat saa den kan bli brukt
        z = FrankeFunction(x, y)
        x_p=np.ravel(x)

        y_p=np.ravel(y)

        z_p=np.ravel(z)
#lager matrisen av forsjellige grader

        #Grad= 5
   
        X = X_matrise(x_p,y_p,g,n)  
   
# skal spaa beta
        I = np.eye(1+g*2)
        E = np.linalg.inv(X.T@X)
        beta =E@(X.T@z_p)
       
            
            
            
        test_X= X_matrise(x_test_p,y_test_p,g,n) 
        z_s = (test_X@beta).T
       
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(x_test,y_test,z_s.reshape(n,n),cmap=cm.viridis,linewidth=0)
        plt.title('Spåd overflate')

        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.plot_surface(x_test,y_test,z_test.reshape(n,n),cmap=cm.viridis,linewidth=0)
        plt.title(' Ekte Overflate')
        #.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\olsA_plot.png')
        """
       
        
        mse, r2=MSE_R2(z_test_p,z_s,n)
        bias, var=Bias_var(z_test_p,z_s)
        M_mse[g-1][j]= mse
        M_r2[g-1][j]= r2
        M_var[g-1][j]= var
        M_bias[g-1][j]= bias
    
   
    

A_mse= np.zeros(6)
A_bias= np.zeros(6)
A_r2= np.zeros(6)
A_var= np.zeros(6)
M_r2T= M_r2.T        
M_mseT= M_mse.T
M_varT= M_var.T
M_biasT= M_bias.T

for i in range(6):
        A_mse[i]=np.sum(M_mse[i])/len(M_mse[i]) 
        A_r2[i]=np.sum(M_r2[i])/9.0 
        A_var[i]=np.sum(M_var[i])/9.0 
        A_bias[i]=np.sum(M_bias[i])/9.0 
        
print(M_mseT[0])

g=[1,2,3,4,5,6]

fig1 = plt.figure()
for i in range(9):
    
    plt.plot(g,M_mseT[i],'-', label="test") 
plt.plot(g,A_mse,'*', label="test")     
plt.xlabel(r'$Polynom grad$')
plt.ylabel(r'$MSE$')
plt.title('')



fig1.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\MSE.png')
plt.show()


fig2 = plt.figure()
for i in range(9):
    
    plt.plot(g,M_r2T[i],'-', label="test") 
plt.plot(g,A_r2,'*', label="test")     
plt.xlabel(r'$Polynom grad$')
plt.ylabel(r'$R2$')
plt.title('')



fig2.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\R2.png')
plt.show()



fig3 = plt.figure()
for i in range(9):
    
    plt.plot(g,M_varT[i],'-', label="test") 
plt.plot(g,A_var,'*', label="test")     
plt.xlabel(r'$Polynom grad$')
plt.ylabel(r'$var$')
plt.title('')



fig3.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\var.png')
plt.show()

fig4 = plt.figure()
for i in range(9):
    
    plt.plot(g,M_biasT[i],'-', label="test") 
plt.plot(g,A_bias,'*', label="test")     
plt.xlabel(r'$Polynom grad$')
plt.ylabel(r'$bias^2$')
plt.title('')



fig4.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\bias.png')
plt.show()

