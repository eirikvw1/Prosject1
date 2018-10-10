from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import numpy as np
from random import random, seed
import matplotlib.pyplot as plt
from matplotlib import cm

def MSE_R2(z,z_s,n):
    mse= (np.sum((z-z_s)**2, axis=0))/n
    r=1-(mse/(sum((z-np.mean(z))**2)))
    return mse, r
def Bias_var(z,z_p):
    var = np.sum(z_p**2)/len(z_p)-(np.sum(z_p)/len(z_p))**2
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
    v=( j*2+1,n)
    X=np.zeros(v)
  
    
    
    for i in range(0,j+1):
        
        
        if i == 0:
            X[0]=np.ones(100)
            
            X[2*i+1]=(x.T)**(i+1)
            
        elif i!=j:
            X[2*i+1]=(x.T)**(i+1)
            X[2*i]=(y.T)**(i)
        else:
            X[2*i]=(y.T)**(i)
       
      
            
  
    return X.T

def Bias_var(z,z_p):
    var = np.sum(z_p**2)/len(z_p)-(np.sum(z_p)/len(z_p))**2
    bias = (z-np.sum(z_p)/len(z_p))**2
    bias = (np.sum(bias)/len(bias))**2
    return bias,var


n=100
antall_train= 10
data_point=1000

x_data = (np.random.rand(data_point,1))
x_train=np.split(x_data,10)[:9]
x_test=np.split(x_data,10)[9:]

y= (np.random.rand(data_point,1))
y_train=np.split(y,10)[:9]
y_test=np.split(y,10)[9:]
sorted(x_test) 
sorted(y_test)


d=(6,9)
MA_mse = np.zeros(d)
MA_r2 = np.zeros(d)
MA_beta = np.zeros(d)
MA_var = np.zeros(d)
MA_bias = np.zeros(d)

Grad= 7
for g in range(1,Grad):
    
    d2=(9,9)
    M_mse = np.zeros(d2)
    M_r2 = np.zeros(d2)
    M_beta = np.zeros(d2)
    M_var = np.zeros(d2)
    M_bias = np.zeros(d2)

    for j in range(len(x_train)):
        lam_values = [1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]
        num_values = len(lam_values)
        x=x_train[j]
        y=y_train[j]
        
        sorted(x)
        sorted(y)
           
        
        
        z_t = FrankeFunction(x, y)
    
    

   
 
        
        X = X_matrise(x,y,g,n)
    
        for l in range(num_values):
            lasso=linear_model.Lasso(alpha= lam_values[l])
            lasso.fit(X,z_t)
        
    
            X_test= X_matrise(x_test[0],y_test[0],g,n)   
            z_s=lasso.predict(X_test)
            z_s = np.reshape(z_s, (100, 1))
            z = FrankeFunction(x_test[0], y_test[0])
        
        
            mse,r2=MSE_R2(z,z_s,n)
            bias,var=Bias_var(z,z_s)
        
            M_mse[j][l]=mse  
            M_r2[j][l]=r2  
            M_bias[j][l]=bias  
            M_var[j][l]=var 
    print('M_mse')         
    print(M_mse[0])        
    M_mseT = M_mse.T
    M_r2T = M_r2.T
    M_varT = M_var.T
    M_biasT = M_bias.T
    print(M_mseT[0])     
    for i in range(9):
       
       MA_mse[g-1][i]=np.sum(M_mseT[i])/9.0
       MA_r2[g-1][i]=np.sum(M_r2T[i])/9.0
       MA_var[g-1][i]=np.sum(M_varT[i])/9.0
       MA_bias[g-1][i]=np.sum(M_biasT[i])/9.0
       max_r2=np.zeros(9)
  

g=[1,2,3,4,5,6]
fig1=plt.figure()
for i in range(6):
    
    plt.plot(lam_values,MA_mse[i],'-', label=("Grad",i+1))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
    
plt.xlabel(r'$Lamda$')
plt.ylabel(r'$MSE$')
plt.title('Polynom grad')
plt.xscale('log')


fig1.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\MSE(L).png')
plt.show()


fig2=plt.figure()
for i in range(6):
    
    plt.plot(lam_values,MA_r2[i],'-', label=("Grade", i+1)) 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
plt.xlabel(r'$lamda$')
plt.ylabel(r'$r2$')
plt.xscale('log')
plt.title('Polynom grad')
fig2.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\r2(L).png')
plt.show()

fig3=plt.figure()
for i in range(6):
    
    plt.plot(lam_values,MA_var[i],'-', label=("Grad",i+1)) 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
plt.xlabel(r'$lamda$')
plt.ylabel(r'$Varians$')
plt.xscale('log')
plt.title('Polynom grad')
fig3.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\var(L).png')
plt.show()

fig4=plt.figure()

for i in range(6):
    
    plt.plot(lam_values,MA_bias[i],'-', label=("Grad",i+1)) 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)   

plt.xlabel(r'$lamda$')
plt.ylabel(r'$Bias^2$')
plt.xscale('log')
plt.title('Polynom grad')
fig4.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\bias(L).png')
plt.show()






