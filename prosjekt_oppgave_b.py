from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

from scipy import linalg





#fig = plt.figure()
#ax = fig.gca(projection='3d')
# trening og test data
data_point=1000
n=100
antall_train= 10

num_values = len(lam_values)
x_data = (np.random.rand(data_point,1))
x_train=np.split(x_data,10)[:9]
x_test=np.split(x_data,10)[9:]
print('antall trenings data', len(x_test))
y= (np.random.rand(data_point,1))
y_train=np.split(y,10)[:9]
y_test=np.split(y,10)[9:]
def MSE_R2(z,z_s,n):
    mse= (np.sum((z-z_s)**2, axis=0))/n
    r=1-(mse/(sum((z-np.mean(z))**2)))
    return mse, r
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

def Bias_var(z,z_p):
    var = np.sum(z_p**2)/len(z_p)-(np.sum(z_p)/len(z_p))**2
    bias = (z-np.sum(z_p)/len(z_p))**2
    bias = (np.sum(bias)/len(bias))**2
    return bias,var



x_test, y_test = np.meshgrid(sorted(x_test[0]),sorted(x_test[0]))
z_test=FrankeFunction(x_test, y_test)

x_test_p=np.ravel(x_test)
y_test_p=np.ravel(y_test)
z_test_p=np.ravel(z_test)
Beta_liste= list()
L_mse=list()

L_r2=list()
L_bias=list()
L_var=list()

d=(1,1)
MA_mse = np.zeros(d)
MA_r2 = np.zeros(d)
MA_beta = np.zeros(d)
MA_var = np.zeros(d)
MA_bias = np.zeros(d)

Grad= 7
for g in range(5,6):
    d2=(1,1)
    M_mse = np.zeros(d2)
    M_r2 = np.zeros(d2)
    M_beta = np.zeros(d2)
    M_var = np.zeros(d2)
    M_bias = np.zeros(d2)
    
    
    for j in range(1):#len(x_train)):
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
        E =X.T@X
    
        lam_values = [1e-4]
        num_values = len(lam_values)
        beta_r = np.zeros((1+g*2,num_values))
        for i,lam in enumerate(lam_values):
            beta_r[:,i] = (linalg.inv(X.T@X + lam*I)@X.T@z_p)
            
            
            
        test_X= X_matrise(x_test_p,y_test_p,g,n) 
        z_s = (test_X@beta_r).T
        Beta_liste.append(beta_r)
        mse= np.zeros((num_values, 1))
        r2= np.zeros((num_values, 1))
        fig = plt.figure()
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(x_test,y_test,z_s.reshape(n,n),cmap=cm.viridis,linewidth=0)
        plt.title('Spåd overflate')

        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.plot_surface(x_test,y_test,z_test.reshape(n,n),cmap=cm.viridis,linewidth=0)
        plt.title(' Ekte Overflate')
        fig.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\r_plot.png')
        """
        for i in range(len(z_s)):
            a=z_s[i]
            m,r=MSE_R2(z_test_p,a,n*n)
            M_mse[j][i]= m
            M_r2[j][i] =r
                 
          
            mse[i]=m
            r2[i]=r

        L_mse.append(mse)
        L_r2.append(r2)
        bias=np.zeros(num_values)
        var=np.zeros(num_values)
        for i in range(len(z_s)):
            
        
            a=z_s[i]
            b,v =Bias_var(z_test_p,a)
            var[i]=v
            bias[i]=b
            M_bias[j][i] = b
            M_var[j][i]  = v
            
            
            
            
    L_bias.append(bias)
    L_var.append(var)
   
    M_mseT = M_mse.T
    M_r2T = M_r2.T
    M_varT = M_var.T
    M_biasT = M_bias.T     
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


fig1.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\MSE(R).png')
plt.show()

print(MA_r2)
fig2=plt.figure()
for i in range(6):
    
    plt.plot(lam_values,MA_r2[i],'-', label=("Grade", i+1)) 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  
plt.xlabel(r'$lamda$')
plt.ylabel(r'$r2$')
plt.xscale('log')
plt.title('Polynom grad')
fig2.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\r2(R).png')
plt.show()

fig3=plt.figure()
for i in range(6):
    
    plt.plot(lam_values,MA_var[i],'-', label=("Grad",i+1)) 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)     
plt.xlabel(r'$lamda$')
plt.ylabel(r'$Varians$')
plt.xscale('log')
plt.title('Polynom grad')
fig3.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\var(R).png')
plt.show()

fig4=plt.figure()

for i in range(6):
    
    plt.plot(lam_values,MA_bias[i],'-', label=("Grad",i+1)) 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)   

plt.xlabel(r'$lamda$')
plt.ylabel(r'$Bias^2$')
plt.xscale('log')
plt.title('Polynom grad')
fig4.savefig('C:\\Users\\eirik\\OneDrive\\Dokumenter\\Fys-stk\\bias(R).png')
plt.show()

plt.plot(lam_values,MA_bias[i],'-', label=("Grad",i+1)) 

plt.xlabel(r'$lamda$')
plt.ylabel(r'$Bias^2$')
plt.xscale('log')
plt.title('Polynom grad')

for i in range(len(mse)):
   plt.plot(lam_values,L_mse[i],'-', label="MSE")    

plt.xlabel(r'$Lamda$')
plt.ylabel(r'$MSE$')
plt.title('9 trenings data')
plt.figure()
for i in range(len(mse)):
   plt.plot(lam_values,L_r2[i],'-', label="R2")    

plt.xlabel(r'$Lamda$')
plt.ylabel(r'$R2$')
plt.title('9 trenings data')



plt.figure()

       

  
for i in range(len(mse)):
   plt.plot(lam_values,L_bias[i],'-', label="R2")    

plt.xlabel(r'$Lamda$')
plt.ylabel(r'$bias$')
plt.title('9 trenings data')
plt.figure()
print()
for i in range(len(mse)):
   plt.plot(lam_values,L_var[i],'-', label="R2")    

plt.xlabel(r'$Lamda$')
plt.ylabel(r'$var$')
plt.title('9 trenings data')
"""



