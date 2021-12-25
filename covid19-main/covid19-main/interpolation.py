import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 999.987084835664 
I0 = 1.007936429757913 
r_beta = 1

def load_table():
    df = pd.read_csv('LUT.csv')
    t = df['t'].values
    I = df['I'].values
    return t, I

def get_hermite_func(t, I):
    # t, I: given t_k and I(t_k)
    dI=I-(I**2)/N
    def hermite_func(x):
        Ix=[]
        for i in x:
            m,n=divmod(i,0.5)
            m=int(m)
            Ix.append(I[m]*(1+2*(i-t[m])/(t[m+1]-t[m]))*pow((i-t[m+1])/(t[m]-t[m+1]),2)\
                +I[m+1]*(1+2*(i-t[m+1])/(t[m]-t[m+1]))*pow((i-t[m])/(t[m+1]-t[m]),2)\
                    +dI[m]*(i-t[m])*pow((i-t[m+1])/(t[m]-t[m+1]),2)\
                        +dI[m+1]*(i-t[m+1])*pow((i-t[m])/(t[m+1]-t[m]),2))
        return Ix
    return hermite_func

def func(t):
    # calculate I(t)
    It = N*I0/(I0+(N-I0)*np.exp(-r_beta*t)) 
    return It


if __name__ == "__main__":
    tk, I_tk = load_table()
    t = np.arange(0, 15, 0.1)

    hermite_func = get_hermite_func(tk, I_tk)

    # calculate
    I = func(t)
    I_hat = hermite_func(t)

    print(np.abs(I_hat - I).max())

    fig = plt.figure()
    plt1=plt.plot(t,I)
    plt.setp(plt1, color='b', linewidth=3.0)
    plt2=plt.plot(t,I_hat)
    plt.setp(plt2, color='w', linewidth=1.0)
    plt.show()
