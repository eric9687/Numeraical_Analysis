import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(data_path):
    df = pd.read_csv(data_path)
    t = df['t'].values
    I = df['I'].values
    return t, I

def solve(t, I, r_beta):
    m = len(t)
    A = np.array((np.exp(-r_beta*t),np.ones(m))).T
    y = 1/I
    X = np.dot(A.T,A)
    X = np.linalg.inv(X)
    X = np.dot(X,A.T)
    X = np.dot(X,y)
    N = 1/X[1]
    I0 = 1/(X[0]+1/N)
    return N, I0

if __name__ == "__main__":
    # known parameters
    r_beta = 1

    # load data
    t, I = load_data('data.csv')
    # solve for unknown parameters
    N, I0 = solve(t, I, r_beta)
    print(N, I0)

    # show result 
    fig = plt.figure()
    plt.plot(t,I,'.')
    m=len(t)
    a = np.arange(t[0],t[m-1],0.1)
    b = [N*I0/(I0+(N-I0)*np.exp(-num)) for num in a]
    plt.plot(a,b,'r')
    plt.xlabel('t')
    plt.ylabel('I')
    plt.grid()
    title='Least Square Method: \n N = '+ str(round(N,5)) +', I0 = ' + str(round(I0,5)) \
        +',\nI(t)='+ str(round(N*I0,5)) +'/(' + str(round(I0,5)) + '+' + str(round(N-I0,5)) \
            + 'e^-t )'
    plt.suptitle(title)
    plt.show()