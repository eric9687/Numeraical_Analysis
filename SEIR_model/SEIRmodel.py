import matplotlib.pyplot
import numpy as np
import gmpy2
import math

# parameter of SEIR model
N=10000.
h = 0.01 
r = 10.
r_beta = r*0.02 
alpha = 0.4 
ganmma = 0.5 


T = 15. # total days of SEIR model


def analysis_error(h_,y): # analzing error of Euler method
    num_steps = int(T / h_)
    y = np.asmatrix(y)
    y_T = y.T
    B = np.asmatrix(np.array([-r_beta/N,r_beta/N,0,0]))
    B = B.T
    A = np.asmatrix(np.array([[0,0,ganmma,0],[0,0,0,0],[ganmma,0,0,0],[0,0,0,0]]))
    C = np.asmatrix(np.array([[0,0,0,0],[0,-alpha,0,0],[0,alpha,-ganmma,0],[0,0,ganmma,0]]))

    y_= B * y_T * A * y + C * y
    dui_x = (2*B*y_T*A+C)*(B*y_T*A+C)*y
    dui_y = 2*B*y_T*A+C
    M = np.max(abs(np.asarray(dui_y)))
    L = np.max(abs(np.asarray(dui_x)))
    fangfa = (1+h_*M)**(num_steps+1)*(0+1/(h_*M)*L*h_**2/2)

    m = -math.log10(bound/2/((1+h_*M)**5*1/2/(h_*M)))  # the m of "sheruwucha"
    m = math.ceil(m) # min of m
    sheru = (1+h_*M)**5*1/2/(h_*M)*10**(-m) 


    return fangfa, m,sheru


def seir_model_Euler_Method(bound,h_):
    num_steps = int(T / h_)
    times = h_ * np.array(range(num_steps + 1))

    s = np.zeros(num_steps + 1)
    e = np.zeros(num_steps + 1)
    i = np.zeros(num_steps + 1)
    r = np.zeros(num_steps + 1)

    s[0] = gmpy2.mpfr(8000.)
    e[0] = gmpy2.mpfr(2000.)
    i[0] = gmpy2.mpfr(0.)
    r[0] = gmpy2.mpfr(0.)

    for step in range(num_steps):
        s[step+1] = s[step] - h_*(r_beta*i[step]*s[step]/N)
        e[step+1] = e[step] + h_*(r_beta*i[step]*s[step]/N - alpha*e[step])
        i[step+1] = i[step] + h_*(alpha*e[step] - ganmma*i[step])
        r[step+1] = r[step] + h_*(ganmma*i[step])
    

    s_max=max(s)
    e_max=max(e)
    i_max=max(i)
    r_max=max(r)
    y = np.vstack([s_max,e_max,i_max,r_max])
    f_e,m,s_e = analysis_error(h_,y)

    # checking the error matching the error bound
    if  f_e > bound/2 :
       
        h_ = h_/10
        return seir_model_Euler_Method(bound,h_)
    if f_e < bound/2 :
        print("h:")
        print(h_)
        print("m:")
        print(m)
        print("fangfa_error:")
        print(f_e)
        print("sheru_error:")
        print(s_e)

        # graph of SEIR model
        s_plot = matplotlib.pyplot.plot(times, s, label = 'S')
        e_plot = matplotlib.pyplot.plot(times, e, label = 'E')
        i_plot = matplotlib.pyplot.plot(times, i, label = 'I')
        r_plot = matplotlib.pyplot.plot(times, r, label = 'R')
        matplotlib.pyplot.legend(('S', 'E', 'I', 'R'), loc = 'upper right')
        
        axes = matplotlib.pyplot.gca()
        axes.set_xlabel('Time in days')
        axes.set_ylabel('Number of persons')
        matplotlib.pyplot.xlim(xmin = 0.)
        matplotlib.pyplot.ylim(ymin = 0.)
        matplotlib.pyplot.show()





if __name__ == "__main__":

    # bound = input("Please input the error bound:")
    bound =10
    seir_model_Euler_Method(bound,h)


    
