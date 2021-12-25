import numpy as np
import matplotlib.pyplot as plt
from interpolation import func

N = 999.987084835664 
I0 = 1.007936429757913 
r_beta = 1


e = 2.7182818 #to avoid using np.exp()
e10 = e
for i in range(1,10):
    e10 *= e
et_integral=(e10-1)**2

## Solving integral by the four fundamental arithmetic operations
def solve_integral (n):
  
    if n % 2 == 1 and n != 1:  # odd num n, but not 1
        return -1/5*(1+e10) + 1/5*n*solve_integral(n-1)
    if n % 2 == 0 and n != 0:  # even num n, but not 0
        return -1/5*(1-e10) + 1/5*n*solve_integral(n-1)
    if n == 0 :
        return (e10-1)/5
    if n == 1 :
        return -(4*e10+6)/25

## Putting the coefficient of Legendre Pn into the list p
def solve_pn(n):

    p =[]
    p.append([])
    p[0].append(1)
    p.append([])
    p[1].append(1)
    i = 2
    while n !=1 and  n!=0:
        if i%2 ==0:
            p.append([])
            p[i].append((2*i-1)/i*p[i-1][0])
            for s in range(1,i//2):
                p[i].append((2*i-1)/i*p[i-1][s]-(i-1)/i*p[i-2][s-1])
            p[i].append(-(i-1)/i*p[i-2][(i-2)//2])
        if i%2 ==1:
            p.append([])
            p[i].append((2*i-1)/i*p[i-1][0])
            for s in range(1,i//2+1):
                p[i].append((2*i-1)/i*p[i-1][s]-(i-1)/i*p[i-2][s-1])
        if i ==n:
            break
        i += 1   

    return p[n]

## Calculating an of Legendre polynomials approximation
def solve_an(n):

    sum_an = 0
    p = solve_pn(n)
    i=0
    if n==0:
        return (e10-1)/5/2
    elif n==1:
        return -(4*e10+6)/25*3/2
    else:
        k=n
        while i != (n//2)+1 :
            sum_an += (2*n+1)/2*p[i]*solve_integral(k)
            i += 1
            k -= 2

    return sum_an

## Calculating pj with putting t 
def solve_pj(n,t):

    x =(-5+t)/-5
    p = solve_pn(n)

    sum_pj=0
    k=0
    while n != 1 and n!= 0:
        sum_pj += p[k]*x**(n-2*k)
        if k == (n//2):
            break
        k += 1
            
    if n==1:
        sum_pj += p[k]*x
    if n==0:
        sum_pj += p[k]
    
    return sum_pj



def get_approx_func(coeffs):
    coeffs = coeffs**2
    # coeffs = N*I0/(I0+(N-I0)/coeffs)

    def func(t):
        n=1
        
        while True:
            Ht = []
            Htt = []
            for i in range(0,n+1): 
                if i==0:
                    for tt in t:
                        Htt.append(solve_an(i)*solve_pj(i,tt))
                    sum_It_ISE = 2/(2*i+1)*solve_an(i)**2
                else:
                    for ttt in range(0,100):
                        Htt[ttt] += solve_an(i)*solve_pj(i,t[ttt])
                    sum_It_ISE += 2/(2*i+1)*solve_an(i)**2
            for k in range(0,100):
                Ht.append(N*I0/(I0+(N-I0)*r_beta/Htt[k]))
            # ISE = N*I0/(I0+(N-I0)/(et_integral-sum_It_ISE))
            ISE=et_integral-sum_It_ISE
            print(n)
            print(ISE)
            if ISE < coeffs:   # 没有完成解决误差问题。。。所以只能观察Legendre多项式逼近结果
                break

            # if n==10:
            #     break
            n += 1
        return Ht
    return func

if __name__ == "__main__":
    bound = input("Please input the error bound:")
    # bound = 100
    approx_func = get_approx_func(float(bound))

    f, (ax1, ax2) = plt.subplots(1, 2)
    t = np.arange(0, 10, 0.1)
    I_approx = approx_func(t)
    I = func(t)

    ax1.plot(t, I)
    ax2.plot(t, I_approx, c='r')
    plt.show()
    