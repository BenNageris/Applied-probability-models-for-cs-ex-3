import math
#n[t][k]-list[t] is a dictonary of frequency in document[t]. k is a word in document i.   
#P[i][k]-list[[]] 
def z(p_ti:list[dict],n_tk:list[dict],alpha:list,t:int,i:int)->float:
    sum=math.log(alpha[i])
    for k in n_tk[t].keys():
        ln_p=math.log(p_ti[i][k])
        ntk=n_tk[t][k]
        sum=sum+ln_p*ntk
    return sum

def max_z(p_ti:list[dict],n_tk:list[dict],alpha:list,t:int)
    max_z= float('-inf')
    for i in len(p_ti):
        max_z=max(max_z,Z(p_ti=p_ti,n_tk=n_tk,alpha=alpha,t=t,i=i))
    return max_z

def calc_ez(z:float,m:float,k:int)->float:
    if z[i]-m<-k:
        return 0
    return math.pow(math.e,z-m)     

def wti(z:list[float],i:int,m:float,k:int):
    if z[i]-m<-k:
        return 0
    numerator= calc_ez(z[i],m,k,i)
    denominator=sum([calc_ez(z[j],m,k) for j in range(len(z))])
    return numerator/denominator   