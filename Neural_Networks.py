import numpy as np

print('''Softmax 
        When we want to compare 2 or more things and get a probability of each event, we use softmax method.
        In this method, we take the score of each of the event and perform the below
        exp(event)/sum(exp(all_events))
        this is chosen as exp, always returns +ve 
        Also sum of all the exp's will be 1 ''')

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    soft_max_val =[(np.exp(L[x]))/sum(np.exp(L)) for x in range(0,len(L))]
    return(soft_max_val)

print('''Maximum Likelihood''')

print(''' Good Model = High Probability 
          Good Model = Low Entropy 
          so for Model A,  4 events each have probability pa,pb,pc,pd. 
          Total Probability = pa*pb*pc*pd
          Another way to do this, 
          Cross Entropy = -ln(pa*pb*pc*pd)
                        = -ln(pa)+(-ln(pb))+....
                  
          ln(pa) is always negative since pa<1, Hence we take neg(ln(pa).
           Resulting value is cross entropy, which we need to minimize.
           Low Entropy = Good Model.''')

def cross_entropy(Y, P):
    #print(type(Y))
    #print(type(P))
    #print("Len is ",len(Y),len(P))
    ce=[]
    '''
    for i in range(0,len(Y)):
        ce_inst=0
        ce_sum = 0 
        ce_sum = Y[i]*np.log(P[i])
        ce_inst = float(sum(Y[i]*np.log(P[i]) + (1-Y[i])*np.log(1-P[i])))*(-1.0)
        ce.append(ce_inst)
    '''
    ce = [-float(Y[i]*np.log(P[i]) + (1-Y[i])*np.log(1-P[i])) for i in range(0,len(Y))]
    return(sum(ce))

def cross_entropy_n_classes():
    '''When you have more than 2 items that can occur, and each has their own probability, corss entropy is defined as
    sum(from 1-n of sum from 1-m of (yij*ln(Pij)))
    m is the number of discrete that can occur.
    n is the number of classes in which they can occur
    '''
    
y=[1,1,0,0]
p=[0.1,0.2,0.5,0.7]
x = cross_entropy(y,p)

