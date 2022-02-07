import pandas as pd
import numpy as np
from sklearn import linear_model
import math
import pickle
import matplotlib.pyplot as plt
%matplotlib inline

def fx_gradient_decent(x,y):
    b_curr=m_curr=0
    iteration=100000
    n=len(x)
    learning_rate=0.0002
    prev_cost=0
    
    for i in range(iteration):
        
        yp = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - yp)])
        
        md = -(2/n) * sum(x *(y - yp))
        bd = -(2/n) * sum(y-yp)
        
        m_curr = m_curr - (learning_rate * md)
        b_curr = b_curr - (learning_rate * bd)
        
        print("m: {}, b: {} , cost{} iterations {}".format(m_curr,b_curr,cost,i))
        if math.isclose(prev_cost,cost,rel_tol=1e-20,abs_tol=0):
            print("OUTTT m: {},b:{},cost{} iterations{}".format(m_curr,b_curr,cost,i))
        prev_cost=cost
        
        
df=pd.read_excel("/Users/fafakhan/Downloads/per_capita_income.xlsx",sheet_name='gradient_descent_cost_function')
fx_gradient_decent(df.math,df.cs)
