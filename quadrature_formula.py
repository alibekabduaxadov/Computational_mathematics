# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:28:19 2024

@author: user
"""
import numpy as np
import sympy as sp
#from sympy import symbols, diff, sqrt, integrate
from tabulate import tabulate
import matplotlib.pyplot as plt

class QF_FF:
    def __init__(self,N,T,sig):
        self.N=N
        self.T=T
        self.sig=sig
        self.A=np.empty(N+1) 
        self.F=np.empty(T)
        self.FI=np.empty(T)
        self.x=np.empty(N+1)
        self.Q=[]
        #self.Nu=[]
        self.Qu=[]
        self.No=[]
        self.h=1/N
        self.f_s = "exp(x)"
        self.f = sp.Symbol('x')
        self.math_results = [["T/r","Sig","I-Q", "||f||","||Ф||","||f||*||Ф||"]]
   # def set_N(self,N):
        #  self.N=N
         #  print("Tugun nuqtalar sonini kiriting !")
         #  self.N=int(input('N='))
    def get_N(self):
            return self.N
   # def set_T(self,T):
        #    print('Chiqarish kerak bo\'lgan jadvalli qiymatlar sonini kiriting')
        #    self.T=int(input("T="))
    def get_T(self):
            return self.T
    def get_sig(self):
        return self.sig
    def get_x(self):
        for k in range(0,self.N+1):
            self.x[k]=self.h*k
        return self.x
    def get_f_s(self):
        return self.f_s
    def get_f(self):
        return self.f
    # Foydalanuvchi Kiritgan (f(x)) string turidagi ifodani sympy ifodasiga aylantirish
    def get_integrate_viev(self):
        return sp.sympify(self.f_s)
    #Kiritilgan funksiyaning (0,1) oraliqda aniq integralini hisoblash 
    def set_integrate_result(self):
        return  sp.integrate(self.get_integrate_viev(), (self.f, 0,1))
    # bu takrorlash T qadam takrorlanadi va T xil sigma uchun A[k] va Q[i] larni hisoblaydi
    #Q[i]=A[k]*f[x_k] lar
    #A[k] koeffisientlar
    def set_koeffisient(self):
        self.get_x()
        for i in range(0,self.T):
            self.A[0]=(np.exp(i*self.x[1])-np.exp(i*self.x[0]))/(i*(np.exp(i*self.x[1])+np.exp(i*self.x[0])))
            for k in range(1,self.N,1):
                self.A[k]= 2*np.exp(i*self.x[k])*(np.exp(i*self.x[k+1])-np.exp(i*self.x[k-1]))/(i*(np.exp(i*self.x[k+1])+np.exp(i*self.x[k]))*(np.exp(i*self.x[k])+np.exp(i*self.x[k-1])))
        
            self.A[self.N]=(np.exp(i*self.x[self.N])-np.exp(i*self.x[self.N-1]))/(i*(np.exp(i*self.x[self.N])+np.exp(i*self.x[self.N-1])))
            self.Q.append([self.A[k]*sp.sympify(self.f_s).subs(self.f,self.x[k]) for k in range(0,self.N+1)])
    def set_Q(self): 
        return [sum(self.Q[i]) for i in range(0,self.T)]
    #def set_I_Q(self):
       # self.Qu.append([abs(self.set_integrate_result()-sum(self.Q[i])) for i in range(0,self.T)])
    #||f(x)|| ning T ta sigma qiymati uchun  normasi
    def set_Norm_fx(self):
        for i in range(0,self.T):
            #F[i]=sp.sqrt(4/3+i+((i)**2)/5)
            self.F[i]=sp.sqrt(sp.integrate((sp.diff(self.get_integrate_viev(),self.f)+i*self.get_integrate_viev())**2, (self.f, 0,1)))
    #||Ф||  norma T ta sigma qiymati uchun
    def set_Norm_Ф(self):
        for i in range(0,self.T):
            self.FI[i] = (1 / np.power(i, 2) - (2 * (1 - np.exp(-i * self.h))) / (np.power(i, 3) * self.h * (1 + np.exp(-i * self.h)))) ** (1 / 2)
    # qiymatlarni jadvalli ko'rinishida chiqarish
    # "T/r","Sig","I-Q", "||f||","||Ф||","||f||*||Ф||" uchun T ta qiymat hosil bo'ladi   
    def set_tabulate(self):
        self.set_Norm_fx()
        self.set_Norm_Ф()
        self.set_koeffisient()
        for i in range(0,self.T):
            self.math_results.append([i,i,abs(self.set_integrate_result()-sum(self.Q[i])),self.F[i],self.FI[i],self.F[i]*self.FI[i] ])
        print(tabulate(self.math_results, headers='firstrow', tablefmt='grid'))
    # visualizatsiya integralni taqribiy vahaqiqiy qiymatlari ayirmasining sigmaga bog'liqligi
    def plotting(self):
        self.set_tabulate()
        plt.figure(figsize=(16,8))
        plt.plot(np.arange(0,self.T),[abs(self.set_integrate_result()-sum(self.Q[i])) for i in range(0,self.T)], linestyle="-")
        plt.xlabel('Sigma')
        plt.ylabel('|Q-I|')
        plt.title('Absalyut xatolikni sigmaga bog\'liqlik grafigi')
        plt.grid(True)
        plt.legend()
        plt.show()   
    
    ############################################
    
