# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:47:35 2024

@author: Alibek
"""

import math
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import sympy as sp
import scipy.integrate as spi
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg

class SSI:
    def __init__(self):
        self.N = 0
        self.t = 0
        self.alpha = 0
        self.fx = ""
        self.precision = 100  # Precision setting
        getcontext().prec = self.precision
    
    def set_parameters(self, N, t, alpha, fx):
        self.N = N
        self.t = Decimal(str(t))
        self.alpha = Decimal(str(alpha))
        self.fx = fx
    
    def set_precision(self, precision):
        self.precision = precision
        getcontext().prec = self.precision
    
    def get_parameters(self):
        return self.N, self.t, self.alpha, self.fx
    
    

    def calculate_C0(self):
        
        try:
            N, t, alpha, fx = self.get_parameters()
            h = (Decimal('1') - t) / N
            k = 0
            ts = Decimal('0')
            while True:
                term = (Decimal('1') - (-1) ** k * Decimal(math.exp(2 * h))) * (h ** (alpha + k)) / \
                       (Decimal(math.factorial(k)) * (alpha + k))
                if abs(term) < 1e-100:  # Check if term is negligible
                    break
                ts += term
                k += 1

            denominator = Decimal('1') - Decimal(math.exp(2 * h))
            C_0 = ts / denominator
            return C_0  
        except Exception as e:
            messagebox.showerror("Error", f"1-An error occurred: {str(e)}")

    def calculate_C1(self):
        try:
            N, t, alpha, fx = self.get_parameters()
            h = (Decimal('1') - t) / N
            k = 0
            ts = Decimal('0')
            while True:
                term = (((Decimal(math.exp(-h)) - (-1) ** k * Decimal(math.exp(3 * h))) * (2 * h) ** (alpha + k) + 
                         (1 + Decimal(math.exp(2 * h))) * (-Decimal(math.exp(-h)) + (-1) ** k * Decimal(math.exp(h))) * h ** (alpha + k)) / 
                        (Decimal(math.factorial(k)) * (alpha + k)))
                if abs(term) < 1e-100:  # Check if term is negligible
                    break
                ts += term
                k += 1
            denominator = Decimal('1') - Decimal(math.exp(2 * h))
            C_1 = ts / denominator
            return C_1
        except Exception as e:
            messagebox.showerror("Error", f"2-An error occurred: {str(e)}")

    def calculate_CB(self, beta):
        try:
            N, t, alpha, fx = self.get_parameters()
            h = (Decimal('1') - t) / N
            k = 0
            ts = Decimal('0')
            while True:
                term = (((Decimal(math.exp(-h * beta)) - (-1) ** k * Decimal(math.exp(beta * h + 2 * h))) * (beta * h + h) ** (alpha + k) + 
                         (Decimal(math.exp(-beta * h + 2 * h)) - (-1) ** k * Decimal(math.exp(h * beta))) * (beta * h - h) ** (alpha + k) + 
                         (1 + Decimal(math.exp(2 * h))) * (-Decimal(math.exp(-h * beta)) + (-1) ** k * Decimal(math.exp(h * beta))) * (h * beta) ** (alpha + k)) / 
                        (Decimal(math.factorial(k)) * (alpha + k)))
                if abs(term) < 1e-100:  # Check if term is negligible
                    break
                ts += term
                k += 1

            denominator = Decimal('1') - Decimal(math.exp(2 * h))
            C_beta = ts / denominator
            return C_beta
        except Exception as e:
            messagebox.showerror("Error", f"3-An error occurred: {str(e)}")

    def calculate_CN(self):
        try:
            N, t, alpha, fx = self.get_parameters()
            h = (Decimal('1') - t) / N
            k = 0
            ts = Decimal('0')
            while True:
                term = (((1 - t - h) ** (alpha + k) - (1 - t) ** (alpha + k)) * 
                        (Decimal(math.exp(t - 1)) * Decimal(math.exp(2 * h)) - (-1) ** k * Decimal(math.exp(1 - t))) / 
                        (Decimal(math.factorial(k)) * (alpha + k)))
                if abs(term) < 1e-100:  # Check if term is negligible
                    break
                ts += term
                k += 1

            denominator = Decimal('1') - Decimal(math.exp(2 * h))
            C_N = ts / denominator
            return C_N
        except Exception as e:
            messagebox.showerror("Error", f"4-An error occurred: {str(e)}")
    
    def compute_F(self):
        
        try:
            N, t, alpha, fx = self.get_parameters()
            #h = (Decimal('1') - t) / N

            fx_sympy = sp.sympify(fx)
            def integrand(x):
                return fx_sympy.subs(sp.Symbol('x'), x) / (Decimal(x) - t) ** (Decimal('1') - alpha)
            integral, _ = spi.quad(integrand, Decimal(str(t)), Decimal('1'))
            
            return Decimal(integral)
            
        except Exception as e:
            messagebox.showerror("Error", f"6-An error occurred: {str(e)}")
            return None  # Return None if an error occurs
            
    def compute_SSS_FF(self):
        try:
            result = self.computeSSS() - self.compute_F()
            # No need to round here
            return Decimal(result)

        except Exception as e:
            messagebox.showerror("Error", f"7-An error occurred: {str(e)}")
            return None  # Return None if an error occurs
        
    def computeSSS(self):
        
        try:
            N, t, alpha, fx = self.get_parameters()
            h = (Decimal('1') - t) / N

            
            SSS = (Decimal(str(self.calculate_C0())) * sp.sympify(fx).subs(sp.Symbol('x'),t) +
                   Decimal(str(self.calculate_C1())) * sp.sympify(fx).subs(sp.Symbol('x'), t + h) +
                   Decimal(str(self.calculate_CN())) * sp.sympify(fx).subs(sp.Symbol('x'), Decimal('1')))
            
            for i in range(2, N):
                C_beta = self.calculate_CB(i)
                SSS += Decimal(str(C_beta)) * sp.sympify(fx).subs(sp.Symbol('x'), h * i + t)
            return Decimal(str(SSS))
        
        except Exception as e:
            messagebox.showerror("Error", f"8-An error occurred: {str(e)}")
            return None  # Return None if an error occurs
        
    def compute_F_for_t(self, t_val, fx, alpha):
        
        try:
            N, t, alpha, fx = self.get_parameters()
            fx_sympy = sp.sympify(fx)
            def integrand(x):
                return fx_sympy.subs(sp.Symbol('x'), x) / (Decimal(x) - Decimal(t_val)) ** (Decimal('1') - alpha)
            integral, _ = spi.quad(integrand, Decimal(str(t_val)), Decimal('1'))
            
            return Decimal(integral)
        except Exception as e:
            messagebox.showerror("Error", f"9-An error occurred: {str(e)}")
            return None 
            
    def compute_SSS_t(self, t_val, fx, alpha): 
         try:
             N, t, alpha, fx = self.get_parameters()
             h = (Decimal('1') - Decimal(t_val)) / N

             
             SSS = (Decimal(str(self.calculate_C0())) * sp.sympify(fx).subs(sp.Symbol('x'),Decimal(t_val)) +
                    Decimal(str(self.calculate_C1())) * sp.sympify(fx).subs(sp.Symbol('x'), Decimal(t_val) + h) +
                    Decimal(str(self.calculate_CN())) * sp.sympify(fx).subs(sp.Symbol('x'), Decimal('1')))
             
             for i in range(2, N):
                 C_beta = self.calculate_CB(i)
                 SSS += Decimal(str(C_beta)) * sp.sympify(fx).subs(sp.Symbol('x'), h * i + Decimal(t_val))
             return Decimal(str(SSS))
         
         except Exception as e:
             messagebox.showerror("Error", f"10-An error occurred: {str(e)}")
             return None
         
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("OQF GUI")
        
        self.N = tk.StringVar()
        self.t = tk.StringVar()
        self.alpha = tk.StringVar()
        self.fx = tk.StringVar()
        
        self.figure = plt.figure(figsize=(6, 4))
        self.canvas = tkagg.FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        

        input_frame = ttk.Frame(self.root)
        input_frame.pack(padx=10, pady=10)

        ttk.Label(input_frame, text="N:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        ttk.Entry(input_frame, textvariable=self.N).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="t:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        ttk.Entry(input_frame, textvariable=self.t).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="alpha:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        ttk.Entry(input_frame, textvariable=self.alpha).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="fx:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        ttk.Entry(input_frame, textvariable=self.fx).grid(row=3, column=1, padx=5, pady=5)

        ttk.Button(input_frame, text="Compute", command=self.compute).grid(row=4, columnspan=2, padx=5, pady=5)

        self.results_frame = ttk.Frame(self.root)
        self.results_frame.pack(padx=10, pady=10)

    def compute(self):
        try:
            
            N = int(self.N.get())
            t = Decimal(self.t.get())
            alpha = Decimal(self.alpha.get())
            fx = self.fx.get()

            qf_ff = SSI()
            qf_ff.set_parameters(N, t, alpha, fx)
            
            FF=qf_ff.compute_F()
            SSS=qf_ff.computeSSS()
            contrast=qf_ff.compute_SSS_FF()
            
            t_values = np.linspace(0, 0.99, 100)
            difference_values=[]
            
            for i in  t_values:
                difference_values.append(abs(qf_ff.compute_F_for_t(i, fx, alpha)-qf_ff.compute_SSS_t(i, fx, alpha)))
                
            
            
            #FF_t=qf_ff.compute_F_for_t(t_val, fx, alpha)
            #SSS_t=qf_ff.compute_SSS_t(t_val, fx, alpha)
            #results = qf_ff.compute_results()

            if [FF,SSS,contrast] is not None:  # Check if results are valid
                self.display_results(FF,SSS,contrast,difference_values)
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid values for N, t, and alpha.")

    def display_results(self,FF,SSS,contrast,dv):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        
        ttk.Label(self.results_frame, text="Results:").pack()
        #ttk.Label(self.results_frame, text=self.calculate_C0()).pack()
        #ttk.Label(self.results_frame, text=self.calculate_C1()).pack()
        #ttk.Label(self.results_frame, text=self.calculate_CN()).pack()
        ttk.Label(self.results_frame, text=FF).pack()
        ttk.Label(self.results_frame, text=SSS).pack()
        ttk.Label(self.results_frame, text=contrast).pack()
        
        
        # Extracting t values and differences from results
          # Adjust the range as needed
        t_values = np.linspace( 0,0.99, 100)
        # Plotting the differences
        # Plotting the differences
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(t_values, dv)
        ax.set_xlabel('t')
        ax.set_ylabel('Difference')
        ax.set_title('Difference Plot')
        ax.grid(True)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
