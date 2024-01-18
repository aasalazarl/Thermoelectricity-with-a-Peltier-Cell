# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:12:41 2019

@author: alejandrosalazar

Based on 'weighted fit example.py' and by Mark Freeman

Performs weighted least squares fit for determination of Peltier coefficient
"""

# Bring in the graphing, data reading and transformation routines
import matplotlib.pyplot as plt 
import numpy as np

# Bring in the curve fitting routines
from scipy.optimize import curve_fit

# Define a function to which to fit
# ******************************************************************
def funcLinear(x,m,b):
     return m*x+b
# ******************************************************************

# Import the data files
data = np.loadtxt('seebeckCoeff_trial2_selectedData.csv',delimiter=',',skiprows=1)

# Parse into data strings
DelT = data[:,5];
x = DelT;
Vs = data[:,2]; #Q_Ip means (Qc-remove)/Ip
y = Vs;

uncDelT = data[:,11];
xunc = uncDelT;
uncVs = data[:,8];
yunc = uncVs;
print(x[0],y[0],xunc[0],yunc[0])

#####################################
# Perform linear fit (weighted least-squares fitting)

params = np.array([1,0]) # initial guess of slope and intercept

# Perform the linear fit
# FitResults = [slope,y-intercept]
fitResults,covariance = curve_fit(funcLinear,x,y,params,sigma=yunc,\
                                  absolute_sigma=True)

# Compute the fit uncertainties (Hughes&Ha, eqn. 7.22)
# fit_unc = 
fit_unc = np.sqrt(np.diag(covariance))

# Annotate plot with fit parameters
    # Correlation coefficient
corrcoeff = covariance[1,0]/((covariance[0,0]*covariance[1,1])**0.5)

xFit = x;
yFit = fitResults[0]*x+fitResults[1];
#print(yFit)

# Print results
print('*** Fit results ***')
print()
print('Covariance matrix','\n',covariance)
print()
print('fit slope, slope uncertainty = ',fitResults[0],',',fit_unc[0])
print()
print('fit intercept, intercept uncertainty = ',fitResults[1],',',fit_unc[1])
print()
print('correlation coefficient = ', corrcoeff)
print()

#####################################
# Chi-squared analysis

# Number of degrees of freedom: v = N - NN (Hughes&Hase, eqn.(8.1))
N = len(y);
NN = 2;
v = N-NN;

# The probability distribution function

#####
# Chi-squared chi2 = sum[(yi-y(xi))/alphai]^2, where alphai = errorBars (Hughes&Hase, eqn.(5.9))
chi2 = 0;
#print(yFit)
for i in range(len(y)):
    chi2 = chi2 + ((y[i] - yFit[i])**2)/yFit[i] # modified formula; not as in Hughes&Hase
    #print(chi2)
#print(chi2)
chi2Min = chi2; # Hughes&Hase, p.104
#print(chi2Min)

# P(chi2Min;v) = integral[X(chi2;v),{x,chi2Min,Infinity}] (Hughes&Hase, eqn.(8.4))
# X(chi2;v) = {[chi2^((v/2)-1))]*exp[-chi2/2]}/{[2^(v/2)]*T(v/2)}, where T = gammaFunction (Hughes&Hase, eqn.(8.3))
from scipy.integrate import quad
from scipy.special import gamma # for complete pacakge: from scipy.special import gamma, factorial
import numpy as np
import math

# Define X(chi2;v)
T = gamma(v/2)
#print(T)
def X(dummyChi2):
    return ((dummyChi2**((v/2)-1))*(np.exp(-dummyChi2/2)))/((2**(v/2))*T)

i, err = quad(X,chi2Min,math.inf)
#print(i,err)

#####
# Reduced chi-squared: chi2v =chi2Min/v
chi2v = chi2Min/v;
#print(chi2v)

# Print chi-squared analysis results
print('*** Chi-squared analysis results ***')
print()
print('Chi-squared:','P(chi2;v), err = ',i,',',err)
print()
print('Reduced chi-squared:','chi2v =',chi2v)
print()
print('Null hypothesis NOT rejected')

#################################
# Make the graph
    # Open figure
fig = plt.figure(1,figsize=(7,5),facecolor='w',edgecolor='k')
    # Clear figure
plt.clf()
    # Set up so fit line can be superposed after plotting the transformed data
plt.hold(True)

# Add data with error bars
a1 = plt.plot(0,0,0)
(a1,caps,_) = plt.errorbar(x,y,yunc,xunc,marker='.',markersize=1,linestyle='none',color='k',elinewidth=0.5)

# Add data without error bars
data_plot, = plt.plot(x,y,'ro',markersize=3,label=r'dataPoints',color='b')

# Add calculated data of the fit using a series of calculated distances
calcxrange = np.array([min(x)-1,max(x)+1])
fit_plot, = plt.plot(calcxrange,funcLinear(calcxrange,fitResults[0],fitResults[1]),\
                     'k-',label='linearFit')

# Annotate plot with fit parameters
    # Correlation coefficient
corrcoeff = covariance[1,0]/((covariance[0,0]*covariance[1,1])**0.5)

plt.hold(False)

# Labels
plt.ylabel("Voltage $V_{s}$ (volts)")
plt.xlabel('|${\Delta}T$| (kelvin)')
plt.legend(handles=[data_plot,fit_plot],loc='best',fontsize=10)
#plt.title('seebeckCoefficient $S$ from $V_{s}$ vs. ${\Delta}T$')

# Save figure(fig) = graph
#plt.savefig(name,format,dpi,bbox_inches)
#plt.savefig('seebeckCoeff_weightedLinFitNchiSqrd_trial2.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig('seebeckCoeff_weightedLinFitNchiSqrd_FINAL.png', format='png', dpi=300, bbox_inches='tight')

####################################