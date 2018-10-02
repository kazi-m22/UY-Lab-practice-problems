import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a - b* np.exp(c * x)


data = pd.read_csv('temperature.csv')
print(data.head())

x = data['day']
y = data.temp
x = np.array([x])
y = np.array([y])
x = np.transpose(x)
y = np.transpose(y)
print(x.shape)
print(y.shape)

initialGuess=[100, 100,-.01]
guessedFactors=[func(i,*initialGuess ) for i in x]
popt,pcov = curve_fit(func, x, y,initialGuess)
###making the actual fit
#one may want to

###preparing data for showing the fit
xont=np.linspace(min(x),max(x),50)
fittedData=[func(x, *popt) for x in xont]
###preparing the figure
fig1 = plt.figure(1)
ax=fig1.add_subplot(1,1,1)
###the three sets of data to plot
ax.plot(x,y,linestyle='',marker='o', color='r',label="data")
ax.plot(x,guessedFactors,linestyle='',marker='^', color='b',label="initial guess")
ax.plot(xont,fittedData,linestyle='-', color='#900000',label="fit with ({0:0.2g},{1:0.2g},{2:0.2g})".format(*popt))
###beautification
ax.legend(loc=0, title="graphs", fontsize=12)
ax.set_ylabel("factor")
ax.set_xlabel("x")
ax.grid()
ax.set_title("$\mathrm{curve}_\mathrm{fit}$")
###putting the covariance matrix nicely
tab= [['{:.2g}'.format(j) for j in i] for i in pcov]
the_table = plt.table(cellText=tab,
                  colWidths = [0.2]*3,
                  loc='upper right', bbox=[0.483, 0.35, 0.5, 0.25] )
plt.text(250,65,'covariance:',size=12)
###putting the plot
plt.show()

