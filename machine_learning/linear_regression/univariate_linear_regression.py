import matplotlib.pyplot as plt
import numpy as np
import requests
"""

This file, alongside multivariate_linear_regression is intended to supersede the currently 
applied linear_regression.py file, which should be deprecated. 

Also known as Simple linear regression, a lienar regression model with one independent variable 
and one dependent variable.

Things to investigate

1. What precision floating-point should we use? 
    a. Are there any benefits to using lower precision in terms of data length?

2. Verify on several test cases. 
    a. A straight + gradient line with no error
    b. a straight - gradient line with no error
    c. A controlled sample with a pre-defined average residual

3. Measure speed of regression technique against data size. 
"""

def collect_dataset():
    """
    Collect Data set of CSGO
    Data contains ADR vs Rating of a Player
    :return dataset obtained from link
    """
    response = requests.get(
        "https://raw.githubusercontent.com/yashLadha/The_Math_of_Intelligence/"
        "master/Week1/ADRvsRating.csv"
    )
    lines = response.text.splitlines()[1:]
    data = []
    for data_point in lines:
        data.append(data_point.split(','))
    return np.array(data).astype(np.float64)

        
        
def linear_regression_coefficients(data):
    x,y = data.T
    xbar,ybar = np.mean(x), np.mean(y)
    m = np.sum(np.dot((x-xbar),(y-ybar)))/np.sum((x-xbar)**2)
    b = ybar - m * xbar
    return [m,b]


def confidence_intervals(xbar,ybar,x,y,m,b):
    """
    Use this if either
    1. The errors in the regression are normally distributed
    2. The number of observations is sufficiently  large so that the estimator of slope coeff. is ~normally distributed (Central Limit theorem)
    """
    sum_square_residuals =np.sum(np.square( m*x+b - y))
    sum_variance_x = np.sum(np.square(x-xbar))
    s_b = np.sqrt((1/(len(x)-2))*sum_square_residuals/sum_variance_x) #standard error of slope. 




def plot(data,m,b):
    x,y = data[:,0],data[:,1]
    plt.grid()
    plt.scatter(x,y,label='Raw data')
    plt.xlabel('CSGO Rank')
    plt.ylabel('ADR')
    xmin,xmax = [np.min(x),np.max(x)]
    xpoints = np.linspace(xmin,xmax, np.int64(xmax-xmin))
    plt.plot(xpoints, xpoints*m + b, color='tab:orange', label='Line of best fit')
    plt.legend()
    plt.show()



def main(data=collect_dataset()):
    m,b = linear_regression_coefficients(data)
    return {
        "slope": m,
        "intercept": b

    }

if __name__ == '__main__':
    main()


