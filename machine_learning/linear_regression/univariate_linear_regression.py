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

def collect_example_dataset():
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

        
        
def linear_regression_coefficients(x,y):
    xbar,ybar = np.mean(x), np.mean(y)
    m = np.sum(np.dot((x-xbar),(y-ybar)))/np.sum((x-xbar)**2)
    b = ybar - m * xbar
    b_err, m_err = standard_error(x,xbar,y,m,b)
    return [m,m_err,b,b_err]


def standard_error(x,xbar,y,m,b):
    """Use this if either
    1. The errors in the regression are normally distributed
    2. The number of observations is sufficiently  large so that the estimator of slope coeff. is ~normally distributed (Central Limit theorem)

    param x       : independent variable valus
    param xbar    : mean of the independent variable
    param y       : dependent variable values
    param m       : Estimated gradient for line of best fit
    param b       : Estimated y-intercept for line of best fit
    """
    n = len(x)
    predicted_y = m*x + b
    sum_square_residuals =np.sum(np.square(y - predicted_y))
    sum_variance_x = np.sum(np.square(x-xbar))
    s_b = np.sqrt((1/(n-2))*sum_square_residuals/sum_variance_x) #standard error of slope. 
    s_a = s_b * np.sqrt((1/n)*np.sum(np.square(x)))
    return s_a, s_b
    



def plot(data,coeff, xlabel='CSGO rank', ylabel='ADR Rating'):
    """Extra plotting utility for this example
    :param data     : (See linear_regression fn)
    :param coeff    : [gradient, gradient error, y-intercept, y-intercept error]
    :return void
    """
    m,m_err,b,b_err = coeff
    x,y = data[:,0],data[:,1]
    plt.grid()
    plt.scatter(x,y,label='Raw data')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    xmin,xmax = [np.min(x),np.max(x)]
    xpoints = np.linspace(xmin,xmax, np.int64(xmax-xmin))
    predicted_y = xpoints*m + b
    plt.plot(xpoints, predicted_y, color='tab:orange', label=f'Line of best fit\n m={m:.5f} ± {m_err:.6f}\nb={b:.5f} ±{b_err:.6f}')
    plt.legend()
    plt.show()



def linear_regression(data=collect_example_dataset()):
    """
    :param data   : Nx2 or 2xN list / numpy array. the independent variable (x) should be first
                    e.g. [[x1,y1],
                          [x2,y2],
                          [x3,y3]]
                    or 
                          [[x1,x2,x3,..],
                           [y1,y2,y3]]
    
    :return       : array containing the estimated gradient and 
                    intercept of the line of best fit. The object 
                    is in the shape
                    [gradient, gradient_error, y-intercept, y-intercept error]

    """
    if type(data) != 'numpy.ndarray':
        data = np.array(data)
    if data.shape[0] > data.shape[1]:
        x,y = data.T
    else:
        x,y = data
    coeff = linear_regression_coefficients(x,y)
    plot(data,coeff)
    return linear_regression


linear_regression()

