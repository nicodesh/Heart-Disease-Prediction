import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from descstats import MyPlot, Univa

import warnings
warnings.filterwarnings(action="ignore", module="sklearn", message="^internal gelsd")


###############################################################
# Linear Regression Class
###############################################################

class LinReg():
    """ A class to realize linear regressions. """

    def __init__(self, x, y):
        """ Class constructor.
        Args:
            x (Pandas Series): The first variable
            y (Pandas Series): The unique feature
        """

        self.x = x
        self.X = x[:, np.newaxis]
        self.y = y

        self.sklearn_lr = LinearRegression()
        self.sklearn_lr = self.sklearn_lr.fit(self.X, self.y)

        self.y_pred = self.sklearn_lr.predict(self.X)
        self.r2 = r2_score(self.y, self.y_pred)
        self.sklearn_coef = self.sklearn_lr.coef_
        self.sklearn_intercept = self.sklearn_lr.intercept_

        self.residuals = self.y - self.y_pred

        self.st_slope, self.st_intercept, self.st_rvalue, self.st_pvalue, self.st_stderr = st.linregress(self.x,self.y)

    def plot(self):
        """ Plot the scatterplot and the linear regression. """

        fig, ax = plt.subplots(figsize=[7,5])        
        MyPlot.scatter(ax, self.x, self.y)
        ax.plot(self.x, self.y_pred, linewidth=1, color="#fcc500")
        MyPlot.bg(ax)
        MyPlot.title(ax, "Scatterplot + Linear regression")
        MyPlot.border(ax)
        plt.show()

    def residuals_distribution(self):
        """ Plot the distribution of the residuals. """

        univ = Univa(self.residuals)
        univ.describe()
        univ.distribution(bins=9)


##################################################################
# Logistic Regression Class
##################################################################

class LogReg():
    """ A class to easily plot and compute simple logistic regressions. """
    
    def __init__(self, data, the_class=False, threshold=0.5):
        """ The constructor of the class
        Args:
            data (Pandas dataframe): The first columns should be the independant variables.
            The last one should be the X variable.
        """
        
        self.data = data.copy()
        self.the_class = the_class
        self.threshold = threshold
        
        # Compute the logistic regression
        self.compute_log_reg()
        
        # Apply the model on the dataset
        self.set_probabilites()
        
        # Apply the threshold decision
        self.apply_threshold(self.threshold)
            
        # Compute the KPIs
        self.compute_kpis()
    
        # Compute the class dataframe if it's a simple logistical regression
        self.compute_class_df()
            
    def compute_log_reg(self):
        """ Compute the logistical regression. """
        
        self.X = self.data.iloc[:,:-1].values
        self.X = sm.add_constant(self.X)
        self.y = self.data.iloc[:,-1]
        self.model = sm.Logit(self.y, self.X).fit(disp=False)      
            
    def set_probabilites(self):
        """ Apply the model. """
        
        self.probabilities = pd.DataFrame(self.X).apply(self.logistic_f, axis=1)
        
    def apply_threshold(self, threshold):
        """ Apply the specified threshold on data. """
        
        self.data['model'] = self.probabilities.apply(self.threshold_decision, args=(threshold,))
        
    def compute_kpis(self):
        """ Compute the KPI attached to the model. """
        
        self.true_pos = len(self.data[self.data.iloc[:,-2] == 1].query('model == 1'))
        self.true_neg = len(self.data[self.data.iloc[:,-2] == 0].query('model == 0'))
        self.false_pos = len(self.data[self.data.iloc[:,-2] == 0].query('model == 1'))
        self.false_neg = len(self.data[self.data.iloc[:,-2] == 1].query('model == 0'))

        self.pos_data = len(self.data[self.data.iloc[:,-2] == 1])
        self.neg_data = len(self.data[self.data.iloc[:,-2] == 0])

        self.pos_predict = len(self.data.query('model == 1'))
        self.neg_predict = len(self.data.query('model == 0'))

        self.total = len(self.data)

        self.success_rate = (self.true_pos + self.true_neg) / self.total
        self.se = self.true_pos / self.pos_data
        self.sp = self.true_neg / self.neg_data
        self.sp_inv = 1 - self.sp
        
    def compute_class_df(self):
        """ Compute the probability by class and create a dataframe. """
 
        if ((self.the_class) and (isinstance(self.the_class, int))):

            # Create the bins from the classes
            self.data['the_class'] = LogReg.create_the_class(self, self.data.iloc[:,0])
            
            # Compute the probability
            the_sum = self.data.iloc[:,1:].groupby('the_class').sum()
            the_count = self.data.iloc[:,1:].groupby('the_class').count()
            self.class_prob = (the_sum / the_count).reset_index()
            
            # Remove classes from the main dataframe
            self.data.drop('the_class', axis=1, inplace=True)
            
        else:
            self.class_prob = None
        
    def roc(self, tests=10):
        """ Compute and plot the ROC curve.
        Args:
            tests (int): number of tested threshold with value between 0 and 1.
            The bigger is this number the more accurate will be the curve.
        
        """
        
        roc_x = []
        roc_y = []
        
        # Compute 100 different threshold
        for threshold in np.flip(np.arange(0, 1.1, 1/tests), axis=0):
            self.apply_threshold(threshold)
            self.compute_kpis()
            roc_x.append(self.sp_inv)
            roc_y.append(self.se)
            
        # Reset on the default values
        self.apply_threshold(self.threshold)
        self.compute_kpis()
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(10,6))
        plt.plot(roc_x, roc_y)

        # Plot the bisectrix
        plt.plot([0, 1], [0, 1], linestyle='dashed', alpha=0.5)

        # Set the axe limits
        plt.axis((-0.005, 1, 0, 1.005))

        MyPlot.bg(ax)
        MyPlot.border(ax)
        MyPlot.title(ax, 'The ROC Curve')
        MyPlot.labels(ax, "1 - Specificity", "Sensibility")
        #plt.show()
        
    def plot(self):
        """ Plot the scatter plot and the probability by class. """
        
        if (len(self.model.params) == 2):
            class_width = self.the_class

            fig, ax = plt.subplots(figsize=(5,7))

            # Scatter plot
            plt.scatter(self.data.iloc[:,0], self.data.iloc[:,1], alpha=0.5, zorder=10)

            # Histogram
            if (self.class_prob is not None):
                plt.bar(
                    self.class_prob.iloc[:,0],
                    self.class_prob.iloc[:,1],
                    width=class_width,
                    color='orange',
                    )

            # Logistic Regression
            x = self.data.iloc[:,0]
            x_min = x.min()
            x_max = x.max()
            num = len(x)
            x = np.linspace(x_min, x_max, num=num)
            y = pd.DataFrame(sm.add_constant(x)).apply(self.logistic_f, axis=1)

            plt.plot(x, y, linewidth=2)

            # Theme
            MyPlot.bg(ax)
            MyPlot.border(ax)

            # Let's plot!
            plt.show()
            
        else:
            raise NameError('You can plot only with simple logistic regression.')
        
    def infos(self):
        """ Returns useful infos"""

        print(f"{len(self.model.params)} parameters (including the intercept)")
        ainfos = []
        for i, x in enumerate(self.model.params):
            ainfos.append({
                'name': f"B{i+1}",
                'Coeff': x,
                'OR': np.exp(x),
                'P-Value': self.model.pvalues[i]
                })

        dfinfos = pd.DataFrame(ainfos)
        dfinfos.set_index(dfinfos['name'], inplace=True)
        dfinfos.index.rename(None, inplace=True)
        dfinfos.drop("name", axis=1, inplace=True)
        display(dfinfos)

        data = {
            'y = 1':[self.true_pos, self.false_neg, self.pos_data],
            'y = 0':[self.false_pos, self.true_neg, self.neg_data],
            'Total':[self.pos_predict, self.neg_predict, self.total]
        }

        df = pd.DataFrame(data=data, columns=['y = 1', 'y = 0', 'Total'], index=['Predict 1', 'Predict 0', 'Total'])

        print("Matrice de confusion:")
        display(df)
        print("")
        print(f"Success rate: {self.success_rate:.2%}")
        print(f"Sensibility: {self.se:.2%}")
        print(f"Specificity: {self.sp:.2%}")
        
    def create_the_class(self, x, labels='num'):
        """ Create bins, based on the class specified by the user. """
        
        the_ceil = np.ceil(x / self.the_class) * self.the_class
        the_floor = the_ceil - self.the_class

        if (labels == 'str'):
            return f"{the_floor}-{the_ceil}"
        
        elif (labels == 'num'):
            return (the_ceil + the_floor) / 2
        
    def logistic_f(self, data):
        """ Logistic regression function. """
        
        logit = 0
        for i, x in enumerate(self.model.params):
            logit += data[i] * x
        
        num = np.exp(logit)
        den = 1 + np.exp(logit)
        
        return num / den
    
    def threshold_decision(self, x, threshold):
        """ Return 1 or 0, depending on the threshold fixed before. """
        
        if (x >= threshold):
            return 1
        
        else:
            return 0


class OWAnova():
    """ Copute ANOVA model. """

    def __init__(self, x, y):
        """ Constructor of the class.
        Args:
            x (Pandas Series): a qualitative variable.
            y (Pandas Series): a quantitative variable.
        """

        self.x = x.copy()
        self.y = y.copy()
        self.mean = y.mean()

        self.classes = []

        for i, the_class in enumerate(self.x.unique()):

            # Keep x var of the specific class
            yi_class = y[self.x == the_class]

            # First class is the Intercept. Its alpha is 0
            if (i == 0):
                self.mean0 = yi_class.mean()

            # Add all the class info in the classes array
            self.classes.append({
                                    'name': the_class,
                                    'ni': len(yi_class),
                                    'mean': yi_class.mean(),
                                    'alpha': yi_class.mean() - self.mean0
                                })

        # Somme des carrés totaux (SCT) / Total Sum of Squares (TSS)
        self.sct = sum([(yj - self.mean)**2 for yj in self.y])

        # Sommes des carrés expliqués (SCE) / Sum of Squares of the Model (SSM)
        self.sce = sum([c['ni'] * (c['mean'] - self.mean)**2 for c in self.classes])

        # Sommes des carrés résiduels (SCR) / Sum of Squares of the Error (SSE)
        self.scr = sum([sum([(yij - c['mean'])**2  for yij in self.y[self.x == c['name']]]) for c in self.classes])

        # Eta squared (Pourcentage de la variance expliquée par le modèle)
        self.eta_squared = self.sce / self.sct

        # Carré Moyen Expliqué (CME)
        self.cme = self.sce / (len(self.classes) - 1)

        # Carré Moyen Résiduel (CMR)
        self.cmr = self.scr / (len(self.x) - len(self.classes))

        # F-Stat + test de Fisher
        dofnum = len(self.classes) - 1
        dofden = len(self.x) - len(self.classes)
        self.fstat = self.cme / self.cmr
        self.fdistrib = st.f(dofnum, dofden)
        self.pvalue = self.fdistrib.sf(self.fstat)

        # F-Stat et test de Fisher avec scipy.stats
        samples = []
        for the_class in self.x.unique():
            samples.append(y[x.values == the_class])
            
        self.scipy = st.f_oneway(*samples)

    def summary(self):
        """ Display all informations in a Dataframe. """
        
        df = pd.DataFrame(self.classes, columns=['name', 'mean', 'alpha', 'ni'])
        df.set_index(df.name, inplace=True)
        df.drop('name', axis=1, inplace=True)
        df.index.rename(None, inplace=True)
        df = df[['mean', 'alpha', 'ni']]
        display(df)

        df = pd.DataFrame(data={
                            'Eta Squared':[self.eta_squared],
                            'F-Stat':[self.scipy.statistic],
                            'P-Value':[self.scipy.pvalue]
        })
        display(df)

class TWAnova():
    """ Perform two ways ANOVA with statsmodels. """

    def __init__(self, data, a, b, y):
        """ The instance constructor.
        Args:
            data (Pandas dataframe): The dataframe with all data.
            a (String): The name of the first explicative variable column.
            b (String): The name of the second explicative variable column.
            y (String): The name of the Y variable column.
        """

        self.data = data.copy()
        self.a = a
        self.b = b
        self.y = y

        formula = f"{self.y} ~ C({self.a}) + C({self.b}) + C({self.a}):C({self.b})"
        model = ols(formula, self.data).fit()
        self.aov_table = anova_lm(model, typ=2)

        # Add Eta squared in the table
        self.aov_table['eta_sq'] = 'NaN'
        self.aov_table['eta_sq'] = self.aov_table[:-1]['sum_sq']/sum(self.aov_table['sum_sq'])