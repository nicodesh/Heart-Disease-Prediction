import math
import scipy.stats as st
import statsmodels.api as sm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from sklearn import decomposition
from sklearn import preprocessing
from sklearn.cluster import KMeans

from IPython.display import display

##############################################################################
# Boxplot Styles Class
##############################################################################

class BoxPlotStyle:
    """ Just a class with some boxplot properties """

    medianprops = {
        'color': '#fcc500',
        'linewidth': 2
    }
        
    meanprops = {
        'marker': 'o',
        'markersize': 8,
        'markeredgecolor': '#d95040',
        'markerfacecolor': '#d95040'
    }

    boxprops = {
        'color': '#3977af',
        'facecolor': '#3977af'
    }

    whiskerprops = {
        'color': '#999999',
        'linewidth': 2
    }

    capprops = {
        'color': '#999999',
        'linewidth': 2
    }


##############################################################################
# MyPlot Class
##############################################################################

class MyPlot():
    """ A list of methods to compute and plot beautiful plots with matplotlib.pyplot. """
    
    def new_plot(width=8, heigth=5, plots=1):
        """ Return one or two empty axe(s). """

        fig = plt.subplots(figsize=[width, heigth])

        if (plots == 1):
            ax = plt.subplot(111)
            return fig, ax

        elif (plots == 2):
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)
            return fig, ax1, ax2
    
    def plot_show():
        """ Plot the plots. """
        
        plt.show()
        
    # Annotations
    def annotate(ax, text, xy, xytext):
        """ Set a fancy annotation. """

        ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            arrowprops={'color': '#d95040','arrowstyle':'simple'}
        )

    #Background
    def bg(ax):
        """ Set a nice background with grid and stuff. """

        ax.set_facecolor('#ebe9f2')
        ax.grid(color='#FFFFFF', linestyle='-')

    # Title
    def title(ax, title, y=1.02):
        """ Set a fancy title. """

        fontsize = 15
        color = '#555555'
        fontweight = 'bold'

        ax.set_title(title, fontsize=fontsize, y=y, color=color, fontweight=fontweight)

    # Border
    def border(ax):
        """ Set nice graph borders. """

        for spine in ax.spines.values():
            spine.set_edgecolor('#c6c1d8')
            spine.set_linewidth(1)

    # Axe Labels
    def labels(ax, x, y):
        """ Set nice axe labels """

        ax.set_xlabel(x)
        ax.set_ylabel(y)

    # Scatterplot
    def scatter(ax, x, y):
        """ Set a wonderful scatter plot. """
        ax.plot(x, y, 'o', alpha=0.5, color="#5086ec")
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)

    # Pie Chart
    def piechart(ax, data, var_name='Legend', unit=''):
        """ Plot a fancy Pie Chart (Donnut Style)

        Based on this code:
        https://matplotlib.org/gallery/pie_and_polar_charts/pie_and_donut_labels.html

        """

        ax.set_aspect('equal')

        the_labels = data.value_counts().index
        the_data = data.value_counts().values

        def func(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return f"{pct:.1f}%\n({absolute:d}{unit})"


        wedges, texts, autotexts = ax.pie(
                                    the_data,
                                    wedgeprops=dict(width=0.7),
                                    autopct=lambda pct: func(pct, the_data),
                                    textprops=dict(color="w"))

        ax.legend(wedges, the_labels,
                  title=var_name,
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=14)

        plt.setp(autotexts, size=8, weight="bold")

    # Linear regression
    def lin_reg(ax, x, y):
        """ Plot a linear regression. """

        Y = y.copy()
        X = pd.DataFrame()
        X['data'] = x.copy()
        X = X[['data']]
        X['intercept'] = 1
        result = sm.OLS(Y, X).fit()
        a, b = result.params['data'], result.params['intercept']

        xs = np.arange(min(x), max(x))
        ax.plot(xs, [a*x+b for x in xs], linewidth=4, color="#fcc500")
        ax.set_ylim(min(y), max(y))

    # Boxplot
    def boxplot(ax, data):
        """ Plot a cool boxplot """

        ax.boxplot(
            data,
            showfliers=False,
            vert=False,
            zorder=2,
            patch_artist=True,
            medianprops=dict(BoxPlotStyle.medianprops),
            meanprops=dict(BoxPlotStyle.meanprops),
            boxprops=dict(BoxPlotStyle.boxprops),
            whiskerprops=dict(BoxPlotStyle.whiskerprops),
            capprops=dict(BoxPlotStyle.capprops)
        )

        # Set background
        MyPlot.bg(ax)

        # Set borders
        MyPlot.border(ax)

        # Set xtick labels
        MyPlot.labels(ax, data.name, "Boxplot")

    # Bivariate boxplots
    def boxplots(ax, x, y, class_size=50):
        """ Plot boxplots by bin. """

        groups = []
        edges = np.arange(0, max(x), class_size) # Create the bins
        edges += class_size # Increase each bin by a class size
        indexes = np.digitize(x, edges) # Associate each value to a bin

        # Create groups by class
        for ind, edge in enumerate(edges):
            values = y.loc[indexes == ind]

            if len(values) > 0:
                group = {
                    'values': values,
                    'class_center': edge - (class_size / 2),
                    'size': len(values),
                    'quartiles': [np.percentile(values, p) for p in [25,50,75]]
                }
                groups.append(group)

        # Boxplots
        ax.boxplot(
            [g["values"] for g in groups],
            positions=[g["class_center"] for g in groups],
            widths=class_size*0.7,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(BoxPlotStyle.medianprops),
            meanprops=dict(BoxPlotStyle.meanprops),
            boxprops=dict(BoxPlotStyle.boxprops),
            whiskerprops=dict(BoxPlotStyle.whiskerprops),
            capprops=dict(BoxPlotStyle.capprops)
        )

        # Set xlabel and ylabel
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)

        # Set correct axes values
        ax.set_xlim(0, max(edges))
        ax.set_ylim(0)

        # Classes size
        for g in groups:
            ax.text(g['class_center'],
                    0, f"(n={g['size']})",
                    horizontalalignment='center',
                    verticalalignment='top'
                   )

        # Put space between xtick labels and the axe
        ax.tick_params(axis='x', which='major', pad=10)

    # ANOVA
    def anova(ax, x, y):
        """ Plot boxplots by modality. """

        mods = x.drop_duplicates().sort_values()
        groups = []

        for m in mods:
            groups.append(y[x == m])


        ax.boxplot(groups,
                   labels=mods,
                   showfliers=False,
                   vert=False,
                   patch_artist=True,
                   showmeans=True,
                   medianprops=dict(BoxPlotStyle.medianprops),
                   meanprops=dict(BoxPlotStyle.meanprops),
                   boxprops=dict(BoxPlotStyle.boxprops),
                   whiskerprops=dict(BoxPlotStyle.whiskerprops),
                   capprops=dict(BoxPlotStyle.capprops)
                  )

    # Histogram
    def hist(ax, data, bins, color='#3977af'):
        """ Plot a pretty histogram """

        histtype = 'bar'
        color = color
        edgecolor = color
        ax.hist(data, bins=bins, histtype=histtype, color=color, edgecolor=color, alpha=0.5, zorder=2)

        # Set background
        MyPlot.bg(ax)

        # Set borders
        MyPlot.border(ax)

        # Set labels
        MyPlot.labels(ax, data.name, "Quantity")

    # Bar Chart
    def bar(ax, data=None, step=None, x=None, y=None):
        """ Plot a rock bar """
        
        if isinstance(data, pd.Series):
            x = data.value_counts().index
            y = data.value_counts().values
            data_name_x = data.name
            data_name_y = "Quantity"
            
        else:
            data_name_x = x.name
            data_name_y = y.name

        ax.bar(x,y, tick_label=x, zorder=2)

        if step:
            new_a = []
            i = 0
            for val in x.sort_values():
                if i == 0:
                    new_a.append(val)

                if i == (step-1):
                    i = 0

                else:
                    i += 1

            ax.set_xticklabels(new_a)
            ax.set_xticks(new_a)

        # Set background
        MyPlot.bg(ax)

        # Set borders
        MyPlot.border(ax)

        # Set labels
        MyPlot.labels(ax, data_name_x, data_name_y)

    # Lorenz Curve
    def lorenz(ax, data):
        """ Plot a Lorenz Curve """

        # Order the data
        l_data = np.sort(data)

        # Y axe: Cumsum, from 0 to 1
        l_y = np.cumsum(l_data) / l_data.sum()
        l_y = np.append([0], l_y)

        # X axe: Linspace, from 0 to 1
        l_x = np.linspace(0, 1, len(l_y))

        ax.fill_between(l_x, l_y, zorder=2)
        ax.fill_between(l_x, l_x, l_y, zorder=2, color='#3977af', alpha=0.5)

        median = l_y[int(round(len(l_y)) / 2)]
        ax.plot([0.5, 0.5], [0, median], color='#fcc500', linestyle=':', linewidth=2, label="Median")
        ax.plot([0, 0.5], [median, median], color='#fcc500', linestyle=':', linewidth=2)

        medial = len(l_y[l_y <= 0.5]) / len(l_y)
        ax.plot([0, medial], [0.5, 0.5], color='#f70028', linestyle=':', linewidth=2, label="Medial")
        ax.plot([medial, medial], [0, 0.5], color='#f70028', linestyle=':', linewidth=2)

        ax.plot([0, 1], [0, 1], color='#999999', linestyle='-', linewidth=1)
        ax.plot([0, 1], [0, 0], color='#999999', linestyle='-', linewidth=1)
        ax.plot([1, 1], [0, 1], color='#999999', linestyle='-', linewidth=1)

        # Set legend
        ax.legend()

        # Set borders
        MyPlot.border(ax)

        # Set labels
        MyPlot.labels(ax, "Products", "Wealth")

        # Set background
        MyPlot.bg(ax)

    def pca_screeplot(ax, pca):
        """ Plot a scree plot of eigenvalues from a sklearn PCA object. """

        scree = pca.explained_variance_ratio_*100
        ax.bar(np.arange(len(scree))+1, scree, zorder=2)
        ax.plot(
                np.arange(len(scree))+1,
                scree.cumsum(),
                c="#FFB246",
                marker='o',
                markersize=12
                )

        # Set background
        MyPlot.bg(ax)

        # Set borders
        MyPlot.border(ax)

        # Set labels
        MyPlot.labels(ax, "Axes of inertia", "Percentage of inertia")

        # Set title
        MyPlot.title(ax, "Scree Plot of Eigenvalues")

        # Set background
        MyPlot.bg(ax)

    def pca_circle(fig, ax, pca, components, labels=None, label_rotation=0, lims=None):
        """ Plot the correlation circle from a sklearn PCA obejct. """

        d1 = components[0] - 1
        d2 = components[1] - 1
        pcs = pca.components_

        # détermination des limites du graphique
        if lims is not None :
            xmin, xmax, ymin, ymax = lims
        elif pcs.shape[1] < 30 :
            xmin, xmax, ymin, ymax = -1, 1, -1, 1
        else :
            xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

        # affichage des flèches
        # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
        if pcs.shape[1] < 30 :
            plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
               pcs[d1,:], pcs[d2,:], 
               angles='xy', scale_units='xy', scale=1, color="#FFB246")

        else:
            lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
            ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='#FFB246'))
        
        # affichage des noms des variables  
        if labels is not None:  
            for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                    plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="#3977af", alpha=0.5)
    
        # affichage du cercle
        circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='#3977af', zorder=3)

        plt.gca().add_artist(circle)

        # définition des limites du graphique
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # affichage des lignes horizontales et verticales
        plt.plot([-1, 1], [0, 0], color='#c6c1d8', ls='--')
        plt.plot([0, 0], [-1, 1], color='#c6c1d8', ls='--')

        # nom des axes, avec le pourcentage d'inertie expliqué
        xlabel = f"F{d1+1} ({pca.explained_variance_ratio_[d1]:.0%})"
        ylabel = f"F{d2+1} ({pca.explained_variance_ratio_[d2]:.0%})"

        # Set labels
        MyPlot.labels(ax, xlabel, ylabel)

        # Set title
        MyPlot.title(ax, f"Correlation Circle on F{d1+1} and F{d2+1}")

        # Set border
        MyPlot.border(ax)

    def pca_factorial_plane(fig, ax, X_projected, pca, components, labels=None, alpha=1, clusters=None, lims=None, **kwargs):
        """ Plot a scatterplot from data projected on the choosen factorial plane. """

        d1 = components[0] - 1
        d2 = components[1] - 1

        # Markersize
        if ('markersize' in kwargs):
            markersize = kwargs['markersize']
        else:
            markersize = 50
    
        # Scatterplot
        if clusters is not None:
            clusters = np.array(clusters)
            for value in np.unique(clusters):
                selected = np.where(clusters == value)
                plt.scatter(
                            X_projected[selected, d1],
                            X_projected[selected, d2],
                            alpha=alpha,
                            label=value,
                            s=markersize
                            )
            plt.legend()

        else:
            plt.scatter(
                        X_projected[:, d1],
                        X_projected[:, d2],
                        alpha=alpha,
                        s=markersize
                        )



        # Labels
        if labels is not None:
            for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                plt.text(x, y, labels[i],
                          fontsize='14', ha='center',va='center') 
            
        # Plot limits
        if lims == None:
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])

        else:
            plt.xlim([lims[0], lims[1]])
            plt.ylim([lims[2], lims[3]])
    
        # Vertical and horizontal lines
        plt.plot([-100, 100], [0, 0], color='#c6c1d8', ls='--')
        plt.plot([0, 0], [-100, 100], color='#c6c1d8', ls='--')

        # Axis name with percentage explained by the component.
        xlabel = f"F{d1+1} ({pca.explained_variance_ratio_[d1]:.0%})"
        ylabel = f"F{d2+1} ({pca.explained_variance_ratio_[d2]:.0%})"

        # Set labels
        MyPlot.labels(ax, xlabel, ylabel)

        # Set title
        MyPlot.title(ax, f"Projection on F{d1+1} and F{d2+1}")

        # Set border
        MyPlot.border(ax)

    def dendrogram(Z, names, figsize=(10,25), save_image=False):
        """ Plot a dendogram from hierarchical linked data, from sklearn. """

        fig = plt.figure(figsize=figsize)
        ax = plt.subplot()
        dendrogram(
            Z,
            labels = names,
            orientation = "left",
        )
        MyPlot.title(ax, "Hierarchical Clustering Dendrogram")
        MyPlot.labels(ax, "Distance", '')

        if save_image:
            fig.savefig(save_image + '.png')
            
        MyPlot.plot_show()

##############################################################################
# Hierarchical Linked Class
##############################################################################

class MyHier():
    """ A class to compute and plot Hiearchical Linked Data. """

    def __init__(self, data):
        """ Class constructor.
        Args:
            data (Pandas DataFrame): the data used to compute the dendrogram.
        """

        # Save data
        self.data = data.copy()
        self.X = self.data.values
        self.names = self.data.index

        # Scale data
        self.X_scaled = preprocessing.StandardScaler().fit_transform(self.X)

        # Compute the hierarchical clustering with the ward method.
        self.Z = linkage(self.X_scaled, 'ward')

    def dendrogram(self, **kwargs):
        """ Plot the Dendrogram. """
        MyPlot.dendrogram(self.Z, self.names, **kwargs)

    def clusters(self, nb_cluster):
        """ Save and return the clusters. """
        
        # Save the clusters
        self.clusters = fcluster(self.Z, nb_cluster, criterion='maxclust')

        # Compute and save the centroïds
        centroids_scaled = np.ndarray(shape=(0, self.X.shape[1]))
        centroids_raw = np.ndarray(shape=(0, self.X.shape[1]))
        centroids_names = np.array([]).astype(int)

        for cluster in np.unique(self.clusters):
            new_row_raw = self.X[np.where(self.clusters == cluster)].mean(axis=0)
            new_row_scaled = self.X_scaled[np.where(self.clusters == cluster)].mean(axis=0)

            centroids_raw = np.vstack((centroids_raw, new_row_raw))
            centroids_scaled = np.vstack((centroids_scaled, new_row_scaled))
            centroids_names = np.append(centroids_names, cluster)

        self.centroids_raw = centroids_raw
        self.centroids_scaled = centroids_scaled
        self.centroids_names = centroids_names

        return self.clusters

    def centroids(self, scaled=True):
        """ Returns the centroids. Either scaled or not scaled. """

        if scaled:
            return self.centroids_scaled

        else:
            return self.centroids_raw

##############################################################################
# PCA Class
##############################################################################

class MyPCA():
    """ A class to realize PCA from a dataframe. """

    def __init__(self, data, n_components, clusters=None, centroids=None, centroids_names=None):
        """ The class constructor.

        Args:
            data (Pandas Dataframe): A dataframe with only quantitative datas.
            components (Int): Number of principal components to compute.

        """

        # Assign the values
        self.data = data.copy()
        self.X = self.data.values
        self.names = self.data.index
        self.features = self.data.columns
        self.n_components = n_components
        self.clusters = clusters
        self.centroids = centroids
        self.centroids_names = centroids_names

        # Scale the data
        self.X_scaled = preprocessing.StandardScaler().fit_transform(self.X)

        # Compute PCA
        self.pca = decomposition.PCA(n_components = self.n_components)
        self.pca.fit(self.X_scaled)
        self.X_projected = self.pca.transform(self.X_scaled)

        # If there are centroids
        if centroids is not None:
            self.centroids_projected = self.pca.transform(self.centroids)

    def scatter(self, d1, d2, fig=None, plot=111, figsize=(10,10), **kwargs):
        """ Points projected on factorial plan """

        if ('data_projected' in kwargs) and (kwargs['data_projected'] == 'centroids'):
            data_projected = self.centroids_projected

            if self.centroids_names is not None:
                clusters = self.centroids_names

            else:
                clusters = None

        else:
            data_projected = self.X_projected
            clusters = self.clusters

        if fig == None:
            fig = plt.subplots(figsize=figsize)

        if ('labels' in kwargs) and (kwargs['labels']):
            kwargs['labels'] = self.names

        ax = plt.subplot(plot)
        MyPlot.pca_factorial_plane(
                                    fig,
                                    ax,
                                    data_projected,
                                    self.pca,
                                    (d1, d2),
                                    clusters=clusters,
                                    **kwargs
                                    )

        if fig == None:
            MyPlot.plot_show()

    def circle(self, d1, d2, fig=None, plot=111, figsize=(10,10), **kwargs):
        """ Corralations Circle projected on factorial plan """

        if fig == None:
            fig = plt.subplots(figsize=figsize)

        labels = None
        if ('labels' in kwargs) and (kwargs['labels']):
            labels = self.features

        ax = plt.subplot(plot)
        MyPlot.pca_circle(fig, ax, self.pca, (d1, d2), labels=labels)

        if fig == None:
            MyPlot.plot_show()

    def scree(self):
        """ Plot the scree plot of eigenvalues. """

        fig, ax = MyPlot.new_plot()
        MyPlot.pca_screeplot(ax, self.pca)
        MyPlot.plot_show()

    def scattercircle(self, d1, d2, labelscatter=None, labelcircle=None, limscatter=None, limcircle=None, **kwargs):
        """ Plot the scatterplot and the correlations circle,
        projected on the factorial plan for the d1 and d2
        specified dimensions. """

        fig = plt.subplots(figsize=(16,7.3))
        self.scatter(d1, d2, fig=fig, plot=121, lims=limscatter, labels=labelscatter, **kwargs)
        self.circle(d1, d2, fig=fig, plot=122, lims=limcircle, labels=labelcircle, **kwargs)
        MyPlot.plot_show()

##############################################################################
# Univa Class
##############################################################################

class Univa():
    """ A class to realize univariate anlysis. """
    
    #################
    # Class Methods #
    #################
        
    def gini(data):
        """ Compute the Gini Coefficient. """

        l_data = np.sort(data)

        lorenz = np.cumsum(l_data) / l_data.sum()
        lorenz = np.append([0], lorenz)

        area_under_curve = lorenz[:-1].sum() / len(lorenz)
        S = 0.5 - area_under_curve
        gini = 2*S

        return gini
    
    ####################
    # Instance methods #
    ####################
    
    def __init__(self, variable, quali=False, discrete=False, unit='', filter_var=[]):
        """ The class constructor. Just the variable is mandatory.
        
        Args:
            variable (Pandas Series): The variable that you want to analyze. 

        """
        
        self.variable = variable
        self.quali = quali
        self.discrete = discrete
        self.unit = unit
        self.filter_var = filter_var
        
        if isinstance(self.filter_var, pd.Series):
            self.cond_list = self.filter_var.drop_duplicates().sort_values()
        
        # Set the filter on the complete data
        self.reset_filter()
        
    def check_annotation(self, ax, **kwargs):
        """ Check if an annotation has to be implemented. """
        
        if ('annotate' in kwargs):
            if kwargs['annotate']['plot_name'] == self.filter_name:
                text = kwargs['annotate']['text']
                xy = kwargs['annotate']['xy']
                xytext = kwargs['annotate']['xytext']
                MyPlot.annotate(ax, text, xy=xy, xytext=xytext)
        
    def set_filter(self, condition):
        """ Set a filter based on a value that we can
         find in the filter variable. """
        
        self.filter_name = f"{self.filter_var.name}={condition}"
        self.filtered_data = self.variable[self.filter_var == condition]
        
    def reset_filter(self):
        """ Set the filter to the complete data. """
        
        self.filter_name = self.variable.name
        self.filtered_data = self.variable
        
    def describe_compute(self):
        """ Compute the variable description. """
        
        desc_var = self.filtered_data
        columns = [self.filter_name]
        
        if self.quali:
            index = ['Mode']
            data = [f"{desc_var.mode().iloc[0]}{self.unit}"] 

        else:
            index = [
                'Sample Size',
                'Total',
                'Min',
                'Max',
                'Mode',
                'Mean',
                'Median',
                'Variance (σ²)',
                'Standard Deviation (σ)',
                'Coefficient of variation (CV)',
                'Skewness',
                'Kurtosis'
            ]

            data = [
                f"{len(desc_var):,.0f}",
                f"{desc_var.sum():,.0f}{self.unit}",
                f"{desc_var.sort_values().iloc[0]}{self.unit}",
                f"{desc_var.sort_values().iloc[-1]}{self.unit}",
                f"{desc_var.mode().iloc[0]:.0f}{self.unit}",
                f"{desc_var.mean():.0f}{self.unit}",
                f"{desc_var.median():.0f}{self.unit}",
                f"{desc_var.var():.2f}{self.unit}",
                f"{desc_var.std():.2f}{self.unit}",
                f"{desc_var.std() / self.variable.mean():.2f}",
                f"{desc_var.skew():.2f}",
                f"{desc_var.kurtosis():.2f}"
            ]

        my_s = pd.DataFrame(index=index, data=data, columns=columns)
        
        return my_s
        
    def describe(self):
        """ Describe the variable as Pandas does. """
        
        if len(self.filter_var) > 0: # A describtion filtered by values of the specified column
            
            # The first dataframe column is the non filtered data
            self.reset_filter()
            final_df = self.describe_compute()
            
            # Then the other columns are the filtered data
            for condition in self.cond_list:
                self.set_filter(condition)
                desc = self.describe_compute()
                final_df = pd.concat([final_df, desc], axis=1)
                
            display(final_df)
                
        else: # Just a regular describe
            display(self.describe_compute())
            
    def distribution_compute(self, ax, step, bins, **kwargs):
        """ Compute the variable distribution. """

        data = self.filtered_data
        plot_name = self.filter_name   
        
        if self.discrete or self.quali:
            MyPlot.bar(ax, data, step)
            MyPlot.title(ax, f"{plot_name} - Bar Chart")
            
            if (len(self.variable.drop_duplicates()) > 5):
                plt.xticks(rotation=45)

        else:
            if not bins: # Sturges Rule
                bins = round(1 + math.log(len(data), 2))
                
            MyPlot.hist(ax, data, bins)
            MyPlot.title(ax, f"{plot_name} - Histogram (bins = {bins})")
            
        self.check_annotation(ax, **kwargs)
            
    def distribution(self, bins=None, step=None, **kwargs):
        """ Plot the variable distribution. """
            
        if len(self.filter_var) > 0:
            for condition in self.cond_list:
                fig, ax = MyPlot.new_plot()
                self.set_filter(condition)
                self.distribution_compute(ax, step, bins, **kwargs)
                
        else:
            fig, ax = MyPlot.new_plot()
            self.distribution_compute(ax, step, bins, **kwargs)
        
        MyPlot.plot_show()
        
    def piechart_compute(self, ax):
        """ Compute a Pie Chart distribution. """

        desc_var = self.filtered_data
        plot_name = self.filter_name
        
        MyPlot.piechart(ax, desc_var, var_name=plot_name, unit=self.unit)
        MyPlot.title(ax, f"{plot_name} - Pie Chart Distribution")
        MyPlot.border(ax)
        
    def piechart(self):
        """ Plot a Pie Chart distribution. """

        if len(self.filter_var) > 0:
            for condition in self.cond_list:
                self.set_filter(condition)
                fig, ax = MyPlot.new_plot()
                self.piechart_compute(ax)
                
        else:
            fig, ax = MyPlot.new_plot()
            self.piechart_compute(ax)
            
        MyPlot.plot_show()
        
    def boxplot_compute(self, ax):
        """ Compute a boxplot. """
        
        if not self.quali:
            data = self.filtered_data
            plot_name = self.filter_name   
            MyPlot.boxplot(ax, data)
            MyPlot.title(ax, f"{plot_name} - Boxplot (without outliers)")
            
    def boxplot(self):
        """ Plot a boxplot. """
        
        if len(self.filter_var) > 0:
            for condition in self.cond_list:
                self.set_filter(condition)
                fig, ax = MyPlot.new_plot()
                self.boxplot_compute(ax)
                
        else:
            fig, ax = MyPlot.new_plot()
            self.boxplot_compute(ax)
            
        MyPlot.plot_show()
        
    def lorenz_compute(self, ax):
        """ Compute a Lorenz Curve. """
        
        if not self.quali:
            data = self.filtered_data
            plot_name = self.filter_name           

            MyPlot.lorenz(ax, data)
            gini_var = round(Univa.gini(data),2)
            MyPlot.title(ax, f"Lorenz Curve - {plot_name} (Gini: {gini_var})")
            
    def lorenz(self):
        """ PLot a Lorenz Curve. """
           
        if len(self.filter_var) > 0:
            for condition in self.cond_list:
                self.set_filter(condition)
                fig, ax = MyPlot.new_plot()
                self.lorenz_compute(ax)
        else:         
            fig, ax = MyPlot.new_plot()
            self.lorenz_compute(ax)
        
        MyPlot.plot_show()


##############################################################################
# Biva Class
##############################################################################

class Biva():
    """ A class to compute and plot bivariate analysis. """
    
    
    #################
    # Class Methods #
    #################
    
    def eta_squared(x, y):
        """ Compute and Return Eta Squared. """

        y_mean = y.mean()
        classes = []

        for the_class in x.unique():
            yi_class = y[x == the_class]
            classes.append({'ni': len(yi_class),
                            'mean': yi_class.mean()})

        SCT = sum([(yj - y_mean)**2 for yj in y])
        SCE = sum([c['ni'] * (c['mean'] - y_mean)**2 for c in classes])

        return SCE / SCT
    
    ####################
    # Instance methods #
    ####################
    
    def __init__(self, x, y, type_x='quanti', type_y='quanti', x_class_size=50):
        """ Init the Biva() instance.
        
        Args:
            x (Pandas Series): the first variable
            y (Pandas Series): the second variable
            type_x (str): 'quanti' or 'quali'
            type_y (str): 'quanti' or 'quali'
            x_class_size (int): if x is quanti but needed to be discretized

        """
        
        self.x = x
        self.y = y
        self.type_x = type_x
        self.type_y = type_y
        self.x_class_size = x_class_size
    
    def describe(self):
        """ Return classic indexes. """
        
        if (self.type_x == 'quanti') and (self.type_y == 'quanti'):

            # R² - Pearson (Or correlation) Coefficient
            coefcor = st.pearsonr(self.x, self.y)[0]
            print(f"Correlation Coefficient (R²): {coefcor:.2f}")

            # σxy - Covariance
            covariance = np.cov(self.x, self.y)[0][1]
            print(f"Covariance (σxy): {covariance:.2f}")
            
        if (self.type_x == 'quali') and (self.type_y == 'quanti'):
            
            # η2 - Eta Squared
            print(f"Eta Squared (η2): {Biva.eta_squared(self.x, self.y):.2f}")
            
        if (self.type_x == 'quali' and (self.type_y == 'quali')):
            
            # χ² - Khi-2
            khi2 = self.khi2(contin=False, heatmap=False)
            print(f"Khi2 (χ²): {khi2:.0f}")
            
            
    def scatterplot(self, linreg=True, **kwargs):
        """ Plot a scatter plot + a linear regression. """

        if (self.type_x == 'quanti') and (self.type_y == 'quanti'):
            fig, ax = MyPlot.new_plot(**kwargs)
            MyPlot.scatter(ax, self.x, self.y)
            MyPlot.bg(ax)
            MyPlot.title(ax, f'Scatter Plot: [{self.x.name}] vs [{self.y.name}]')
            MyPlot.border(ax)
        
            if linreg:
                MyPlot.lin_reg(ax, self.x, self.y)
            
        MyPlot.plot_show()
            
    def boxplots(self, **kwargs):
        """ Plot boxplots for a quanti-quanti analysis. """
        
        if (self.type_x == 'quanti') and (self.type_y == 'quanti'):
            fig, ax = MyPlot.new_plot(**kwargs)
            MyPlot.boxplots(ax, self.x, self.y, class_size=self.x_class_size)
            MyPlot.bg(ax)
            MyPlot.border(ax)
            MyPlot.title(ax, f"[{self.y.name}] boxplots by [{self.x.name}] class")
            
            MyPlot.plot_show()
            
    def anova(self, **kwargs):
        """ Compute and plot an ANOVA analysis. """
        
        if (self.type_x == 'quali') and (self.type_y == 'quanti'):
            fig, ax = MyPlot.new_plot(**kwargs)
            MyPlot.anova(ax, self.x, self.y)
            MyPlot.bg(ax)
            MyPlot.title(ax, f"Analysis of Variance (η2 = {Biva.eta_squared(self.x, self.y):.2f})")
            MyPlot.border(ax)
            MyPlot.labels(ax, self.y.name, self.x.name)
            MyPlot.plot_show()
            
    def khi2(self, contin=False, heatmap=False, width=5, heigth=10):
        """ Compute a khi2 test. """

        dftemp = pd.concat([self.x, self.y], axis=1)

        X = self.x.name
        Y = self.y.name

        dfpivot = dftemp[[X, Y]].pivot_table(index=X, columns=Y, aggfunc=len, fill_value=0)

        tx = dftemp[X].value_counts()
        ty = dftemp[Y].value_counts()
        
        if contin:
            dfpivot.loc[:, "Total"] = tx
            dfpivot.loc["total", :] = ty
            dfpivot.loc["total", "Total"] = len(dftemp)
            contin_table = dfpivot.copy()
            dfpivot = dfpivot.drop('total').drop('Total', axis=1)

        tx = pd.DataFrame(tx)
        ty = pd.DataFrame(ty)

        tx.columns = ["c"]
        ty.columns = ["c"]
        n = len(dftemp)

        indep = tx.dot(ty.T) / n

        measure = ((dfpivot - indep)**2) / indep
        xi_n = measure.sum().sum()
        
        # Print Khi-2
        if (contin == False) and (heatmap == False):
            return xi_n
        else:
            print(f"χ² : {xi_n:.0f}")
        
        # Print Contingence Table
        if contin:
            display(contin_table)
        
        # Print heatmap
        if heatmap:
            fig = plt.figure(figsize=(width,heigth))
            plot = sns.heatmap(measure / xi_n, annot=dfpivot)
            plot = plt.title(f"Khi-2 = {xi_n:.0f} - Between {self.x.name} and {self.y.name}")
            plt.show()