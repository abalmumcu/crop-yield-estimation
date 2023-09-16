import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plt_corr(dataset,size=(8,8)):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataset.corr())
    fig.colorbar(cax)
    ax.yaxis.set_major_locator(MultipleLocator(1)) 
    ax.xaxis.set_major_locator(MultipleLocator(1)) 
    ax.set_xticklabels([''] + dataset.columns.values.tolist())
    ax.set_yticklabels([''] +dataset.columns.values.tolist())
    plt.title('Correlation Map')
    plt.show()

def plot_index(indx,df_date,size=(8,8),save_path="",plt_color='r-o',plt_title=""):
    fig = plt.figure(figsize=size)
    graph = fig.add_subplot(111)
    graph.plot(df_date,indx,plt_color)
    graph.set_xticks(df_date)
    plt.title(plt_title)
    plt.locator_params(axis='x', nbins=10)
    if save_path != "":
        plt.savefig(save_path+".png")
    plt.show()

