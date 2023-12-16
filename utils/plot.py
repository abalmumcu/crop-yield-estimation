import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score 
import plotly.graph_objects as go
from geopy.geocoders import Nominatim



def plt_corr(dataset,size=(8,8),save_fig=False,save_path=""):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    cax = ax.matshow(dataset.corr())
    fig.colorbar(cax)
    ax.yaxis.set_major_locator(MultipleLocator(1)) 
    ax.xaxis.set_major_locator(MultipleLocator(1)) 
    ax.set_xticklabels([''] + dataset.columns.values.tolist())
    ax.set_yticklabels([''] +dataset.columns.values.tolist())
    plt.title('Correlation Map')
    if save_fig == True:
        plt.savefig(save_path)
    plt.show()



def plot_index(indx,df_date,size=(8,8),save_path="",plt_color='r-o',plt_title=""):
    fig = plt.figure(figsize=size)
    graph = fig.add_subplot(111)
    graph.plot(df_date,indx,plt_color)
    graph.set_xticks(df_date)
    plt.title(plt_title)
    plt.locator_params(axis='x', nbins=10)
    if save_path != "":
        plt.savefig(save_path)
    plt.show()



def plot_r2(y_test,pred_test,yield_data,save_path=""):
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(y_test, pred_test)
    ax.plot([yield_data.min(), yield_data.max()], [yield_data.min(), yield_data.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    #regression line

    ax.annotate("R² = {:.3f}".format(r2_score(y_test, pred_test)), (300, 2000),fontsize=20, fontweight='bold')
    plt.savefig(save_path)
    plt.show()



def get_coordinates(city, state):
    geolocator = Nominatim(user_agent="my-app")
    location = geolocator.geocode(f"{city}, {state}")
    if location:
        return location.latitude, location.longitude
    else:
        return None

def visualize_cities_on_map(city_dict):
    fig = go.Figure()

    for state, cities in city_dict.items():
        for city in cities:
            coordinates = get_coordinates(city, state)
            if coordinates:
                lat, lon = coordinates
                fig.add_trace(
                    go.Scattergeo(
                        lon=[lon],
                        lat=[lat],
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=5
                        ),
                        hoverinfo='text',
                        text=city
                    )
                )

    fig.update_layout(
        title_text='Cities in America',
        showlegend=False,
        geo=dict(
            scope='usa',
            projection_type='albers usa',  
            showland=True,
            landcolor='rgb(217, 217, 217)',
            countrycolor='rgb(255, 255, 255)',
            showocean=True,  
            oceancolor='rgb(204, 255, 255)',  
            showrivers=False, 
            rivercolor='rgb(0, 0, 255)', 
        )
    )

    fig.show()