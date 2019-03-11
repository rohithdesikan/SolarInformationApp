# --------------------------------------------------------------
######### Import Relevant Packages
from flask import Flask, render_template, request, redirect, Markup

# Base Packages
import numpy as np
import pandas as pd
import dill

# Geo Packages
import geopandas as gpd
import geopy as gpy
from geopy.geocoders import Nominatim
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import shapefile
import pysal as ps
import fiona

import json

# Data Visualization Packages
# Bokeh
import bokeh as bk
from bokeh.io import output_notebook, show, output_file, save
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import (ColumnDataSource as cds, Plot, DatetimeAxis, PanTool, WheelZoomTool, HoverTool,
                          tickers, BoxAnnotation, Panel, Range1d, LabelSet, Label, NumeralTickFormatter,
                          LogColorMapper, GeoJSONDataSource, LinearColorMapper, ColorBar,
                          LogTicker, BasicTicker, CategoricalColorMapper, FixedTicker, AdaptiveTicker)
from bokeh.palettes import viridis, magma, inferno, cividis, Greens, Blues, PuRd, YlOrRd, YlOrBr, RdYlGn, RdYlGn8 as palette
from bokeh.embed import file_html, components
from bokeh.layouts import layout, gridplot
from bokeh.resources import INLINE, CDN

# output_notebook()

# Import Machine Learning Packages
from sklearn import base
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn import pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import FeatureUnion

# --------------------------------------------------------------



# --------------------------------------------------------------
######### Define a constant style for all plots
def style(p):
        # Title
        p.title.align = 'center'
        p.title.text_font_size = '16pt'
        p.title.text_font = 'sans serif'

        # Axis titles
        p.xaxis.axis_label_text_font_size = '12pt'
        p.xaxis.axis_label_text_font_style = 'bold'
        p.yaxis.axis_label_text_font_size = '12pt'
        p.yaxis.axis_label_text_font_style = 'bold'

        # Tick labels
        p.xaxis.major_label_text_font_size = '10pt'
        p.yaxis.major_label_text_font_size = '10pt'

        return p
# --------------------------------------------------------------



# --------------------------------------------------------------
######### Get Pickled Data
replica_final = dill.load(open('replica_final.pkd', 'rb'))

# --------------------------------------------------------------



# --------------------------------------------------------------
######### Function to pull data and clean based on user input
def obtain_clean_data():
    replica_final = dill.load(open('replica_final.pkd', 'rb'))
    ypred_train_gbr_lidar = dill.load(open('ypred_train_gbr_lidar.pkd', 'rb'))
    ypred_test_gbr_lidar = dill.load(open('ypred_test_gbr_lidar.pkd', 'rb'))
    ypred_train_dem = dill.load(open('ypred_train_dem.pkd', 'rb'))
    ypred_test_dem = dill.load(open('ypred_test_dem.pkd', 'rb'))


    replica_f = replica_final.fillna(method='ffill', axis = 0)

    # Find the columns within the data and send them to a list
    replica_cols = replica_f.columns.tolist()

    # Split at company_ty which is the split between lidar and demographic
    cty = replica_cols.index('company_ty')

    r_lidar = replica_f.iloc[:,:cty]
    r_lidar.drop(['state_abbr', 'county_name', 'centroid_x', 'centroid_y', 'polygons'], axis = 1, inplace = True)
    r_demographic = replica_f.iloc[:,cty:-2]

    savings = replica_f['Savings [$/yr]']

    # Set up Lidar data ML sets
    X_train_lidar, X_test_lidar, y_train_lidar, y_test_lidar = train_test_split(r_lidar,
                                                                                savings,
                                                                                test_size = 0.2,
                                                                                random_state = 42)
    y_train_lidar_indices = y_train_lidar.index.values.tolist()
    y_test_lidar_indices = y_test_lidar.index.values.tolist()

        # Set up demographic ML sets
    X_train_dem, X_test_dem,  y_train_dem, y_test_dem = train_test_split(r_demographic,
                                                                        savings,
                                                                        test_size = 0.2,
                                                                        random_state = 42)


    train_one_hot = pd.get_dummies(X_train_dem[['company_ty',
                                            'climate_zone_description',
                                            'moisture_regime',
                                            'locale']])
    X_train_dem = X_train_dem.join(train_one_hot)
    X_train_dem.drop(['company_ty', 'climate_zone_description', 'moisture_regime', 'locale'], axis = 1, inplace = True)

    test_one_hot = pd.get_dummies(X_test_dem[['company_ty',
                                            'climate_zone_description',
                                            'moisture_regime',
                                            'locale']])
    X_test_dem = X_test_dem.join(test_one_hot)
    X_test_dem.drop(['company_ty', 'climate_zone_description', 'moisture_regime', 'locale'], axis = 1, inplace = True)

    y_train_dem_indices = y_train_lidar.index.values.tolist()
    y_test_dem_indices = y_test_lidar.index.values.tolist()


    X_train_ensemble = np.vstack((ypred_train_gbr_lidar, ypred_train_dem)).T
    y_train_ensemble = y_train_lidar.values.tolist()
    X_test_ensemble = np.vstack((ypred_test_gbr_lidar, ypred_test_dem)).T
    y_test_ensemble = y_test_lidar.values.tolist()

    # Build the ensemble model
    ensemble_linear = Ridge(alpha = 2, fit_intercept = True)
    ensemble_linear.fit(X_train_ensemble, y_train_ensemble)
    ypred_train_ensemble = ensemble_linear.predict(X_train_ensemble)
    ypred_test_ensemble = ensemble_linear.predict(X_test_ensemble)


    replica_small = replica_final[['geoid', 'state_abbr', 'county_name' , 'polygons', 'avg_yearly_bill_dlrs']]
    replica_train = replica_small.iloc[y_train_lidar_indices, :]
    replica_train['Savings_Predictions'] = ypred_train_ensemble

    replica_test = replica_small.iloc[y_test_lidar_indices, :]
    replica_test['Savings_Predictions'] = ypred_test_ensemble
    replica_plot = pd.concat([replica_train, replica_test], axis = 0)

    # Turning replica data frame into a geopandas dataframe
    replica_plot.rename({'polygons':'geometry'},inplace=True,axis=1)
    replica_geoplot = gpd.GeoDataFrame(replica_plot, geometry='geometry')

    # # Chose the demographic subset of the data
    # replica_cols = replica_final.columns.tolist()
    #
    # plys = replica_cols.index('polygons')
    # cty = replica_cols.index('company_ty')
    # locale = replica_cols.index('locale')
    #
    # r_base = replica_final.iloc[:,:plys]
    # r_dems = replica_final.iloc[:,cty:locale+1]
    #
    # rdems = pd.concat([r_base,r_dems],axis=1)
    # rdems[['Savings [$/yr]','Savings [% Annual Income]']] = replica_final.iloc[:,-2:]
    #
    # # Work with the Demographic Subset to get Mean HH Income and Total Population
    # rdems2 = replica_final[['geoid','state_abbr','county_name','polygons','hh_med_income','pop_total']]
    #
    #
    # # Choose the lidar subset of the data
    # r_lid = replica_final.iloc[:,:cty]
    # r_end = replica_final.iloc[:,locale+1:]
    # rlidar = pd.concat([r_lid,r_end],axis=1)
    #
    # # Work with the Lidar Subset to get dev roof area and MWH gen potential
    # rlidar2 = rlidar.copy()
    # # Choose the Multifamily vs Single Family subset of the data
    # devp_m2_cols = [x for x in rlidar2.columns[rlidar2.columns.str.contains('devp_m2')]]
    # hh_cols = [x for x in rlidar2.columns[rlidar2.columns.str.contains('hh')]]
    # mwh_cols = [x for x in rlidar2.columns[rlidar2.columns.str.contains('mwh')]]
    #
    # rlidar2['hh'] = rlidar2[hh_cols].sum(axis=1)
    # rlidar2['devp_m2'] = rlidar2[devp_m2_cols].sum(axis=1)
    # rlidar2['mwh'] = rlidar2[mwh_cols].sum(axis=1)
    # rlidar2 = rlidar2[['hh','devp_m2','mwh','Savings [% Annual Income]']]
    #
    #
    # rplot = pd.concat([rdems2,rlidar2],axis=1)
    #
    #
    # USA Basemap File
    file = '/Users/rohithdesikan/Desktop/Data Analysis/The Data Incubator/Capstone Project/states_21basic/states.shp'
    mapdf = gpd.read_file(file)
    #
    # # Turning replica data frame into a geopandas dataframe
    # rplot.rename({'polygons':'geometry'},inplace=True,axis=1)
    # rgeo = gpd.GeoDataFrame(rplot, geometry='geometry')

    return mapdf, replica_geoplot
# --------------------------------------------------------------




# --------------------------------------------------------------
######### Function to graph data
def create_graph(mapdf,rgeo,statef):
    # mapdf: Basemap of the US
    # rgeo: Replica data as a geo data frame
    # statef: The state chosen by the user

    state_abbr = list(mapdf.loc[mapdf['STATE_NAME'].isin([statef]),'STATE_ABBR']) # Find the abbreviation

    # Find the map and replica as dataframes
    mapstate_df = mapdf[mapdf['STATE_NAME'].isin([statef])]
    rstate_df = rgeo[rgeo['state_abbr'] == state_abbr[0]]


    # Convert the state_dataframe to a json object
    mapjson = mapstate_df.to_json()
    rjson = rstate_df.to_json()

    # Find the latitude and longitude from the base map dataframe
    latmin = round(float(mapstate_df['geometry'].bounds['miny']))-1
    latmax = round(float(mapstate_df['geometry'].bounds['maxy']))+1
    longmin = round(float(mapstate_df['geometry'].bounds['minx']))-1
    longmax = round(float(mapstate_df['geometry'].bounds['maxx']))+1
    longdist = abs(longmin-longmax)
    latdist = latmax-latmin

    if longdist>latdist:
        fig_width=750
        fig_height = int(fig_width*(latdist/longdist))
    elif latdist>longdist:
        fig_height=750
        fig_width = int(fig_height*(longdist/latdist))
    elif latdist==longdist:
        fig_width=750
        fig_height=750


    # Make the geoJSON datasource
    geo_source = GeoJSONDataSource(geojson=rjson)


    # Plot on Bokeh outside of a Function
    # n=6 # Number of colors on the choropleth and legend

    # Plot the Predictions from the ML Model once they are done.
    # Plot on Bokeh outside of a Function

    # Set the colormapper
    # Greens1 = Greens[n]
    # Greens2 = Greens1[::-1]

    palette.reverse()
    cmapper = LinearColorMapper(palette=palette,
                                low=rstate_df['Savings_Predictions'].min(),
                                high=rstate_df['Savings_Predictions'].max())

    color_bar = bk.models.ColorBar(color_mapper=cmapper, ticker=BasicTicker(desired_num_ticks=8),
                     label_standoff=12, border_line_color=None, location=(0,0))

    # Set up the figure
    ppreds = figure(
        title=statef + ' Annual Solar Savings Predictions',
        x_axis_location=None,
        y_axis_location=None,
        x_range=(longmin,longmax),
        y_range=(latmin,latmax),
        tools="pan,wheel_zoom,box_zoom,reset,hover,save"
    )

    # Disable grid lines
    ppreds.grid.grid_line_color = None

    # Plot the required variable
    ppreds.patches('xs',
                  'ys',
                  source=geo_source,
                  fill_color={'field': "Savings_Predictions", 'transform': cmapper},
                  fill_alpha=0.4,
                  line_color="white",
                  line_width=1)

    ppreds.add_tools(HoverTool(
        tooltips= [
        ("State",'@state_abbr'),
        ("County Name", "@county_name"),
        ("Annual Electric Bill", "@avg_yearly_bill_dlrs{1.11}"),
        ("Annual Savings [$]","@Savings_Predictions{1.11}"),
        ("(Long, Lat)", "($x, $y)")
        ],
        formatters={
        "@State":"printf",
        "@county_name": "printf",
        "($x, $y)": "numeral"
        },
        mode='mouse'
    ))


    ppreds.add_layout(color_bar, 'right')
    # p.legend.location = "top_right"
    # p.legend.click_policy="hide"

    style(ppreds)
    # show(ppreds)



    # grid = gridplot([p1, p2, p3, p4, ppreds], ncols=2, plot_width=fig_width, plot_height=fig_height)
    grid = gridplot([ppreds], ncols=2, plot_width=fig_width, plot_height=fig_height)

    # show(grid)

    script, div = components(grid)

    return grid, script, div
# --------------------------------------------------------------




# --------------------------------------------------------------
######### HTML Connection Code
app = Flask(__name__)

######### Main page
@app.route('/')
def index():
  return render_template('index.html')


######### About page
@app.route('/about')
def about():
  return render_template('about.html')

######### Flowchart page
@app.route('/flowchart')
def flowchart():
  return render_template('flowchart.html')


######### State Page
@app.route('/state',methods =['GET','POST'])
def state():
    if request.method == 'GET':
        states = ['Alabama',
                 'Alaska',
                 'Arizona',
                 'Arkansas',
                 'California',
                 'Colorado',
                 'Connecticut',
                 'Delaware',
                 'District of Columbia',
                 'Florida',
                 'Georgia',
                 'Hawaii',
                 'Idaho',
                 'Illinois',
                 'Indiana',
                 'Iowa',
                 'Kansas',
                 'Kentucky',
                 'Louisiana',
                 'Maine',
                 'Maryland',
                 'Massachusetts',
                 'Michigan',
                 'Minnesota',
                 'Mississippi',
                 'Missouri',
                 'Montana',
                 'Nebraska',
                 'Nevada',
                 'New Hampshire',
                 'New Jersey',
                 'New Mexico',
                 'New York',
                 'North Carolina',
                 'North Dakota',
                 'Ohio',
                 'Oklahoma',
                 'Oregon',
                 'Pennsylvania',
                 'Rhode Island',
                 'South Carolina',
                 'South Dakota',
                 'Tennessee',
                 'Texas',
                 'Utah',
                 'Vermont',
                 'Virginia',
                 'Washington',
                 'West Virginia',
                 'Wisconsin',
                 'Wyoming']

        return render_template('state.html',states=states)
    else:
        statef=request.form['state']
        mapdf, replica_geoplot = obtain_clean_data()
        grid,script,div = create_graph(mapdf,replica_geoplot,statef)
        # return render_template('stock.html', graph_num=plot)
        return Markup(file_html(grid,resources=CDN,template='state.html'))

if __name__ == '__main__':
  app.run(port=33507,debug=True)
