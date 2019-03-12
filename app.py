# --------------------------------------------------------------
######### Import Relevant Packages
from flask import Flask, render_template, request, redirect, Markup

# Base Packages
import numpy as np
import pandas as pd

# File IO and System Packages
import dill

# File Download from Website
import requests

# Geo Packages
import geopandas as gpd

# Data Visualization Packages
# Bokeh
import bokeh as bk
from bokeh.io import output_notebook, show, output_file, save
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import (ColumnDataSource as cds, Plot, DatetimeAxis, PanTool, WheelZoomTool, HoverTool,
                          tickers, BoxAnnotation, Panel, Range1d, LabelSet, Label, NumeralTickFormatter,
                          LogColorMapper, GeoJSONDataSource, LinearColorMapper, ColorBar,
                          LogTicker, BasicTicker, CategoricalColorMapper, FixedTicker, AdaptiveTicker)
from bokeh.palettes import RdYlGn6 as palette
from bokeh.embed import file_html, components
from bokeh.layouts import layout, gridplot
from bokeh.resources import INLINE, CDN

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
######### Perform Data Gathering and Initial Analysis

# --------------------------------------------------------------



# --------------------------------------------------------------
######### Function to pull data and clean based on user input
def obtain_clean_data():
    # USA Basemap File
    replica_fewplot = dill.load(open('replica_few_plot.pkd', 'rb'))
    file = 'states_21basic/states.shp'
    mapdf = gpd.read_file(file)

    return mapdf, replica_fewplot
# --------------------------------------------------------------

# --------------------------------------------------------------
######### Function to graph exploratory data
def create_exploratory_graph(replica_few_plot):
    # mapdf: Basemap of the US
    # rgeo: Replica data as a geo data frame
    # statef: The state chosen by the user

    replica_few_plot1 = pd.DataFrame(replica_few_plot)
    replica_few_plot2 = replica_few_plot1.drop('geometry', axis = 1)
    colormap = {'NV': 'green', 'UT': 'red'}
    colors = [colormap[x] for x in replica_few_plot2['state_abbr']]
    replica_few_plot2['Colors']=colors

    src = cds(replica_few_plot2)

    p1 = figure(plot_width=1000, plot_height=600,
       x_axis_label = 'Total Annual Solar Potential [GWh]',
       y_axis_label ='Observed Annual Savings [$/year]',
       title = 'Relationship between Annual Solar Savings and Solar Irradiation',
       tools="save,pan,undo,wheel_zoom,box_zoom,reset")

    p1.scatter(x='gwh',
               y='Savings [$/yr]',
               size=6, fill_color='Colors',
               line_color = 'black',
               alpha=0.8,
               source=src,
               legend='state_abbr')


    # Add hover tool functionality
    p1.add_tools(HoverTool(
        tooltips=[
            ('State',                        '@state_abbr'),
            ('Total Annual Solar Potential [GWh]',      '@gwh'),
            ( 'Observed Savings',         '@{Savings [$/yr]}{1.11}'), # use @{ } for field names with spaces
        ],
        formatters={
            'gwh'     : 'numeral', # use 'datetime' formatter for 'date' field
            '{Savings [$/yr]}'  : 'numeral',   # use 'printf' formatter for 'OADB ' field
        },
        mode='mouse'
    ))

    # Add labels
    p1.toolbar_location = 'right'
    p1.legend.location = "top_right"

    style(p1)

    grid = gridplot([p1], ncols = 2)

    # show(grid)

    script, div = components(grid)

    return grid, script, div
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

    palette.reverse()
    cmapper = LinearColorMapper(palette=palette,
                                low=rstate_df['Savings_Predictions'].min(),
                                high=rstate_df['Savings_Predictions'].max())

    low = rstate_df['Savings_Predictions'].min()
    high = rstate_df['Savings_Predictions'].max()
    diff = high - low
    ticks = [int(diff/6), int(diff/3), int(diff/2), int(2*diff/3), int(5*diff/6), int(diff)]

    color_bar = bk.models.ColorBar(color_mapper=cmapper, ticker=BasicTicker(desired_num_ticks = 6),
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

@app.route('/explore')
def explore():
    mapdf, replica_geoplot = obtain_clean_data()
    grid,script,div = create_exploratory_graph(replica_geoplot)
    # return render_template('stock.html', graph_num=plot)
    return Markup(file_html(grid,resources=CDN,template='explore.html'))

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
