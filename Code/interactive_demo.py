
# start a light-weight webserver
# Go to iCloud folder and run:
# http-server -p 8000

import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_player
import dash_bootstrap_components as dbc

import plotly
import plotly.express as px
import plotly.graph_objects as go

import base64
from PIL import Image

# df = pd.read_csv("suicide_rates.csv")
# df = pd.read_pickle("current_tracker.pickle")

df = pd.read_csv("joined.csv")

df.loc[:, 'border_width'] = df.loc[:, 'unique_id'].astype(int)%2
df.loc[:, 'simple_id'] = df.loc[:, 'unique_id'].astype(int) #%30

# img = Image.open('../data/dbro_map.png')
# img.LOAD_TRUNCATED_IMAGES = True

# app = dash.Dash(__name__)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

#---------------------------------------------------------------
app.layout = html.Div([
        html.Div([
            html.Pre(children= "Dybbølsbro",
            style={"text-align": "center", "font-size":"100%", "color":"black"})
        ]),

        html.Div([
            dcc.Graph(id='img_plot')
            ]),

        dbc.Row([
                dbc.Col([
                    dcc.Slider(id='frame-slider',
                        min=2000,
                        max=20000,
                        value=12933,
                        step=1,)
                ], 
                style={'padding': '0% 30%'}),
            ]),


        dbc.Row([
            dbc.Col([], width=5),
            dbc.Col([
                dcc.Input(id='inpp',
                    placeholder='Enter a value...',
                    type='text',
                    value='',
                    style={"width": "100%", "size": "30"}),
            ]),
            dbc.Col([], width=5),
        ], justify="center"),

        dbc.Row([
            dbc.Col([], width=4),
            dbc.Button('⏪ -20 frames', id='dec_button', color="dark"),
            dbc.Button('+20 frames ⏩', id='inc_button', color="dark", style={'padding': '10px', 'margin-left': '20px'}),
            dbc.Col([], width=4),
            ], justify="center", style={'margin-top': '20px'}),


        # Video Players
        dbc.Row([
            dbc.Col([
                dash_player.DashPlayer(
                    id='video-player',
                    url='http://localhost:8000/Videos/24032021/Processed/2403_S7_sync.mp4',
                    controls=False,
                    width='96%'
                ),
            ]),

            dbc.Col([
                dash_player.DashPlayer(
                    id='video-player2',
                    url='http://localhost:8000/Videos/24032021/Processed/2403_edi_sync.mp4',
                    controls=False,
                    width='96%'
                ),
            ]),

            dbc.Col([
                dash_player.DashPlayer(
                    id='video-player3',
                    url='http://localhost:8000/Videos/24032021/Processed/2403_G6_sync.mp4',
                    controls=False,
                    width='96%'
                ),
            ]),
            ], justify="center", 
            style={
                'margin-left': '10px',
                'margin-top': '15px',
                }),

        ])

# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------

@app.callback(
    Output('inpp','value'),
    [Input('frame-slider','drag_value')]
)

def update_input_field(val):
    t = (val/30)/60
    return u'frame: {} - time: {:.1f} min'.format(val, t)

@app.callback(
    Output('img_plot','figure'),
    [Input('frame-slider','drag_value')]
)

def update_img_plot(val):
    # print(years_chosen)

    fig = go.Figure()

    # Add image
    img_width = 1920
    img_height = 1080
    scale_factor = 0.5
    fig.add_layout_image(
            x=0,
            sizex=img_width,
            y=0,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            source="assets/dbro_map.png"
    )
    fig.update_xaxes(showgrid=False, range=(0, img_width), visible=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, scaleanchor='x', range=(img_height, 0), visible=False, showticklabels=False)
    
    # source="https://i.imgur.com/gaFSyAI.png"
    
    # Line shape added programatically
    # fig.add_shape(
    #     type='line', xref='x', yref='y',
    #     x0=650, x1=1080, y0=380, y1=180*val, line_color='cyan'
    # )

    # points = test.iloc[(1500*(val-1)):(1500*val), :]
    frame = val
    window = 150

    points = df[df['frame_id'].between(frame-window, frame)]

    _max = points['frame_id'].max()
    _min = points['frame_id'].min()
    diff = _max-_min
    points.loc[:, 'opacity'] = 1-((_max-points.loc[:, 'frame_id'])/diff).round(2)

    fig.add_trace(go.Scatter(
    x=points['x'],
    y=points['y'],
    text=points['simple_id'],
    mode = "markers",
    # title="layout.hovermode='x'",
    marker_line=dict(
            width=points['border_width'],
            color='Black'
            ),
    marker=dict(
            color=points['color'],
            size=6,
            opacity=points['opacity']
            ),
    ))

    # Set dragmode and newshape properties; add modebar buttons
    fig.update_layout(
        dragmode='drawrect',
        newshape=dict(line_color='cyan'),
        title_text='Bike Detections',
        height=900,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

@app.callback(
    Output('frame-slider','value'),
    [dash.dependencies.Input('inc_button', 'n_clicks'),
    dash.dependencies.Input('dec_button', 'n_clicks')],
    [dash.dependencies.State('frame-slider', 'value')])

def update_output(inc, dec, value):
    pressed_btn = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'inc_button' in pressed_btn:
        return value+20
    else:
        return value-20

@app.callback([Output('video-player', 'seekTo'),
              Output('video-player2', 'seekTo'),
              Output('video-player3', 'seekTo')],
              [Input('frame-slider', 'value')])
def update_prop_seekTo(val):
    frame = val/30
    return frame, frame, frame

if __name__ == '__main__':
    app.run_server(port=8050, host='127.0.0.1', debug=True)

