# start a light-weight webserver
# Go to iCloud folder and run:
# http-server -p 8000

import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_player
import dash_bootstrap_components as dbc

import plotly.graph_objects as go

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import json
import easygui

df = pd.read_csv(easygui.fileopenbox(msg="Choose tracks file.")) 

df.loc[:, 'border_width'] = df.loc[:, 'unique_id'].astype(int)%2
df.loc[:, 'simple_id'] = df.loc[:, 'unique_id'].astype(int)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# -------------------------------------------------------------------
# Components
# -------------------------------------------------------------------

def make_item(i, title="Unknown"):
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H2(
                    dbc.Button(
                        title,
                        color="link",
                        id=f"group-{i}-toggle",
                    )
                )
            ),
            dbc.Collapse(
                dbc.CardBody(f"This is the content of group {i}..."),
                id=f"collapse-{i}",
                style={"maxHeight": "230px", "overflow": "scroll"}
            ),
        ]
    )

# -------------------------------------------------------------------
# Layout
#---------------------------------------------------------------
app.layout = html.Div([
        html.H1(["Cyclist Analysis", dbc.Badge("Alpha", className="ml-1")], style={'margin-left': '20px', 'margin-top': "25px", 'margin-bottom': "25px"}, id="hej"),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(dcc.Graph(id='img_plot')),
                    width={"size": 8},
                    style={'background-color': 'black'}
                ),
                dbc.Col(
                    html.Div([make_item(1, "🚲 All cyclists"), make_item(2, "🚳 Incidents"), make_item(3)], className="accordion"),
                    style={'background-color': 'grey'}
                ),
            ],
            no_gutters=True,
        ),


        dbc.Row([
                dbc.Col([
                    dcc.Slider(id='frame-slider',
                        min=300,
                        max=10000,
                        value=300,
                        step=1,)
                ], 
                style={'padding': '0% 30%', 'margin-top': '10px'}),
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
                    url='http://localhost:8000/Data/24032021/Videos/Processed/2403_S7_sync.mp4',
                    controls=False,
                    width='96%'
                ),
            ]),

            dbc.Col([
                dash_player.DashPlayer(
                    id='video-player2',
                    url='http://localhost:8000/Data/24032021/Videos/Processed/2403_edi_sync.mp4',
                    controls=False,
                    width='96%'
                ),
            ]),

            dbc.Col([
                dash_player.DashPlayer(
                    id='video-player3',
                    url='http://localhost:8000/Data/24032021/Videos/Processed/2403_G6_sync.mp4',
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
    marker=dict(
            color=points['color'],
            size=6,
            opacity=points['opacity']
            ),
    ))

    # Set dragmode and newshape properties; add modebar buttons
    fig.update_layout(
        dragmode='drawclosedpath',
        newshape=dict(line_color='cyan'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return fig

@app.callback(
    Output('frame-slider','value'),
    [Input('inc_button', 'n_clicks'),
    Input('dec_button', 'n_clicks'), 
    Input({'type': 'incident', 'index': ALL}, 'n_clicks')],
    [State('frame-slider', 'value')])

def update_output(inc, dec, incident, value):
    ctx = dash.callback_context
    pressed_btn = [p['prop_id'] for p in ctx.triggered][0]
    print("hwat")
    if 'inc_button' in pressed_btn:
        return value+20
    elif 'dec_button' in pressed_btn:
        return value-20
    else:
        button_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
        return int(button_id)

@app.callback([Output('video-player', 'seekTo'),
              Output('video-player2', 'seekTo'),
              Output('video-player3', 'seekTo')],
              [Input('frame-slider', 'value')])

def update_prop_seekTo(val):
    frame = val/30
    return frame, frame, frame


@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in range(1, 4)],
    [Input(f"group-{i}-toggle", "n_clicks") for i in range(1, 4)],
    [State(f"collapse-{i}", "is_open") for i in range(1, 4)],
)
def toggle_accordion(n1, n2, n3, is_open1, is_open2, is_open3):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "group-1-toggle" and n1:
        return not is_open1, False, False
    elif button_id == "group-2-toggle" and n2:
        return False, not is_open2, False
    elif button_id == "group-3-toggle" and n3:
        return False, False, not is_open3
    return False, False, False

# From SVG path to numpy array of coordinates, each row being a (row, col) point
def path_to_coords(path):
    indices_str = [el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")]
    return np.array(indices_str, dtype=float)


@app.callback(
    Output("collapse-2", "children"),
    Input("img_plot", "relayoutData"),
    prevent_initial_call=True,
)
def update_incident_list(relayout_data):
    if relayout_data['shapes'] is not None:
        path_string = relayout_data['shapes'][0]['path']
        coordinates_raw = [coor.replace("M", "").replace("Z", "").split(",") for coor in path_string.split("L")]
        coordinates = [(float(a), float(b)) for a, b in coordinates_raw]
        polygon = Polygon(coordinates)

        is_inside = lambda row: polygon.contains(Point(row['x'], row['y']))
        points_inside = df[df.apply(is_inside, axis=1)]
        incidents = points_inside.groupby('unique_id').first()
        print(incidents)

        btns = [dbc.ListGroupItem(f"{incidents.iloc[i]['frame_id']}", id={'type': 'incident', 'index': f"{incidents.iloc[i]['frame_id']}"}, n_clicks=incidents.iloc[i]['frame_id'], action=True) for i in range(len(incidents))]

    list_group = dbc.ListGroup(btns)
    return list_group

if __name__ == '__main__':
    app.run_server(port=8050, host='127.0.0.1', debug=True)