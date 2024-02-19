import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import base64
import io

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.title = 'Data Visualization'

loaded_data = None

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
    'width': '97.8%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '4px',
    'borderStyle': 'solid',
    'borderColor': '#4CAF50', 
    'borderRadius': '10px',
    'textAlign': 'center',
    'margin': '10px',
    'backgroundColor': '#f0f0f0', 
    'color': '#333',
    'fontFamily': 'Arial, sans-serif', 
    'fontSize': '1.2em', 
    'cursor': 'pointer', 
    'transition': 'background-color 0.3s ease',  
},



        multiple=True
    ),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename):
    global loaded_data
    content_type, content_string = contents.split(',')
    
    decoded = pd.read_excel(io.BytesIO(base64.b64decode(content_string)))
    loaded_data = decoded
    
    columns = list(decoded.columns)
    
    return html.Div([
        html.H6(filename),
        html.H6([html.B('Select Columns:')], className='column-select-header'),
        dcc.Dropdown(
            id='column-dropdown',
            options=[{'label': i, 'value': i} for i in columns],
            multi=True
        ),
        html.Div(id='output-datatype')
    ])

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children

@app.callback(
    Output('output-datatype', 'children'),
    [Input('column-dropdown', 'value')]
)
def update_datatype(selected_columns):
    if selected_columns is not None:
        options = [
            {'label': 'Bar Chart', 'value': 'bar'},
            {'label': 'Vertical Bar Chart', 'value': 'ver_bar'},
            {'label': 'Line Chart', 'value': 'line'},
            {'label': 'Pie Chart', 'value': 'pie'},
            {'label': 'Scatter Plot', 'value': 'scatter'}
        ]
        return html.Div([
            dcc.Dropdown(
                id='chart-type',
                options=options,
                value='bar'
            ),
            html.Button('Plot', id='plot-button'),
            html.Div(id='output-plot')
        ])
    else:
        return ''

@app.callback(
    Output('output-plot', 'children'),
    [Input('plot-button', 'n_clicks')],
    [State('column-dropdown', 'value'),
     State('chart-type', 'value')]
)
def plot(n_clicks, selected_columns, chart_type):
    if n_clicks:
        try:
            if chart_type == 'bar':
                plotly_fig = render_bar_chart(selected_columns)
            elif chart_type == 'ver_bar':
                plotly_fig = render_ver_bar_chart(selected_columns)
            elif chart_type == 'line':
                plotly_fig = render_line_chart(selected_columns)
            elif chart_type == 'pie':
                plotly_fig = generate_pie_chart(selected_columns)
            elif chart_type == 'scatter':
                plotly_fig = render_scatter_plot(selected_columns)
            else:
                raise ValueError("Invalid chart type.")
                
            return dcc.Graph(figure=plotly_fig)
        except Exception as e:
            return f"Error generating plot: {str(e)}"

def render_bar_chart(selected_columns):
    if loaded_data is None or len(selected_columns) < 2 or len(selected_columns) > 4:
        return html.Div("Select between two to four columns for bar chart.")
    
    group_column = selected_columns[0]
    sum_columns = selected_columns[1:]

    grouped_data = loaded_data.groupby(group_column)[sum_columns].sum().reset_index()
    fig = px.bar(grouped_data, x=sum_columns, y=group_column,
                 orientation='h',
                 title=f'Horizontal Bar Chart of {", ".join(sum_columns)} by {group_column}',
                 labels={col: f"Total {col}" for col in sum_columns},
                 category_orders={group_column: sorted(grouped_data[group_column].unique())},
                 barmode='group') 
    fig.update_layout(height=700, width=1098, showlegend=True)
    fig.update_traces(marker=dict(line=dict(width=0.8)))
    return fig

def render_ver_bar_chart(selected_columns):
    if loaded_data is None or len(selected_columns) < 2 or len(selected_columns) > 4:
        return html.Div("Select between two to four columns for bar chart.")

    group_column = selected_columns[0]
    sum_columns = selected_columns[1:]

    grouped_data = loaded_data.groupby(group_column)[sum_columns].sum().reset_index()

    
    fig = px.bar(grouped_data, x=group_column, y=sum_columns,
                 orientation='v', 
                 title=f'Vertical Bar Chart of {", ".join(sum_columns)} by {group_column}',
                 labels={col: f"Total {col}" for col in sum_columns},
                 category_orders={group_column: sorted(grouped_data[group_column].unique())},
                 barmode='group')

    fig.update_layout(height=700, width=1098, showlegend=True)
    fig.update_traces(marker=dict(line=dict(width=0.8)))
    return fig
  
def render_line_chart(selected_columns):
    if loaded_data is None or len(selected_columns) < 2 or len(selected_columns) > 4:
        return html.Div("Select between two to four columns for bar chart.")
    x_column, y_column = selected_columns

    fig = px.line(loaded_data, x=x_column, y=y_column, title=f'Line Chart: {y_column} over {x_column}',
                  labels={x_column: f'{x_column}', y_column: f'{y_column}'},
                  markers=True) 
    fig.update_layout(height=600,width=1078)
    return fig

def render_scatter_plot(selected_columns):
    if loaded_data is None or len(selected_columns) != 2:
        return html.Div("Select exactly two columns for scatter plot.")

    x_column, y_column = selected_columns

    fig = px.scatter(loaded_data, x=x_column, y=y_column,
                     title=f'Scatter Plot: {y_column} vs {x_column}',
                     labels={x_column: f"{x_column}", y_column: f"{y_column}"})
    return fig

def generate_pie_chart(selected_columns, filters=None):
    if loaded_data is None or filters is not None:
        filtered_data = loaded_data.copy() 
        for column, value in filters.items():
            filtered_data = filtered_data[filtered_data[column] == value]
    else:
        filtered_data = loaded_data

    fig = px.pie(filtered_data, names=selected_columns[0], values=selected_columns[1], 
                 title=f'Pie Chart for {selected_columns[0].capitalize()} ({selected_columns[1].capitalize()})')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(width=750, height=750)
    return fig
    
if __name__ == '__main__':
    app.run_server(debug=True)
