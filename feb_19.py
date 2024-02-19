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
    html.Div(id='output-plot')  
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
    Output('output-plot', 'children'),
    [Input('column-dropdown', 'value')]
)
def update_plot(selected_columns):
    if selected_columns is not None:
        plot_divs = []
        for chart_type in [ 'vertical_bar', 'line', 'pie', 'scatter']: # Add "bar" in the list for the bar graph
            try:
                # if chart_type == 'bar':
                #     plotly_fig = render_bar_chart(selected_columns)
                if chart_type == 'vertical_bar':
                    plotly_fig = render_ver_bar_chart(selected_columns)
                elif chart_type == 'line':
                    plotly_fig = render_line_chart(selected_columns)
                elif chart_type == 'pie':
                    plotly_fig = generate_pie_chart(selected_columns)
                elif chart_type == 'scatter':
                    plotly_fig = render_scatter_plot(selected_columns)
                else:
                    raise ValueError("Invalid chart type.")
                
                plot_divs.append(html.Div([
                    html.H3(f'{chart_type.capitalize()} Chart'),
                    dcc.Graph(figure=plotly_fig)
                ]))
            except Exception as e:
                plot_divs.append(html.Div([
                    html.H3(f'{chart_type.capitalize()} Chart'),
                    html.Div(f"Error generating plot: {str(e)}")
                ]))
        
        return plot_divs
    else:
        return ''

# def render_bar_chart(selected_columns):
#     if loaded_data is None or len(selected_columns) < 2 or len(selected_columns) > 4:
#         raise ValueError("Select between two to four columns for bar chart.")

#     group_column = selected_columns[0]
#     sum_columns = selected_columns[1:]

#     grouped_data = loaded_data.groupby(group_column)[sum_columns].sum().reset_index()
#     fig = px.bar(grouped_data, x=sum_columns, y=group_column,
#                  orientation='h',
#                  title=f'Horizontal Bar Chart of {", ".join(sum_columns)} by {group_column}',
#                  labels={col: f"Total {col}" for col in sum_columns},
#                  category_orders={group_column: sorted(grouped_data[group_column].unique())},
#                  barmode='group')
#     fig.update_layout(height=500, width=700, showlegend=True)
#     fig.update_traces(marker=dict(line=dict(width=0.8)))
#     return fig

def render_ver_bar_chart(selected_columns):
    if loaded_data is None or len(selected_columns) < 2 or len(selected_columns) > 4:
        raise ValueError("Select between two to four columns for bar chart.")

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
        raise ValueError("Select between two to four columns for line chart.")

    x_column, *sum_columns = selected_columns

    grouped_data = loaded_data.groupby(x_column)[sum_columns].sum().reset_index()
    
    fig = px.line(grouped_data, x=x_column, y=sum_columns, title=f'Line Chart: {", ".join(sum_columns)} over {x_column}',
                  labels={x_column: f'{x_column}', 'value': 'Total'},
                  markers=True)
    fig.update_layout(height=700, width=1098)
    return fig

def render_scatter_plot(selected_columns):
    if loaded_data is None or len(selected_columns) != 2:
        raise ValueError("Select exactly two columns for scatter plot.")

    x_column, y_column = selected_columns

    fig = px.scatter(loaded_data, x=x_column, y=y_column,
                     title=f'Scatter Plot: {y_column} vs {x_column}',
                     labels={x_column: f"{x_column}", y_column: f"{y_column}"})
    fig.update_layout(height=500, width=700)
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
    fig.update_layout(height=700, width=1098)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
