
import logging
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH
import pandas as pd
import plotly.express as px
import base64
import io
import json
import google.generativeai as genai
import os
import re  # Added for regex pattern matching

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()  # Configure Google Gemini API
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
except Exception as e:
    logger.error(f"Error configuring Gemini API: {str(e)}")
    raise

# Modern color palette
COLORS = {
    'primary': '#2C3E50',    # Dark blue/slate for headers
    'secondary': '#3498DB',   # Bright blue for accents
    'background': '#ECF0F1',  # Light gray for background
    'text': '#2C3E50',        # Dark text
    'success': '#2ECC71',     # Green for success elements
    'border': '#BDC3C7',      # Light gray for borders
    'ai': '#9B59B6',          # Purple for AI elements
    'chart_colors': px.colors.qualitative.G10  # Professional chart color scheme
}

# Custom CSS for better styling
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.title = 'AI-Enhanced Data Visualization Dashboard'

# Global variables to store data and recommendations
loaded_data = None
data_summary = None
stored_viz_recommendations = []  # Store visualization recommendations to avoid repeated API calls

# Custom styles
container_style = {
    'maxWidth': '1200px',
    'margin': '0 auto',
    'padding': '20px',
    'backgroundColor': COLORS['background'],
    'fontFamily': 'Roboto, sans-serif',
    'color': COLORS['text'],
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'borderRadius': '8px',
}

card_style = {
    'padding': '20px',
    'borderRadius': '8px',
    'backgroundColor': 'white',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.05)',
    'marginBottom': '20px',
}

header_style = {
    'color': COLORS['primary'],
    'padding': '10px 0',
    'marginBottom': '20px',
    'borderBottom': f'2px solid {COLORS["secondary"]}',
    'textAlign': 'center',
}

upload_style = {
    'width': '100%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '2px',
    'borderStyle': 'dashed',
    'borderColor': COLORS['secondary'],
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px 0',
    'backgroundColor': 'white',
    'color': COLORS['text'],
    'fontFamily': 'Roboto, sans-serif',
    'fontSize': '1.1em',
    'cursor': 'pointer',
    'transition': 'all 0.3s ease',
}

dropdown_style = {
    'marginBottom': '20px',
    'fontFamily': 'Roboto, sans-serif',
}

button_style = {
    'backgroundColor': COLORS['secondary'],
    'color': 'white',
    'border': 'none',
    'padding': '10px 15px',
    'borderRadius': '4px',
    'cursor': 'pointer',
    'fontWeight': '500',
    'textAlign': 'center',
    'margin': '10px 0',
    'transition': 'background-color 0.3s ease',
}

ai_card_style = {
    **card_style,
    'borderLeft': f'4px solid {COLORS["ai"]}',
}

ai_button_style = {
    **button_style,
    'backgroundColor': COLORS['ai'],
}

# App layout with improved UI and AI integration
app.layout = html.Div(style=container_style, children=[
    html.Div(style=header_style, children=[
        html.H1('AI-Enhanced Data Visualization Dashboard', style={'fontWeight': '500'}),
        html.P('Upload your Excel data and create interactive visualizations with AI assistance', 
               style={'color': COLORS['text'], 'opacity': '0.8'})
    ]),
    
    html.Div(style=card_style, children=[
        html.H3('Data Upload', style={'color': COLORS['primary'], 'marginBottom': '15px'}),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.I(className='fas fa-upload', style={'marginRight': '10px'}),
                'Drag and Drop or ',
                html.A('Select Excel Files', style={'color': COLORS['secondary'], 'fontWeight': '500'})
            ]),
            style=upload_style,
            multiple=True
        ),
    ]),
    
    html.Div(id='output-data-upload'),
    html.Div(id='output-plot'),
    
    # Store for visualization recommendations
    dcc.Store(id='viz-recommendations-store')
])

def get_visualization_recommendations(headers, force_refresh=False):
    """Get AI recommendations for which columns to visualize together"""
    global stored_viz_recommendations
    
    # Return stored recommendations if available and not forcing refresh
    if stored_viz_recommendations and not force_refresh:
        return stored_viz_recommendations
    
    prompt = f"""
You are given a dataset with the following columns: {headers}.

Your task is to generate *only* visualization recommendations in the form of valid (x, y) column pairs. Follow this exact format:

Visualization Recommendations
x-axis: [column name]
y-axis: [column name]

Instructions:
- Only recommend pairs using columns from the provided list above.
- Do NOT suggest or create any new or imaginary column names.
- The column names must exactly match those in the list (case-sensitive).
- Only output column *pairs* — do not suggest single-column visualizations (e.g., histograms or pie charts).

Return only valid x-y axis pairs using the actual column names.
"""

    
    try:
        # Use a different model for visualization recommendations
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        analysis = response.text
        
        # Parse the visualization recommendations
        viz_recommendations = []
        logger.info(f"Analysis: {analysis}")
        
        # Improved pattern matching to capture all recommendations
        pattern = r'Visualization Recommendations\s*x-axis:\s*([^\n]+)[\s\n]*y-axis:\s*([^\n]+)'
        matches = re.findall(pattern, analysis, re.IGNORECASE)
        
        for x_col, y_col in matches:
            x_col = x_col.strip("'\" \n")
            y_col = y_col.strip("'\" \n")
            if x_col and y_col:
                viz_recommendations.append({
                    'x_column': x_col,
                    'y_column': y_col,
                    'description': f"Plot {y_col} against {x_col}"
                })
        
        # If the above pattern doesn't work, try an alternative pattern
        if not viz_recommendations:
            # Look for x-axis/y-axis pairs without the "Visualization Recommendations" header
            alt_pattern = r'x-axis:\s*([^\n]+)[\s\n]*y-axis:\s*([^\n]+)'
            matches = re.findall(alt_pattern, analysis, re.IGNORECASE)
            
            for x_col, y_col in matches:
                x_col = x_col.strip("'\" \n")
                y_col = y_col.strip("'\" \n")
                if x_col and y_col:
                    viz_recommendations.append({
                        'x_column': x_col,
                        'y_column': y_col,
                        'description': f"Plot {y_col} against {x_col}"
                    })
        
        logger.info(f"Extracted recommendations: {viz_recommendations}")
        
        # Store the recommendations for future use
        stored_viz_recommendations = viz_recommendations
        return viz_recommendations
    except Exception as e:
        logger.error(f"Error generating visualization recommendations: {str(e)}")
        return []

def generate_data_summary(df):
    """Generate a summary of the dataframe for AI context"""
    summary = {
        'columns': list(df.columns),
        'shape': df.shape,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'sample_data': df.head(3).to_dict(),
        'numeric_columns': list(df.select_dtypes(include=['number']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': list(df.select_dtypes(include=['datetime']).columns),
    }
    return summary

def parse_contents(contents, filename):
    global loaded_data, data_summary
    content_type, content_string = contents.split(',')

    decoded = pd.read_excel(io.BytesIO(base64.b64decode(content_string)))
    loaded_data = decoded

    # Generate data summary for AI
    data_summary = generate_data_summary(decoded)
    
    columns = list(decoded.columns)
    
    # Get visualization recommendations (initial call)
    viz_recommendations = get_visualization_recommendations(columns)

    return html.Div([
        html.Div(style=card_style, children=[
            html.H4(f'File: {filename}', style={'color': COLORS['primary'], 'marginBottom': '15px'}),
            
            html.Div([
                html.Div([
                    html.H5('Data Preview:', style={'marginBottom': '10px', 'color': COLORS['primary']}),
                    html.Div(style={'overflowX': 'auto'}, children=[
                        dash.dash_table.DataTable(
                            data=decoded.head(5).to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in decoded.columns],
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'fontFamily': 'Roboto, sans-serif',
                                'padding': '8px',
                                'textAlign': 'left'
                            },
                            style_header={
                                'backgroundColor': COLORS['secondary'],
                                'color': 'white',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ]
                        )
                    ]),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H5('Select Columns for Visualization:', 
                        style={'marginBottom': '10px', 'color': COLORS['primary']}),
                    dcc.Dropdown(
                        id='column-dropdown',
                        options=[{'label': i, 'value': i} for i in columns],
                        multi=True,
                        placeholder="Select columns to visualize",
                        style=dropdown_style
                    ),
                    html.Div(id='output-datatype'),
                    html.P("Note: Select 2-4 columns for most visualizations. For scatter plots, select exactly 2 columns.", 
                        style={'fontSize': '0.9em', 'color': '#666', 'marginTop': '5px'})
                ]),
            ])
        ]),
        
        # AI Suggestions Card
        html.Div(style=ai_card_style, children=[
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'}, children=[
                html.I(className='fas fa-robot', style={'color': COLORS['ai'], 'fontSize': '24px', 'marginRight': '10px'}),
                html.H4('AI Visualization Assistant', style={'color': COLORS['ai'], 'margin': '0'})
            ]),
            html.P("Get intelligent suggestions for which columns to visualize together based on your data.", 
                  style={'marginBottom': '15px'}),
            
            # Display immediate visualization recommendations if available
            html.Div([
                html.H5("Recommended Visualizations", 
                       style={'color': COLORS['ai'], 'marginBottom': '15px'}) if viz_recommendations else None,
                
                # New chip-style UI for recommendations
                html.Div([
                    html.Div(
                        [
                            html.Div(
                                f"{rec['y_column']} vs {rec['x_column']}",
                                id={'type': 'viz-chip', 'index': i},
                                style={
                                    'display': 'inline-block',
                                    'backgroundColor': COLORS['ai'],
                                    'color': 'white',
                                    'padding': '8px 15px',
                                    'borderRadius': '20px',
                                    'margin': '5px',
                                    'cursor': 'pointer',
                                    'transition': 'all 0.3s ease',
                                    'fontSize': '0.9em',
                                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                    'userSelect': 'none',
                                }
                            ) for i, rec in enumerate(viz_recommendations)
                        ],
                        style={
                            'display': 'flex',
                            'flexWrap': 'wrap',
                            'marginBottom': '15px'
                        }
                    ),
                    
                    # Dropdown alternative for recommendations
                    html.Div([
                        html.Label("Or select from dropdown:", style={'marginRight': '10px', 'fontSize': '0.9em'}),
                        dcc.Dropdown(
                            id='recommendation-dropdown',
                            options=[
                                {'label': f"{rec['y_column']} vs {rec['x_column']}", 
                                 'value': json.dumps({'x': rec['x_column'], 'y': rec['y_column']})}
                                for rec in viz_recommendations
                            ],
                            placeholder="Select a recommendation",
                            style={'width': '300px', 'display': 'inline-block'}
                        ),
                        html.Button(
                            "Apply",
                            id='apply-dropdown-rec',
                            style={
                                'backgroundColor': COLORS['ai'],
                                'color': 'white',
                                'border': 'none',
                                                                'padding': '8px 15px',
                                'borderRadius': '4px',
                                'cursor': 'pointer',
                                'marginLeft': '10px',
                                'fontSize': '0.9em'
                            }
                        )
                    ], style={'marginTop': '10px', 'display': 'flex', 'alignItems': 'center'})
                ]) if viz_recommendations else None,
            ]),
            
            html.Hr(style={'margin': '20px 0'}) if viz_recommendations else None,
            
            html.Div([
                html.Button(
                    [
                        html.I(className='fas fa-sync', style={'marginRight': '8px'}),
                        "Refresh Recommendations"
                    ],
                    id='refresh-recommendations',
                    style=ai_button_style
                )
            ])
        ])
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
    Output('viz-recommendations-store', 'data'),
    [Input('refresh-recommendations', 'n_clicks')],
    [State('viz-recommendations-store', 'data')],
    prevent_initial_call=True
)
def refresh_recommendations(n_clicks, current_data):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    if data_summary is None:
        return []
    
    # Force refresh of recommendations
    viz_recommendations = get_visualization_recommendations(data_summary['columns'], force_refresh=True)
    return viz_recommendations

@app.callback(
    Output('column-dropdown', 'value'),
    [
        Input({'type': 'viz-chip', 'index': ALL}, 'n_clicks'),
        Input('apply-dropdown-rec', 'n_clicks')
    ],
    [
        State({'type': 'viz-chip', 'index': ALL}, 'id'),
        State('recommendation-dropdown', 'value')
    ],
    prevent_initial_call=True
)
def update_column_dropdown(chip_clicks, dropdown_click, chip_ids, dropdown_value):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Get the ID of the clicked button
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle viz-chip clicks
    if 'viz-chip' in triggered_id:
        try:
            button_data = json.loads(triggered_id)
            button_index = button_data['index']
            
            # Get the corresponding recommendation
            if stored_viz_recommendations and button_index < len(stored_viz_recommendations):
                rec = stored_viz_recommendations[button_index]
                return [rec['x_column'], rec['y_column']]
        except Exception as e:
            logger.error(f"Error handling viz-chip click: {str(e)}")
    
    # Handle dropdown selection
    elif triggered_id == 'apply-dropdown-rec' and dropdown_value:
        try:
            selected_pair = json.loads(dropdown_value)
            return [selected_pair['x'], selected_pair['y']]
        except Exception as e:
            logger.error(f"Error handling dropdown selection: {str(e)}")
    
    raise dash.exceptions.PreventUpdate

# Fix the clientside callback for chip animation
@app.callback(
    Output({'type': 'viz-chip', 'index': MATCH}, 'style'),
    Input({'type': 'viz-chip', 'index': MATCH}, 'n_clicks'),
    State({'type': 'viz-chip', 'index': MATCH}, 'style'),
    prevent_initial_call=True
)
def update_chip_style(n_clicks, current_style):
    if n_clicks:
        # Create a copy of the current style
        new_style = dict(current_style)
        # Add a subtle animation effect
        new_style['transform'] = 'scale(0.95)'
        new_style['boxShadow'] = '0 1px 2px rgba(0,0,0,0.2)'
        return new_style
    return current_style

@app.callback(
    Output('output-plot', 'children'),
    [Input('column-dropdown', 'value')]
)
def update_plot(selected_columns):
    if selected_columns is not None and len(selected_columns) > 0:
        plot_divs = []
        chart_types = ['vertical_bar', 'line', 'pie', 'scatter']
        
        for chart_type in chart_types:
            try:
                if chart_type == 'vertical_bar':
                    plotly_fig = render_ver_bar_chart(selected_columns)
                    title = 'Vertical Bar Chart'
                elif chart_type == 'line':
                    plotly_fig = render_line_chart(selected_columns)
                    title = 'Line Chart'
                elif chart_type == 'pie':
                    plotly_fig = generate_pie_chart(selected_columns)
                    title = 'Pie Chart'
                elif chart_type == 'scatter':
                    plotly_fig = render_scatter_plot(selected_columns)
                    title = 'Scatter Plot'
                else:
                    raise ValueError("Invalid chart type.")
                
                # Update figure layout for consistent styling
                plotly_fig.update_layout(
                    template='plotly_white',
                    colorway=COLORS['chart_colors'],
                    title_font=dict(family="Roboto, sans-serif", size=18, color=COLORS['primary']),
                    font=dict(family="Roboto, sans-serif", color=COLORS['text']),
                    legend=dict(
                        font=dict(family="Roboto, sans-serif", size=12),
                        bordercolor=COLORS['border'],
                        borderwidth=1,
                    ),
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                
                plot_divs.append(html.Div(style=card_style, children=[
                    html.H3(title, style={'color': COLORS['primary'], 'marginBottom': '15px'}),
                    dcc.Graph(
                        figure=plotly_fig,
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': f'{title.replace(" ", "_").lower()}',
                                'height': 700,
                                'width': 1200,
                                'scale': 2
                            }
                        }
                    )
                ]))
            except Exception as e:
                plot_divs.append(html.Div(style=card_style, children=[
                    html.H3(f'{chart_type.capitalize()} Chart', style={'color': COLORS['primary']}),
                    html.Div(
                        f"Cannot generate this visualization with the selected columns: {str(e)}",
                        style={
                            'padding': '20px',
                            'backgroundColor': '#FFF8E1',
                            'border': '1px solid #FFE082',
                            'borderRadius': '4px',
                            'color': '#FF8F00'
                        }
                    )
                ]))
        
        return plot_divs
    else:
        return html.Div(style=card_style, children=[
            html.P("Please select columns to generate visualizations.", 
                  style={'textAlign': 'center', 'color': '#666'})
        ])

def render_ver_bar_chart(selected_columns):
    if loaded_data is None or len(selected_columns) < 2 or len(selected_columns) > 4:
        raise ValueError("Select between two to four columns for bar chart.")

    group_column = selected_columns[0]
    sum_columns = selected_columns[1:]

    grouped_data = loaded_data.groupby(group_column)[sum_columns].sum().reset_index()

    fig = px.bar(grouped_data, x=group_column, y=sum_columns,
                 orientation='v',
                 title=f'Distribution of {", ".join(sum_columns)} by {group_column}',
                 labels={col: f"Total {col}" for col in sum_columns},
                 category_orders={group_column: sorted(grouped_data[group_column].unique())},
                 barmode='group')

    fig.update_layout(height=600, width=None)
    fig.update_traces(marker=dict(line=dict(width=0.8)))
    return fig

def render_line_chart(selected_columns):
    if loaded_data is None or len(selected_columns) < 2 or len(selected_columns) > 4:
        raise ValueError("Select between two to four columns for line chart.")

    x_column, *sum_columns = selected_columns

    grouped_data = loaded_data.groupby(x_column)[sum_columns].sum().reset_index()
    
    fig = px.line(grouped_data, x=x_column, y=sum_columns, 
                  title=f'Trend of {", ".join(sum_columns)} over {x_column}',
                  labels={x_column: f'{x_column}', 'value': 'Total'},
                  markers=True)
    fig.update_layout(height=600, width=None)
    return fig

def render_scatter_plot(selected_columns):
    if loaded_data is None or len(selected_columns) != 2:
        raise ValueError("Select exactly two columns for scatter plot.")

    x_column, y_column = selected_columns

    fig = px.scatter(loaded_data, x=x_column, y=y_column,
                     title=f'Relationship between {y_column} and {x_column}',
                     labels={x_column: f"{x_column}", y_column: f"{y_column}"},
                     trendline="ols")  # Add trend line for better analysis
    fig.update_layout(height=600, width=None)
    return fig

def generate_pie_chart(selected_columns, filters=None):
    if loaded_data is None:
        raise ValueError("No data loaded")
        
    if len(selected_columns) < 2:
        raise ValueError("Select at least two columns for pie chart (category and value)")
        
    filtered_data = loaded_data.copy()
    if filters is not None:
        for column, value in filters.items():
            filtered_data = filtered_data[filtered_data[column] == value]

    fig = px.pie(filtered_data, names=selected_columns[0], values=selected_columns[1],
                 title=f'Distribution of {selected_columns[1]} by {selected_columns[0]}')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=600, width=None)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


