import logging
import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import plotly.express as px
import base64
import io
import json
import google.generativeai as genai
import os
import requests
import re  # Added for regex pattern matching

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()# Configure Google Gemini API
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

loaded_data = None
data_summary = None

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
    
    # AI Suggestions Section (initially hidden)
    html.Div(id='ai-suggestions-section', style={'display': 'none'}),
    
    html.Div(id='output-plot')
])

def get_visualization_recommendations(headers):
    """Get AI recommendations for which columns to visualize together"""
    prompt = f"""Analyze the following dataset with columns {headers} and provide visualization recommendations.
    Please provide exactly ONE visualization recommendation in the following format:
    
    Visualization Recommendations
    x-axis: [column name]
    y-axis: [column name]
    
    Important: Only include column names that exist in the dataset. Do not make up column names.
    The column names should be exactly as they appear in the headers list."""
    
    try:
        # Use a different model for visualization recommendations
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        analysis = response.text
        
        # Parse the visualization recommendations
        viz_recommendations = []
        if "Visualization Recommendations" in analysis:
            viz_section = analysis.split("Visualization Recommendations")[1].strip()
            pattern = r'x-axis:\s*([^\n]+)[\s\n]*y-axis:\s*([^\n]+)'
            matches = re.findall(pattern, viz_section, re.IGNORECASE)
            
            for x_col, y_col in matches:
                x_col = x_col.strip("'\" \n")
                y_col = y_col.strip("'\" \n")
                if x_col and y_col:
                    viz_recommendations.append({
                        'x_column': x_col,
                        'y_column': y_col,
                        'description': f"Plot {y_col} against {x_col}"
                    })
        
        return viz_recommendations
    except Exception as e:
        logger.error(f"Error generating visualization recommendations: {str(e)}")
        return []

def parse_contents(contents, filename):
    global loaded_data, data_summary
    content_type, content_string = contents.split(',')

    decoded = pd.read_excel(io.BytesIO(base64.b64decode(content_string)))
    loaded_data = decoded

    # Generate data summary for AI
    data_summary = generate_data_summary(decoded)
    
    columns = list(decoded.columns)
    
    # Get visualization recommendations
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
                html.H5("Quick Visualization Recommendation", 
                       style={'color': COLORS['ai'], 'marginBottom': '10px'}) if viz_recommendations else None,
                html.Div([
                    html.Div(style={
                        'padding': '15px',
                        'borderRadius': '5px',
                        'border': f'1px solid {COLORS["border"]}',
                        'marginBottom': '10px',
                        'backgroundColor': '#f9f9f9'
                    }, children=[
                        html.Div([
                            html.Strong("Recommended Plot: "),
                            html.Span(f"{rec['y_column']} vs {rec['x_column']}")
                        ]),
                        html.Button(
                            "Apply this recommendation",
                            id={'type': 'apply-quick-rec', 'index': i},
                            style={
                                'backgroundColor': COLORS['ai'],
                                'color': 'white',
                                'border': 'none',
                                'padding': '8px 12px',
                                'borderRadius': '4px',
                                'cursor': 'pointer',
                                'fontSize': '0.9em',
                                'marginTop': '10px'
                            }
                        )
                    ]) for i, rec in enumerate(viz_recommendations)
                ]) if viz_recommendations else None,
            ]),
            
            html.Hr(style={'margin': '20px 0'}) if viz_recommendations else None,
            
            html.Button(
                children=[
                    html.I(className='fas fa-magic', style={'marginRight': '8px'}),
                    'Generate AI Suggestions'
                ],
                id='generate-ai-suggestions',
                style=ai_button_style
            ),
            # Use dcc.Loading for better loading state management
            dcc.Loading(
                id="loading-suggestions",
                type="circle",
                color=COLORS['ai'],
                children=html.Div(id='ai-suggestions-output')
            ),
        ])
    ])

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

def get_ai_suggestions(data_summary):
    """Get AI suggestions for column combinations to visualize"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""
    You are a data visualization expert. Based on the following dataset summary, suggest 3-5 interesting column combinations 
    that would make meaningful visualizations. For each suggestion, explain why these columns might have an interesting relationship.
    
    Dataset Summary:
    - Columns: {data_summary['columns']}
    - Shape: {data_summary['shape']}
    - Numeric columns: {data_summary['numeric_columns']}
    - Categorical columns: {data_summary['categorical_columns']}
    - Datetime columns: {data_summary['datetime_columns']}
    
    For each suggestion:
    1. List the specific columns to visualize together (2-4 columns)
    2. Suggest which visualization type would be best (bar, line, scatter, or pie)
    3. Explain why this combination might reveal interesting insights
    
    Format your response as a list of suggestions, each with columns, chart type, and explanation.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI suggestions: {str(e)}"

def parse_ai_suggestions(suggestions_text):
    """Parse the AI suggestions into a structured format"""
    # This is a simple parser - in production you might want a more robust solution
    suggestions = []
    current_suggestion = None
    
    for line in suggestions_text.split('\n'):
        line = line.strip()
        continue
            
        # Look for numbered suggestions
        if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) and len(line) > 3:
            # Save previous suggestion if it exists
            if current_suggestion:
                suggestions.append(current_suggestion)
            
            # Start new suggestion
            current_suggestion = {
                'title': line[2:].strip(),
                'columns': [],
                'chart_type': '',
                'explanation': ''
            }
        elif current_suggestion:
            # Look for columns section
            if 'columns:' in line.lower() or 'column:' in line.lower():
                cols = line.split(':', 1)[1].strip()
                current_suggestion['columns'] = [c.strip() for c in cols.split(',')]
            
            # Look for chart type
            elif 'chart:' in line.lower() or 'visualization:' in line.lower() or 'type:' in line.lower():
                chart_type = line.split(':', 1)[1].strip().lower()
                if 'bar' in chart_type:
                    current_suggestion['chart_type'] = 'vertical_bar'
                elif 'line' in chart_type:
                    current_suggestion['chart_type'] = 'line'
                elif 'scatter' in chart_type:
                    current_suggestion['chart_type'] = 'scatter'
                elif 'pie' in chart_type:
                    current_suggestion['chart_type'] = 'pie'
                else:
                    current_suggestion['chart_type'] = 'vertical_bar'  # default
            
            # Add to explanation
            elif 'explanation:' in line.lower() or 'why:' in line.lower():
                current_suggestion['explanation'] = line.split(':', 1)[1].strip()
            elif current_suggestion.get('explanation'):
                current_suggestion['explanation'] += ' ' + line
    
    # Add the last suggestion
    if current_suggestion:
        suggestions.append(current_suggestion)
    
    return suggestions

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
    [Output('ai-suggestions-output', 'children')],
    [Input('generate-ai-suggestions', 'n_clicks')],
    [State('ai-suggestions-output', 'children')],
    prevent_initial_call=True
)
def generate_and_process_suggestions(n_clicks, current_output):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    if data_summary is None:
        return [html.P("Please upload data first to get AI suggestions.", 
                      style={'color': 'red'})]
    
    # Get AI suggestions
    suggestions_text = get_ai_suggestions(data_summary)
    suggestions = parse_ai_suggestions(suggestions_text)
    
    # Create suggestion cards
    suggestion_cards = []
    for i, suggestion in enumerate(suggestions):
        columns = suggestion.get('columns', [])
        chart_type = suggestion.get('chart_type', 'vertical_bar')
        explanation = suggestion.get('explanation', '')
        title = suggestion.get('title', f'Suggestion {i+1}')
        
        # Only include valid suggestions
        if columns and len(columns) >= 2:
            # Filter to only include columns that exist in the dataset
            valid_columns = [col for col in columns if col in loaded_data.columns]
            
            if len(valid_columns) >= 2:
                suggestion_cards.append(html.Div(style={
                    'padding': '15px',
                    'borderRadius': '5px',
                    'border': f'1px solid {COLORS["border"]}',
                    'marginBottom': '10px',
                    'backgroundColor': '#f9f9f9'
                }, children=[
                    html.H5(title, style={'color': COLORS['primary'], 'marginBottom': '10px'}),
                    html.Div([
                        html.Strong("Columns: "),
                        html.Span(", ".join(valid_columns))
                    ]),
                    html.Div([
                        html.Strong("Chart Type: "),
                        html.Span(chart_type.replace('_', ' ').title())
                    ]),
                    html.Div([
                        html.Strong("Explanation: "),
                        html.Span(explanation)
                    ], style={'marginBottom': '10px'}),
                    html.Button(
                        "Apply this suggestion",
                        id={'type': 'apply-suggestion', 'index': i},
                        style={
                            'backgroundColor': COLORS['ai'],
                            'color': 'white',
                            'border': 'none',
                            'padding': '8px 12px',
                            'borderRadius': '4px',
                            'cursor': 'pointer',
                            'fontSize': '0.9em'
                        }
                    )
                ]))
    
    if not suggestion_cards:
        return [html.P("Couldn't generate meaningful suggestions for this dataset. Try selecting columns manually.", 
                      style={'fontStyle': 'italic'})]
    
    return [html.Div([
        html.H5("AI-Generated Visualization Suggestions", 
               style={'color': COLORS['ai'], 'marginBottom': '15px', 'marginTop': '10px'}),
        html.Div(suggestion_cards)
    ])]

@app.callback(
    Output('column-dropdown', 'value'),
    [Input({'type': 'apply-suggestion', 'index': dash.dependencies.ALL}, 'n_clicks'),
     Input({'type': 'apply-quick-rec', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State({'type': 'apply-suggestion', 'index': dash.dependencies.ALL}, 'id'),
     State({'type': 'apply-quick-rec', 'index': dash.dependencies.ALL}, 'id')],
    prevent_initial_call=True
)
def apply_suggestion(suggestion_clicks, quick_rec_clicks, suggestion_ids, quick_rec_ids):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Get the ID of the clicked button
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    button_data = json.loads(button_id)
    button_type = button_data['type']
    button_index = button_data['index']
    
    if button_type == 'apply-suggestion':
        # Get the corresponding suggestion from the detailed AI suggestions
        suggestions_text = get_ai_suggestions(data_summary)
        suggestions = parse_ai_suggestions(suggestions_text)
        
        if button_index < len(suggestions):
            suggestion = suggestions[button_index]
            columns = suggestion.get('columns', [])
            
            # Filter to only include columns that exist in the dataset
            valid_columns = [col for col in columns if col in loaded_data.columns]
            
            if valid_columns:
                return valid_columns
    
    elif button_type == 'apply-quick-rec':
        # Get the corresponding quick recommendation
        viz_recommendations = get_visualization_recommendations(data_summary['columns'])
        
        if button_index < len(viz_recommendations):
            rec = viz_recommendations[button_index]
            columns = [rec['x_column'], rec['y_column']]
            
            # Filter to only include columns that exist in the dataset
            valid_columns = [col for col in columns if col in loaded_data.columns]
            
            if valid_columns:
                return valid_columns
    
    raise dash.exceptions.PreventUpdate

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


