from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = 'ash'
loaded_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global loaded_data
    file = request.files['file']
    if file:
        loaded_data = pd.read_excel(file)
        print(loaded_data.dtypes)

        columns = list(loaded_data.columns)
        return render_template('upload.html', columns=columns)
    return render_template('index.html', error="File not provided.")

@app.route('/plot', methods=['POST'])
def plot():
    if loaded_data is None:
        return render_template('upload.html', error="No data loaded.")
    
    selected_columns = request.form.getlist('columns')
    if not selected_columns:
        return render_template('upload.html', error="Select at least one column.")

    chart_type = request.form.get('chart_type', 'bar')

    try:
        if chart_type == 'bar':
            plotly_fig = render_bar_chart(selected_columns)
        elif chart_type == 'line':
            plotly_fig = render_ver_bar_chart(selected_columns)
        elif chart_type == 'pie':
            plotly_fig = generate_pie_chart(selected_columns)
        elif chart_type== 'scatter':
            plotly_fig = render_scatter_plot(selected_columns)
        else:
            raise ValueError("Invalid chart type.")
    except Exception as e:
        return render_template('upload.html', error=f"Error generating plot: {str(e)}")

    try:
        plotly_html = plotly_fig.to_html(full_html=False)
    except Exception as e:
        return render_template('upload.html', error=f"Error converting plot to HTML: {str(e)}")

    return render_template('plotly_template.html', plotly_html=plotly_html)


def render_bar_chart(selected_columns):
    if len(selected_columns) < 2 or len(selected_columns) > 4:
        return render_template('upload.html', error="Select between two to four columns for bar chart.")

    # Extract columns
    group_column = selected_columns[0]
    sum_columns = selected_columns[1:]

    grouped_data = loaded_data.groupby(group_column)[sum_columns].sum().reset_index()

    fig = px.bar(grouped_data, x=sum_columns, y=group_column,
                 orientation='h',
                 title=f'Horizontal Bar Chart of {", ".join(sum_columns)} by {group_column}',
                 labels={col: f"Total {col}" for col in sum_columns},
                 category_orders={group_column: sorted(grouped_data[group_column].unique())},
                 barmode='group') 

    fig.update_layout(height=600, width=1000, showlegend=True)
    fig.update_traces(marker=dict(line=dict(width=0.8)))
    return fig



def render_ver_bar_chart(selected_columns):
    if len(selected_columns) != 2:
        return render_template('upload.html', error="Select exactly two columns for bar chart.")

    group_column, sum_column = selected_columns

    grouped_data = loaded_data.groupby(group_column)[sum_column].sum().reset_index()

    fig = px.scatter(grouped_data, x=group_column, y=sum_column, 
                 title=f'Vertical Bar Chart of Total {sum_column} by {group_column}',
                 labels={sum_column: f"Total {sum_column}"})
    return fig

def render_scatter_plot(selected_columns):
    if len(selected_columns) != 2:
        return render_template('upload.html', error="Select exactly two columns for scatter plot.")

    x_column, y_column = selected_columns

    fig = px.scatter(loaded_data, x=x_column, y=y_column,
                     title=f'Scatter Plot: {y_column} vs {x_column}',
                     labels={x_column: f"{x_column}", y_column: f"{y_column}"})
    
    return fig

def generate_pie_chart(selected_columns):
    fig = px.pie(loaded_data, names=selected_columns[0], values=selected_columns[1], 
                 title=f'Pie Chart for {selected_columns[0].capitalize()} ({selected_columns[1].capitalize()})')
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)
