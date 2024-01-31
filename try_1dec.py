from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
from io import BytesIO
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'asdd'
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

    if chart_type == 'bar':
        return render_bar_chart(selected_columns)
    elif chart_type == 'line':
        return render_line_chart(selected_columns)
    elif chart_type == 'pie':
        return generate_pie_chart(selected_columns)

    return render_template('upload.html', error="Invalid chart type.")

def render_bar_chart(selected_columns):
    if len(selected_columns) != 2:
        return px.scatter()  # You can customize this empty chart as needed

    group_column, sum_column = selected_columns

    grouped_data = loaded_data.groupby(group_column)[sum_column].sum().reset_index()

    num_unique_values = len(grouped_data[group_column].unique())

    fig = px.bar(grouped_data, x=group_column, y=sum_column,
                 labels={group_column: group_column.capitalize(), sum_column: f'Total {sum_column}'},
                 title=f'Total {sum_column} by {group_column.capitalize()}')

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=grouped_data[group_column].tolist(),
            ticktext=grouped_data[group_column].tolist(),
            tickangle=-45,
            type='category',
            range=[0, num_unique_values]
        ),
        yaxis=dict(title=f"Total {sum_column}"),
        bargap=0.2
    )

    return render_chart(fig)


def render_line_chart(selected_columns):
    if len(selected_columns) != 2:
        return render_template('upload.html', error="Select exactly two columns for the line chart.")

    x_column, y_column = selected_columns

    grouped_data = loaded_data.groupby(x_column)[y_column].mean().reset_index()

    # Use Plotly Express for line chart
    fig = px.line(grouped_data, x=x_column, y=y_column,
                  labels={x_column: x_column.capitalize(), y_column: f'Average {y_column}'},
                  title=f'Line Chart of Average Values for {x_column.capitalize()}')

    # Add dots for column markings on the x-axis
    fig.update_traces(mode='lines+markers')

    return render_chart(fig)


def generate_pie_chart(selected_columns):
    # Use Plotly Express for pie chart
    fig = px.pie(loaded_data, names=selected_columns[0], values=selected_columns[1], 
                 title=f'Pie Chart for {selected_columns[0].capitalize()} ({selected_columns[1].capitalize()})')

    return render_chart(fig)

def render_chart(fig):
    img = BytesIO()
    fig.write_image(img, format='png')
    img.seek(0)
    img_str = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode('utf-8')
    return render_template('results.html', img=img_str)

if __name__ == '__main__':
    app.run(debug=True)
