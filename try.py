from flask import Flask, render_template, request
import pandas as pd
import base64
import matplotlib.pyplot as plt
import io
from flask import session


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
    plt.figure(figsize=(36, 18))
    unique_values = loaded_data[selected_columns[0]].unique()

    for value in unique_values:
        subset = loaded_data[loaded_data[selected_columns[0]] == value]
        plt.bar(str(value), subset[selected_columns[1:]].mean())

    plt.title(f'Bar Chart of Average Values for {selected_columns[0]}')
    plt.xlabel(selected_columns[0])
    plt.ylabel('Average Value')
    plt.legend(loc='upper right')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()

    return render_chart(img)


def render_line_chart(selected_columns):
    plt.figure(figsize=(16, 10))
    unique_values = loaded_data[selected_columns[0]].unique()

    for value in unique_values:
        subset = loaded_data[loaded_data[selected_columns[0]] == value]
        plt.scatter(str(value), subset[selected_columns[1:]].mean(), label=str(value))

    plt.title(f'Line Chart of Average Values for {selected_columns[0]}')
    plt.xlabel(selected_columns[0])
    plt.ylabel('Average Value', loc='center')
    plt.legend(loc='upper right')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()

    return render_chart(img)

def generate_pie_chart(selected_columns):
    column_name = selected_columns[0]
    value_column = selected_columns[1]  

    grouped_data = loaded_data.groupby(column_name)[value_column].sum()

    labels = grouped_data.index.tolist()
    sizes = grouped_data.values.tolist()
    colors = plt.cm.Paired.colors[:len(labels)]
    explode = tuple([0.1] * len(labels))

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(f'Pie Chart for {column_name.capitalize()} ({value_column.capitalize()})')

    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close()

    return render_chart(img)


def render_chart(img):
    img_str = "data:image/png;base64," + base64.b64encode(img.getvalue()).decode('utf-8')
    return render_template('results.html', img=img_str)

if __name__ == '__main__':
    app.run(debug=True)
