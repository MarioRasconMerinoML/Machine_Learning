def plot_3D_predictions(y_test, predictions, model_dir, model_name):
    import matplotlib as mpl
	import matplotlib.pyplot as plt
	mpl.rc('axes', labelsize=14)
	mpl.rc('xtick', labelsize=12)
	mpl.rc('ytick', labelsize=12)

	# Plotly
	from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
	import plotly.figure_factory as ff

	pd.options.display.max_columns = None
	from plotly.subplots import make_subplots
	import plotly.graph_objects as go
	pd.options.plotting.backend = "plotly"
	from IPython.display import display,HTML
    # It plots the 3D Scatter cloud of points with errors per variable
    z=y_test[:,2],
    fig = go.Figure(data=[go.Scatter3d(
        x=y_test[:,0] - predictions[:,0],
        y=y_test[:,1] - predictions[:,1],
        z=y_test[:,2] - predictions[:,2],
        name='Errors',
        mode='markers',
        marker=dict(
            size=8,
            color=z,                # set color to an array/list of desired values
            colorscale='turbo',   # choose a colorscale
            opacity=0.5
        ),
        showlegend = True
    )])

    fig.update_layout(
        title_text="Multi Output Regression Model Predictions for " + model_name,
                     scene=dict(
                         xaxis_title='SFC_kg_Ns',
                         yaxis_title='FN[N]',
                         zaxis_title='WH2[g/s]')
                     )
    title = "Multi Output Regression Model Errors for " + model_name
    fig.update_layout(title = title)
    fig.show()
    fig.write_html(os.path.join(model_dir,model_name) + title + '.html')