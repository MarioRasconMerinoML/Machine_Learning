def plot_3D_predictions(y_test, predictions, model_dir, model):
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
    # It plots the 3D Scatter cloud of points, test points and prediction points
    z=y_test[:,2],
    fig = go.Figure(data=[go.Scatter3d(
        x=y_test[:,0],
        y=y_test[:,1],
        z=y_test[:,2],
        name='Test_Data',
        mode='markers',
        marker=dict(
            size=8,
            color=z,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.5
        ),
        showlegend = True
    )])

    fig.add_trace(go.Scatter3d(
        x= predictions[:,0],
        y= predictions[:,1],
        z= predictions[:,2],
        name='Multi-Regression fit for ' + model.name,
        mode = 'markers', 
        marker=dict(
            size=6,
            color='rgba(255, 0, 0, .9)',# set color to an array/list of desired values
            opacity=0.4),
            showlegend = True,)
                 )
    fig.update_layout(
        title_text="Multi Output Regression Model Predictions for " + model.name,
                     scene=dict(
                         xaxis_title='SFC',
                         yaxis_title='FN[N]',
                         zaxis_title='WH2[g/s]')
                     )
    title = "Multi Output Regression Model Predictions for " + model.name
    fig.update_layout(title = title)
    fig.write_html(os.path.join(model_dir,model.name) + title + '.html')