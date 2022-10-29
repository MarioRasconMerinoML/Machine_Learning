def plot_errors_in_predictions(y_test, predictions, model_dir, model):
    # It plots the histogram of errors (y_test - y_pred) in %
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

    # Initialize figure with subplots
    fig = make_subplots(
        rows=4, cols=1,
        vertical_spacing = 0.1,
        shared_xaxes=True,
    )

    sfc = go.Histogram(x=y_test[:,0] - predictions[:,0], histnorm='percent', name = 'A error')
    fn = go.Histogram(x=y_test[:,1] - predictions[:,1], histnorm='percent',name = 'B error')
    wh2 = go.Histogram(x=y_test[:,2] - predictions[:,2], histnorm='percent',name = 'C error')
    pt3 = go.Histogram(x=y_test[:,3] - predictions[:,3], histnorm='percent',name = 'D error')

    fig.update_layout(title_text='Errors Distribution %')

    fig.append_trace(sfc, 1, 1)
    fig.append_trace(fn, 2, 1)
    fig.append_trace(wh2, 3, 1)
    fig.append_trace(pt3, 4, 1)
    
    title = 'Errors Distribution in % for ' + model.name
    fig.update_layout(title = title)
    fig.write_html(os.path.join(model_dir,model.name) + title + '.html')