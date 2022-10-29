def plot_residuals_r2(y_test, predictions, model_dir, model):
    # Plot residuals for each predicted variable in plotly
    from sklearn.metrics import r2_score
	# To plot pretty figures
	#%matplotlib inline
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
	import plotly.express as px
    '''
    target[0,0] = Var A
    target[0,1] = Var B
    target[0,2] = Var C
    target[0,3] = Var D
    target[0,4] = Var E
    '''
    # Initialize figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[1, 1],
        row_heights=[1, 1],
        vertical_spacing = 0.2,
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    fig.add_trace(
        go.Scatter(x=y_test[:,0], y=predictions[:,0], 
                   name = 'SFC[kg/Ns]',
                   mode="markers",
                   hoverinfo="text",
                   showlegend=True, 
                   marker=dict(color='rgba(10, 250, 10, .9)', size=4, opacity=0.5)),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Ground truth SFC",row=1, col=1)
    fig.update_yaxes(title_text="Prediction SFC",row=1, col=1)
    fig.add_annotation(xref="x domain", yref="y domain", x=0.35, y=0.6,
                       text="R2 Score = " + str(r2_score(y_test[:,0], predictions[:,0])),row=1, col=1) 
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_test[:,0].min(), y0=y_test[:,0].min(),
        x1=y_test[:,0].max(), y1=y_test[:,0].max(), row=1, col=1
    )


    fig.add_trace(
        go.Scatter(x=y_test[:,1], y=predictions[:,1], name = 'FN[N]',
                   mode="markers",hoverinfo="text",
                   showlegend=True, marker=dict(color='rgba(0, 0, 200, .9)', size=4, opacity=0.6)),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Ground truth FN[N]",row=1, col=2)
    fig.update_yaxes(title_text="Prediction FN[N]",row=1, col=2)
    fig.add_annotation(xref="x domain", yref="y domain", x=0.35, y=0.6,
                       text="R2 Score = " + str(r2_score(y_test[:,1], predictions[:,1])),row=1, col=2) 

    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_test[:,1].min(), y0=y_test[:,1].min(),
        x1=y_test[:,1].max(), y1=y_test[:,1].max(), row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=y_test[:,2], y=predictions[:,2], name = 'WH2[g/s]',
                   mode="markers",hoverinfo="text",
                   showlegend=True, marker=dict(color='rgba(200, 0, 100, .9)', size=4, opacity=0.6)),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Ground truth WH2[g/S]",row=2, col=1)
    fig.update_yaxes(title_text="Prediction WH2[g/s]",row=2, col=1)
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.35,
        y=0.6,
        text="R2 Score = " + str(r2_score(y_test[:,2], predictions[:,2])),row=2, col=1) 

    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_test[:,2].min(), y0=y_test[:,2].min(),
        x1=y_test[:,2].max(), y1=y_test[:,2].max(), row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=y_test[:,3], y=predictions[:,3], name = ' Pt3[Pa]',
                   mode="markers",hoverinfo="text",
                   showlegend=True, marker=dict(color='rgba(255,165,0, .9)', size=4, opacity=0.7)),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Ground truth Pt3[Pa]",row=2, col=2)
    fig.update_yaxes(title_text="Prediction Pt3[Pa]",row=2, col=2)
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.35,
        y=0.6,
        text="R2 Score = " + str(r2_score(y_test[:,3], predictions[:,3])),row=2, col=2) 

    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y_test[:,3].min(), y0=y_test[:,3].min(),
        x1=y_test[:,3].max(), y1=y_test[:,3].max(), row=2, col=2
    )
    title = 'Residuals for ' + model.name
    fig.update_layout(title = title)
    fig.write_html(os.path.join(model_dir,model.name) + title + '.html')
