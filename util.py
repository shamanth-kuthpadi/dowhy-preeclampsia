import pandas as pd
import networkx as nx
import numpy as np
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
from pydot import graph_from_dot_data
import dowhy.gcm.independence_test
from dowhy.gcm.independence_test import generalised_cov_measure as gvm
import dowhy.gcm.ml
from dowhy.gcm.ml import regression as rg
from sklearn.ensemble import GradientBoostingRegressor

# loader from txt file to pandas dataframe
def load_from_txt(filepath):
    data = pd.read_csv(filepath, sep='\t')
    return data

# rename nodes in a networkx if they are integers
def rename_nodes(G, new_node_names):
    mapping = {old_name: new_name for old_name, new_name in zip(G.nodes(), new_node_names)}
    G = nx.relabel_nodes(G, mapping)

    return G

# create a networkx from a graphviz
def gviz_to_nx(dot):
    dot.save('temp.dot')
    G = nx.drawing.nx_agraph.read_dot('temp.dot')
    return G

# calculate SHD
def shd(actual, predicted):
    actual_adj_mat = nx.adjacency_matrix(actual).todense()
    predicted_adj_mat = nx.adjacency_matrix(predicted).todense()
    predicted_adj_mat = (predicted_adj_mat != 0).astype(int)
    diff_mat = actual_adj_mat - predicted_adj_mat
    diff_mat = np.abs(diff_mat)

    return np.sum(diff_mat)

# calculate the Forbenius norm
def fnorm(actual, predicted):
    actual_adj_mat = nx.adjacency_matrix(actual).todense()
    predicted_adj_mat = nx.adjacency_matrix(predicted).todense()

    diff_mat = actual_adj_mat - predicted_adj_mat
    dotted = np.dot(diff_mat.T, diff_mat)
    trace = np.trace(dotted)

    return np.sqrt(trace)

# display a GeneralGraph
def disp_graph(graph, labels=None):
    pyd = GraphUtils.to_pydot(graph, labels=labels)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def disp_graph_nx(G, fig_size=(10, 10)):
    """
    Displays a NetworkX graph using pydot for rendering.
    
    Parameters:
    - G (networkx.Graph): The graph to display.
    - fig_size (tuple): Size of the matplotlib figure, default is (10, 10).
    """
    # Convert NetworkX graph to a PyDot graph
    pydot_graph = nx.nx_pydot.to_pydot(G)
    # Render graph to PNG
    tmp_png = pydot_graph.create_png()
    # Read the PNG image
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    # Plot with adjusted figure size
    plt.figure(figsize=fig_size)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

# GeneralGraph to networkx
def genG_to_nx(G, labels):
    pyd = GraphUtils.to_pydot(G, labels=labels)
    dot_data = pyd.to_string()
    pydot_graph = graph_from_dot_data(dot_data)[0]
    predicted_graph = nx.drawing.nx_pydot.from_pydot(pydot_graph)
    predicted_graph = nx.DiGraph(predicted_graph)
    predicted_graph = rename_nodes(predicted_graph, labels)
    return predicted_graph

def create_gradient_boost_regressor(**kwargs):
    return rg.SklearnRegressionModel(GradientBoostingRegressor(**kwargs))
def gcm(X, Y, Z=None):
    return gvm.generalised_cov_based(X, Y, Z=Z, prediction_model_X=create_gradient_boost_regressor,
                                 prediction_model_Y=create_gradient_boost_regressor)

def preproc(betas_df, pds_df):
    betas_df_t = betas_df.set_index('Unnamed: 0').T.reset_index().rename(columns={'index': 'sampleid'})

    merged_df = pd.merge(pds_df, betas_df_t, on='sampleid')

    def clean_column_names(col_name):
        if "::" in col_name:
            return col_name.split("::")[0]
        return col_name

    merged_df.columns = [clean_column_names(col) for col in merged_df.columns]
    df = merged_df.drop(columns=['Unnamed: 0', 'sampleid', 'Gestwk', 'Ethnicity', 'Dataset', 'Babygender'])
    df['Group'] = df['Group'].map({'Control': 0, 'Preeclampsia': 1})

    return df