from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.search.FCMBased.lingam.utils import make_dot
from util import *
import dowhy.gcm.falsify
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.falsify import apply_suggestions
from dowhy import CausalModel
import cdt
from cdt.causality.graph import PC
from cdt.causality.graph import CCDr
from cdt.causality.graph import GES

cdt.SETTINGS.rpath = '/usr/local/bin/Rscript'
cdt.SETTINGS.GPU = 1

class EstimateEffect:
    def __init__(self, data):
        self.data = data
        self.graph = None
        self.graph_ref = None
        self.model = None
        self.estimand = None
        self.estimate = None
        self.est_ref = None
    
    # For now, the only prior knowledge that the prototype will allow is required/forbidden edges
    # pk must be of the type => {'required': [list of edges to require], 'forbidden': [list of edges to forbid]}
    def find_causal_graph(self, algo='pc', pk=None):
        df = self.data.to_numpy()
        labels = list(self.data.columns)
        try:
            match algo:
                case 'pc':
                    cg = pc(data=df, show_progress=False, node_names=labels)
                    cg = pdag2dag(cg.G)
                    predicted_graph = genG_to_nx(cg, labels)
                    self.graph = predicted_graph
                case 'ges':
                    cg = ges(X=df, node_names=labels)
                    cg = pdag2dag(cg['G'])
                    predicted_graph = genG_to_nx(cg, labels)
                    self.graph = predicted_graph
                case 'icalingam':
                    model = lingam.ICALiNGAM()
                    model.fit(df)
                    pyd_lingam = make_dot(model.adjacency_matrix_, labels=labels)
                    pyd_lingam = pyd_lingam.pipe(format='dot').decode('utf-8')
                    pyd_lingam = (pyd_lingam,) = graph_from_dot_data(pyd_lingam)
                    dot_data_lingam = pyd_lingam.to_string()
                    pydot_graph_lingam = graph_from_dot_data(dot_data_lingam)[0]
                    predicted_graph = nx.drawing.nx_pydot.from_pydot(pydot_graph_lingam)
                    predicted_graph = nx.DiGraph(predicted_graph)
                    self.graph = predicted_graph
                case 'CCDr':
                    model = PC()
                    predicted_graph = model.predict(self.data)
                    self.graph = predicted_graph
            
            if pk is not None:
                # ensuring that pk is indeed of the right type
                if not isinstance(pk, dict):
                    print(f"Please ensure that the prior knowledge is of the right form")
                    raise
                # are there any edges to require
                if 'required' in pk.keys():
                    eb = pk['required']
                    self.graph.add_edges_from(eb)
                # are there any edges to remove
                if 'forbidden' in pk.keys():
                    eb = pk['forbidden']
                    self.graph.remove_edges_from(eb)
        
        except Exception as e:
            print(f"Error in creating causal graph: {e}")
            raise

        return self.graph

    # What if user already has a graph they would like to input
    def input_causal_graph(self, graph):
        self.graph = graph

    def refute_cgm(self, n_perm=100, indep_test=gcm, cond_indep_test=gcm, apply_sugst=True, show_plt=False):
        try:
            result = falsify_graph(self.graph, self.data, n_permutations=n_perm,
                                  independence_test=indep_test,
                                  conditional_independence_test=cond_indep_test, plot_histogram=show_plt)
            self.graph_ref = result
            if apply_sugst is True:
                self.graph = apply_suggestions(self.graph, result)
            
        except Exception as e:
            print(f"Error in refuting graph: {e}")
            raise

        return self.graph
    
    def create_model(self, treatment, outcome):
        model_est = CausalModel(
                data=self.data,
                treatment=treatment,
                outcome=outcome,
                graph=self.graph
            )
        self.model = model_est
        return self.model

    def identify_effect(self, method=None):
        try:
            if method is None:
                identified_estimand = self.model.identify_effect()
            else:
                identified_estimand = self.model.identify_effect(method=method)

            self.estimand = identified_estimand
        except Exception as e:
            print(f"Error in identifying effect: {e}")
            raise

        print("Note that you can also use other methods for the identification process. Below are method descriptions taken directly from DoWhy's documentation")
        print("maximal-adjustment: returns the maximal set that satisfies the backdoor criterion. This is usually the fastest way to find a valid backdoor set, but the set may contain many superfluous variables.")
        print("minimal-adjustment: returns the set with minimal number of variables that satisfies the backdoor criterion. This may take longer to execute, and sometimes may not return any backdoor set within the maximum number of iterations.")
        print("exhaustive-search: returns all valid backdoor sets. This can take a while to run for large graphs.")
        print("default: This is a good mix of minimal and maximal adjustment. It starts with maximal adjustment which is usually fast. It then runs minimal adjustment and returns the set having the smallest number of variables.")
        return self.estimand
    
    def estimate_effect(self, method_cat='backdoor.linear_regression', ctrl_val=0, trtm_val=1):
        estimate = None
        try:
            match method_cat:
                case 'backdoor.linear_regression':
                    estimate = self.model.estimate_effect(self.estimand,
                                                  method_name=method_cat,
                                                  control_value=ctrl_val,
                                                  treatment_value=trtm_val,
                                                  confidence_intervals=True,
                                                  test_significance=True)
                # there are other estimation methods that I can add later on, however parameter space will increase immensely
            self.estimate = estimate
        except Exception as e:
            print(f"Error in estimating the effect: {e}")
            raise
        
        print("Note that it is ok for your treatment to be a continuous variable, DoWhy automatically discretizes at the backend.")
        return self.estimate
    
    # should give a warning to users if the estimate is to be refuted

    def refute_estimate(self,  method_name="placebo_treatment_refuter", placebo_type='permute', subset_fraction=0.9):
        ref = None
        try:
            match method_name:
                case "placebo_treatment_refuter":
                    ref = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name,
                        placebo_type=placebo_type
                    )
                
                case "random_common_cause":
                    ref = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name
                    )
                case "data_subset_refuter":
                    ref = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name,
                        subset_fraction=subset_fraction
                    )
                case "ALL":
                    ref_placebo = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name,
                        placebo_type=placebo_type
                    )
                    ref_rand_cause = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name
                    )
                    ref_subset = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name,
                        subset_fraction=subset_fraction
                    )
                    ref = [ref_placebo, ref_rand_cause, ref_subset]
            if not isinstance(ref, list) and ref.refutation_result['is_statistically_significant']:
                print("Please make sure to take a revisit the pipeline as the refutation p-val is significant: ", ref.refutation_result['p_value'])
    
            self.est_ref = ref
        
        except Exception as e:
            print(f"Error in refuting estimate: {e}")
            raise
            
        return self.est_ref
    
    def get_all_information(self):
        return {'graph': self.graph, 
                'graph_refutation_res': self.graph_ref,
                'estimand_expression': self.estimand,
                'effect_estimate': self.estimate,
                'estimate_refutation_res': self.est_ref
                }
    
