import numpy as np

from CHAID import Tree


class CHAIDDecisionTree:
    """

    Class for unify CHAID algorithm into auto-ml framework

    """
    def __init__(self,
                 alpha_merge: float = 0.05,
                 max_depth: int = 2,
                 min_parent_node_size: int = 30,
                 min_child_node_size: int = 0,
                 split_threshold: int = 0,
                 ):
        """
        Parameters
        ----------
        alpha_merge
        max_depth
        min_parent_node_size
        min_child_node_size
        split_threshold
        """
        self.model: Tree = None
        self.alpha_merge: float = alpha_merge if alpha_merge > 0 else 0.05
        self.max_depth: int = max_depth if max_depth > 0 else 2
        self.min_parent_node: int = min_parent_node_size if min_parent_node_size > 0 else 30
        self.min_child_node: int = min_child_node_size if min_child_node_size > 0 else 0
        self.split_threshold: int = split_threshold if split_threshold > 0 else 0

    def predict(self, x: np.array = None) -> np.array:
        """
        Predict

        Parameters
        ----------
        x: np.array
            Test data set

        Returns
        -------
        np.array: Predictions
        """
        return self.model.model_predictions()

    def train(self, x: np.ndarray, y: np.array):
        """
        Train CHAID classifier

        Parameters
        ----------
        x: np.array
            Train data set

        y: np.array
            Train target data
        """
        self.model = Tree.from_numpy(ndarr=x,
                                     arr=y,
                                     alpha_merge=self.alpha_merge,
                                     max_depth=self.max_depth,
                                     min_parent_node_size=self.min_parent_node,
                                     min_child_node_size=self.min_child_node,
                                     split_threshold=self.split_threshold,
                                     dep_variable_type='categorical'
                                     )
