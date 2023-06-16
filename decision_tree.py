import pandas as pd

from pandas.core.frame import DataFrame
from pandas.api.types import is_numeric_dtype
from copy import deepcopy
from dataclasses import dataclass, astuple


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        return isinstance(self.value, Leaf)

    # update somehow put in desctree class
    def prediction(self):
        return self.value.df["label"].mode().values[0]

    def __str__(self, indent=0):
        tab = "    "

        if self.is_leaf():
            return self.prediction()
        else:
            _, f, s = astuple(self.value)
            sym = "=" if isinstance(s, str) else "<="

            return f"""(
{(indent+1) * tab}{f} {sym} {s}?
{(indent+1) * tab}Y->{self.left.__str__(indent+1)}
{(indent+1) * tab}N->{self.right.__str__(indent+1)}
{indent * tab})"""


@dataclass
class Leaf:
    df: DataFrame
    gini_impurity: float


@dataclass
class Internal:
    df: DataFrame
    feature: str
    split: ...


def height(node):
    if node is None:
        return 0

    left_height = height(node.left)
    right_height = height(node.right)

    return max(left_height, right_height) + 1


def truncate(node, height):
    if node is None:
        return None

    if height == 1:
        if node.is_leaf():
            return node
        else:
            return Node(Leaf(node.value.df, leaf_gini_imp(node.value.df)))

    node.left = truncate(node.left, height - 1)
    node.right = truncate(node.right, height - 1)
    return node


def leaf_gini_imp(df):
    ret = 1
    ps = df["label"].value_counts(normalize=True)
    for p in ps:
        ret -= pow(p, 2)
    return ret


def predict_row(row, dtree):
    while not dtree.is_leaf():
        _, feature, split = astuple(dtree.value)
        cond = None

        if is_numeric_dtype(split):
            cond = row[feature] <= split
        else:
            cond = row[feature] == split

        if cond:
            dtree = dtree.left
        else:
            dtree = dtree.right
    return dtree.prediction()


def weighted_average_imp(tree):
    leaves = 0

    def total(node):
        nonlocal leaves

        if node.is_leaf():
            leaves += 1
            return node.value.gini_impurity * len(node.value.df)
        else:
            return total(node.left) + total(node.right)

    avg = total(tree) / len(tree.value.df)
    return avg, leaves


def split_df(df, feature, split):
    if isinstance(split, str):
        return df[df[feature] == split], df[df[feature] != split]
    # if not string must be numeric
    return df[df[feature] <= split], df[df[feature] > split]


class DecisionTree:
    def __init__(self, data, max_height=999, min_split_sz=2, ccp_alpha=0):
        self.df = data
        self.max_height = max_height
        self.min_split_sz = min_split_sz
        self.ccp_alpha = ccp_alpha

    def best_split(self, df):
        imps = {}

        df_len = len(df)
        for feature, col in df.drop("label", axis=1).items():
            for split in col.unique():
                left_data, right_data = split_df(df, feature, split)

                left_imp = leaf_gini_imp(left_data)
                right_imp = leaf_gini_imp(right_data)

                # weighted avg of node impurities
                imp = (len(left_data) * left_imp + len(right_data) * right_imp) / df_len

                imps[(feature, split)] = imp

        return min(imps, key=imps.get)

    def create_node(self, data):
        imp = leaf_gini_imp(data)
        if imp != 0 and len(data) > self.min_split_sz:
            feature, split = self.best_split(data)
            return Node(Internal(data, feature, split))
        return Node(Leaf(data, imp))

    def visit(self, dtree, height=1):
        if height == self.max_height or dtree.is_leaf():
            return

        df, feature, split = astuple(dtree.value)
        left_data, right_data = split_df(df, feature, split)

        dtree.left = self.create_node(left_data)
        dtree.right = self.create_node(right_data)

        self.visit(dtree.left, height + 1)
        self.visit(dtree.right, height + 1)

    def prune(self):
        cost_complexities = {}
        temp = deepcopy(self.dtree)

        # don't simplify to single node
        for h in range(height(temp), 1, -1):
            truncate(temp, h)
            cost, leaves = weighted_average_imp(temp)
            cc = cost + self.ccp_alpha * leaves
            cost_complexities[h] = cc

        min_ccp_height = min(cost_complexities, key=cost_complexities.get)
        truncate(self.dtree, min_ccp_height)

        return min_ccp_height

    def fit(self):
        # each node contains its split, if any, and the data associated with it prior to the split
        feature, split = self.best_split(self.df)
        self.dtree = Node(Internal(self.df, feature, split))
        self.visit(self.dtree)

    def predict(self, df):
        df["prediction"] = df.apply(lambda r: predict_row(r, self.dtree), axis=1)
        return df["prediction"]

    def confusion_matrix(self, df):
        return pd.crosstab(self.predict(df), df["label"])

    def accuracy(self, df):
        self.predict(df)
        acc = len(df[df["prediction"] == df["label"]]) / len(df)
        return round(acc * 100, 2)
