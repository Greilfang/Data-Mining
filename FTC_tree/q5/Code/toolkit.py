import numpy as np
def copy_ftc(node):
    cur_node = Node(category=node.category, time_stamp=node.time_stamp, amount=node.amount)
    if not node.children:
        return cur_node
    else:
        for v, child in node.children.items():
            new_node = copy_ftc(child)
            cur_node.children[v] = new_node
    return cur_node


def copy_tree(tree):
    cur_tree = FTC_Tree()
    cur_tree.root = copy_ftc(tree.root)
    return cur_tree


def choose_random_k(tree_names, k):
    np.random.shuffle(tree_names)
    return tree_names[:k]


def split_dataset(dataset, results):
    ds1, ds2 = None, None
    vipno1 = [vipno for vipno in results.keys() if results[vipno] == 0]
    vipno2 = [vipno for vipno in results.keys() if results[vipno] == 1]
    ds1, ds2 = dataset[dataset["vipno"].isin(vipno1)], dataset[dataset["vipno"].isin(vipno2)]
    return ds1, ds2

c = 1
num_k = 15

class Node:
    def __init__(self, category="root", time_stamp=None, amount=0):
        self.category = category
        self.children = dict()
        self.time_stamp = time_stamp
        self.amount = amount

    def insert_purchase_record(self, cur_level, amount, pluno, pur_time):
        if cur_level == 0:
            self.category = pluno
            self.amount = self.amount + amount
            return
        else:
            self.category = pluno[:-cur_level]
            self.amount = self.amount + amount
            next_category = pluno[:-cur_level + 1] if not cur_level == 1 else pluno
            assert next_category != ""
            if next_category not in self.children:
                self.children[next_category] = Node(category=next_category, time_stamp=pur_time, amount=0)
            self.children[next_category].insert_purchase_record(cur_level - 1, amount, pluno, pur_time)

    def union(self, another_node):
        union_amount = self.amount + another_node.amount
        cur_node = Node(category=self.category, time_stamp=None, amount=union_amount)
        uv = list(set(self.children.keys()).union(set(another_node.children.keys())))
        for v in uv:
            if v in self.children and v in another_node.children:
                new_node = self.children[v].union(another_node.children[v])
            else:
                new_node = copy_ftc(self.children[v]) if v in self.children else copy_ftc(another_node.children[v])
            cur_node.children[v] = new_node

        return cur_node

    def intersect(self, another_node):
        inter_amount = self.amount + another_node.amount
        cur_node = Node(category=self.category, time_stamp=None, amount=inter_amount)
        iv = list(set(self.children.keys()).intersection(set(another_node.children.keys())))
        for v in iv:
            new_node = self.children[v].intersect(another_node.children[v])
            cur_node.children[v] = new_node
        if cur_node.category == "root":
            cur_node.amount = sum(child.amount for child in self.children.values())

        return cur_node

    def get_amt_node(self):
        total_node, total_amt = 0, 0
        if not self.children:
            return total_node, total_amt

        for _, child in self.children.items():
            n_node, n_amt = child.get_amt_node()
            total_node = total_node + n_node + 1
            total_amt = total_amt + n_amt + child.amount
        return total_node, total_amt

    def get_max_amt(self):
        max_dist = float("-inf")
        for _, child in self.children.items():
            if child.amount > max_dist:
                max_dist = child.amount
        return max_dist

    def distance(self, another_node, children_depth, layer_dict, W):
        total_dist = 0
        another_children_amt_sum = another_node.get_children_amt_sum()
        if len(self.children) == 0:
            return total_dist
        hier_dist = 0
        for v, child in self.children.items():
            check_total = child.distance(another_node.children[v], children_depth + 1, layer_dict, W)
            total_dist = total_dist + check_total
            check_hier = child.amount / another_children_amt_sum
            hier_dist = hier_dist + check_hier

        hier_dist = hier_dist / layer_dict[children_depth]
        total_dist = total_dist + hier_dist * (children_depth + 1) * 1 / W
        return total_dist

    def get_children_amt_sum(self):
        children_amt = 0
        for child in self.children.values():
            children_amt = children_amt + child.amount

        return children_amt

    #
    # def distance(self, another_node, children_depth, layer_dict):
    #     total_dist = 0
    #     if len(self.children) == 0:
    #         return total_dist
    #     hier_dist = 0
    #     for v, child in self.children.items():
    #         check_total = child.distance(another_node.children[v],children_depth+1,layer_dict)
    #         total_dist = total_dist+check_total
    #         check_hier = (child.amount-self.get_children_amt_sum())/(another_node.amount-another_node.get_amt_sum()) / layer_dict[children_depth]
    #         hier_dist = hier_dist+check_hier
    #
    #     total_dist = total_dist + hier_dist* children_depth * 1/10
    #     return total_dist

    # 表示一颗FTC_Tree的类

    def tune(self, threshold_amt):
        for v in list(self.children.keys()):
            if self.children[v].amount < threshold_amt:
                del self.children[v]
            else:
                self.children[v].tune(threshold_amt)

    def get_layer_freq(self, depth, freq_dict):
        if not len(self.children) == 0:
            freq_dict[depth] = freq_dict[depth] + 1
            for child in self.children.values():
                child.get_layer_freq(depth + 1, freq_dict)

        # for _, child in self.children.items():
        #     if not len(child.children):
        #         freq_dict[depth] = freq_dict[depth] + 1
        #     child.get_layer_freq(depth + 1, freq_dict)

    def get_residence(self):
        children_amt_sum = self.get_children_amt_sum()
        self.amount = self.amount - children_amt_sum
        for child in self.children.values():
            child.get_residence()


class FTC_Tree:
    def __init__(self):
        self.root = Node()
        self.n_level = 4
        self.height = 0

    def insert_purchase_record(self, level, amount, pluno, pur_time):
        cur_level = level
        finest_pluno = pluno[:level + 1]
        self.root.insert_purchase_record(cur_level, 1, finest_pluno, pur_time)
        self.root.category = "root"

    def union(self, another_tree):
        union_tree = FTC_Tree()
        union_tree.height = max(self.height, another_tree.height)
        # 从根节点开始合并
        union_tree.root = self.root.union(another_tree.root)
        return union_tree

    def intersect(self, another_tree):
        inter_tree = FTC_Tree()
        inter_tree.height = self.height
        # 从根节点开始取交集
        if another_tree is None:
            assert another_tree is not None
        inter_tree.root = self.root.intersect(another_tree.root)
        return inter_tree

    def get_avg_amt(self):
        n_node, n_amt = self.root.get_amt_node()
        return n_amt / n_node

    def get_max_amt(self):
        max_amt = self.root.get_max_amt()
        return max_amt

    def get_layer_freq(self, depth):
        freq_dict = dict()
        for l in range(0, self.n_level + 1):
            freq_dict[l] = 0
        self.root.get_layer_freq(depth, freq_dict)
        return freq_dict

    def distance(self, another_tree):
        inter_tree = self.intersect(another_tree)
        union_tree = self.union(another_tree)
        # 获得每层的节点数
        children_depth = 0
        layer_dict = inter_tree.get_layer_freq(children_depth)
        # print(layer_dict)
        W = (union_tree.height + 1) * union_tree.height / 2
        dist = 1 - inter_tree.root.distance(union_tree.root, children_depth, layer_dict, W)
        return dist

    def tune(self, threshold_amt):
        self.root.tune(threshold_amt)

    def get_residence(self):
        self.root.get_residence()

