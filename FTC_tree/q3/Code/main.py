import pandas as pd
import numpy as np
import datetime
import math
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from toolkit import *
import time
import pickle


class Solution:
    def __init__(self):
        self.FTC_trees = dict()
        self.distance_matrix = None
        pass

    def build_tree(self, vipno, vipno_goods):
        n_good = vipno_goods.shape[0]
        for i in range(n_good):
            good = vipno_goods.iloc[i]
            level, amt, pluno, pur_time = good["level"], 1, good["pluno"], good['sldatime']
            if level > self.FTC_trees[vipno].height:
                self.FTC_trees[vipno].height = level
            self.FTC_trees[vipno].insert_purchase_record(level, amt, pluno, pur_time)

    def preprocess(self, ds):
        # 获得有多少个用户
        self.FTC_trees = dict()
        vipnos = np.unique(ds["vipno"])
        for vipno in vipnos:
            if vipno not in self.FTC_trees:
                self.FTC_trees[vipno] = FTC_Tree()
            # 摘取出对一个用户所有购买商品的条目
            vipno_goods = ds[ds["vipno"] == vipno]
            self.build_tree(vipno, vipno_goods)

    def get_k_initial_points(self, num_k):
        assert self.distance_matrix is not None
        vipnos = sorted(list(self.FTC_trees.keys()))
        max_dist = np.max(self.distance_matrix)
        posx, posy = np.where(self.distance_matrix == max_dist)
        idx0, idx1 = vipnos[posx[0]], vipnos[posy[0]]
        return [self.FTC_trees[idx0], self.FTC_trees[idx1]]

    def get_k_initial_points2(self, num_k):
        assert self.distance_matrix is not None
        vipnos = sorted(list(self.FTC_trees.keys()))
        n_vipno = len(vipnos)
        initial_pts = list()
        row_index = [i for i in range(n_vipno)]
        np.random.shuffle(row_index)
        start_pt = row_index[0]
        initial_pts.append(start_pt)
        for k in range(1, num_k):
            sub_matrix = self.distance_matrix[initial_pts]
            sub_matrix[:, initial_pts] = 0
            dist_to_centroid = np.sum(sub_matrix, 0)
            dist_to_centroid[initial_pts] = 0
            new_cols = np.where(dist_to_centroid == max(dist_to_centroid))

            initial_pts.append(new_cols[0][0])
        return [self.FTC_trees[vipnos[idx]] for idx in initial_pts]

    def transaction_kmeans(self, k):
        vipnos = list(sorted(self.FTC_trees.keys()))
        if len(vipnos) < num_k:
            return True, None, None
        results = dict()
        flag = False
        cores = self.get_k_initial_points2(k)
        #idx0, idx1 = choose_random_k(vipnos, k)
        #cores = [copy_tree(self.FTC_trees[idx0]), copy_tree(self.FTC_trees[idx1])]
        max_iter = 30
        for itr in range(1, max_iter + 1):
            vipno_min = dict()
            for vipno, vipno_tree in self.FTC_trees.items():
                dist0 = vipno_tree.distance(cores[0])
                dist1 = vipno_tree.distance(cores[1])
                vipno_min[vipno] = 0 if dist0 < dist1 else 1

            if vipno_min == results:
                return flag, cores, results

            results = vipno_min

            for i in range(k):
                cvs = [vipno for vipno, result in results.items() if result == i]
                if not len(cvs) == 0:
                    cpts = {cv: ct for cv, ct in self.FTC_trees.items() if cv in cvs}
                    cores[i] = self.get_centroid_tree(cpts)
                else:
                    print("early end")
                    flag = True
                    return flag, results, cores
        return flag, cores, results

    def get_centroid_tree(self, cpts):
        union_tree, centroid_tree = None, None
        for v in cpts:
            union_tree = copy_tree(cpts[v]) if union_tree is None else union_tree.union(cpts[v])
        cur_amt, mindist = 1, float("inf")
        amt_step = union_tree.get_avg_amt()
        amt_end = union_tree.get_max_amt()
        centroid_tree = copy_tree(union_tree)
        while cur_amt < amt_end:
            union_tree.tune(cur_amt)
            dist = sum([cpt.distance(union_tree) for cpt in cpts.values()])
            if dist < mindist:
                mindist = dist
                centroid_tree = copy_tree(union_tree)

            cur_amt = cur_amt + amt_step
        return centroid_tree

    def get_distance_matrix(self):
        n_vipno = len(self.FTC_trees.keys())
        vipnos = list(sorted(self.FTC_trees.keys()))
        dm = np.empty((n_vipno, n_vipno))
        for i in range(n_vipno):
            for j in range(n_vipno):
                tree1 = self.FTC_trees[vipnos[i]]
                tree2 = self.FTC_trees[vipnos[j]]
                dm[i, j] = dm[j, i] = tree1.distance(tree2)
        self.distance_matrix = dm
        return dm

    def get_silhouette_coefficient(self, distances, result, cores):
        rkeys = sorted(result.keys())
        result = np.array([result[v] for v in rkeys])
        n_sample = len(result)
        scs = np.empty(n_sample)

        dm = distances

        for i in range(n_sample):
            # 获得簇所在索引
            min_dist = float("inf")
            for c in range(len(cores)):
                if c == result[i]:
                    continue
                else:
                    outpts = dm[i, np.where(result == c)]
                    outdist = np.sum(outpts, axis=1) / outpts.shape[1]
                    if outdist < min_dist:
                        min_dist = outdist

            inpts = dm[i, np.where(result == result[i])]
            if inpts.shape[1] == 1:
                scs[i] = 0
            else:
                in_dist = np.sum(inpts, axis=1) / (inpts.shape[1] - 1)
                assert (max(min_dist, in_dist) != 0)
                scs[i] = (min_dist - in_dist) / max(min_dist, in_dist)
        return np.mean(scs)

    def get_compactness(self, results, cores):
        avg_compactness = 0
        for i in range(len(cores)):
            cvs = [vipno for vipno, result in results.items() if result == i]
            cpts = {cv: ct for cv, ct in self.FTC_trees.items() if cv in cvs}
            compactness = np.mean(np.array([cores[i].distance(ct) for ct in cpts.values()]))
            avg_compactness = avg_compactness + compactness
        return avg_compactness / len(cores)

    def variance(self, results, cores, i):
        centroid_tree = cores[i]
        cvs = [vipno for vipno, result in results.items() if result == i]
        cpts = {cv: ct for cv, ct in self.FTC_trees.items() if cv in cvs}

        vari = np.array([centroid_tree.distance(ct) for ct in cpts.values()])
        return np.sum(vari * vari)

    def bic(self, results, cores, *args):
        C = args[-1]
        if results is None and cores is None:
            cores = [self.get_centroid_tree(self.FTC_trees)]
            results = {cv: 0 for cv in self.FTC_trees.keys()}
        all_n_vipno, all_n_pluno = None, None
        for arg in args[:-1]:
            all_n_vipno = list(arg["vipno"]) if all_n_vipno is None else list(all_n_vipno) + list(arg["vipno"])
            all_n_pluno = list(arg["pluno5"]) if all_n_pluno is None else list(all_n_pluno) + list(arg["pluno5"])
        all_n_pluno = len(np.unique(all_n_pluno))
        all_n_vipno = len(np.unique(all_n_vipno))
        L = 0
        ''' 关注一下这个C具体代表什么'''
        for i in range(len(args[:-1])):
            n_vipno, n_pluno = len(np.unique(args[i]["vipno"])), len(np.unique((args[i]["pluno5"])))
            fai2 = 1 / (n_vipno - C) * solution.variance(results, cores, i) if n_vipno > C else 0.1
            L = L + n_vipno * np.log(n_vipno) - n_vipno * np.log(all_n_vipno) - n_vipno / 2 * np.log(
                2 * math.pi) - all_n_pluno * n_vipno / 2 * np.log(fai2) - (n_vipno - C) / 2
        L = L - np.log(C) * (all_n_pluno + 1) * C/2
        return L


if __name__ == "__main__":
    dataset = pd.read_csv("trade_new.csv").fillna(0)
    # 处理类别
    dataset["vipno"] = dataset["vipno"].astype("str")
    # dataset.loc[np.where(dataset["amt"] < 0)]["amt"] = 0
    dataset["amt"] = np.abs(dataset["amt"])
    dataset["pluno"] = dataset["pluno"].astype("str")
    dataset["pluno5"] = [t[:5] for t in dataset["pluno"]]
    # 处理时间
    dataset['sldatime'] = [datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in dataset['sldatime']]
    # 将时间转化为级别
    dataset['level'] = datetime.datetime.now() - dataset['sldatime']
    dataset['level'] = [int(t.days / 30) for t in dataset['level']]
    dataset['level'] = dataset['level'] - np.min(dataset['level']) + 1
    dataset['level'] = 4 - np.log2(dataset['level']).astype("int")
    original_dataset = deepcopy(dataset)
    datasets = list()
    finals = list()
    datasets.append({"ds": dataset, "core": None})
    solution = Solution()
    solution.preprocess(dataset)
    dm = solution.get_distance_matrix()
    with open("dm.pkl",'wb') as f:
        pickle.dump(dm, f)
    while not len(datasets) == 0:
        print("corruption !")
        dataset, cur_core = datasets.pop().values()
        solution.preprocess(dataset)
        solution.get_distance_matrix()
        end_early, cores, results = solution.transaction_kmeans(2)
        if end_early:
            finals.append({"ds": dataset, "core": cur_core})
            continue
        else:
            dataset1, dataset2 = split_dataset(dataset, results)

        new_bic = solution.bic(results, cores, dataset1, dataset2, 2)
        current_bic = solution.bic(None, None, dataset, 1)
        print("new_bic:", new_bic)
        print("current_bic:", current_bic)
        print('-' * 30)
        if new_bic > current_bic:
            datasets.append({"ds": dataset1, "core": cores[0]})
            datasets.append({"ds": dataset2, "core": cores[1]})
        else:
            finals.append({"ds": dataset, "core": cur_core})

    final_results = dict()
    final_cores = list()
    category = 0
    for final in finals:
        for vipno in np.unique(final["ds"]["vipno"]):
            final_results[vipno] = category
        category = category + 1
        final_cores.append(final["core"])

    solution.preprocess(original_dataset)
    dm = solution.get_distance_matrix()

    rkeys, rvalue = sorted(final_results.keys()), final_results.values()
    result = np.array([final_results[v] for v in rkeys])
    dm[np.diag_indices_from(dm)] = 0
    print(rvalue)
    print("Split {} clusters in total".format(len(np.unique(list(rvalue)))))
    print("SC:", silhouette_score(dm, result, metric="precomputed"))
    print("CP:", solution.get_compactness(final_results, final_cores))
