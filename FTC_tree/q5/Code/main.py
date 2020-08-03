import pandas as pd
import numpy as np
import datetime
import math
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from toolkit import *
import pickle


def get_k_initial_points2(dm, num_k):
    assert dm is not None
    n_vipno = dm.shape[0]
    ipts = list()

    row_index = [i for i in range(n_vipno)]
    np.random.shuffle(row_index)
    start_pt = row_index[0]

    ipts.append(start_pt)

    for k in range(1, num_k):
        sub_dm = dm[ipts]
        dist_to_centroid = np.sum(sub_dm, 0)
        dist_to_centroid[ipts] = 0
        # print(dist_to_centroid)
        new_cols = np.where(dist_to_centroid == max(dist_to_centroid))
        ipts.append(new_cols[0][0])
    # print(ipts, dm[ipts[0], ipts[1]])
    return ipts


class Solution:
    def __init__(self):
        self.FTC_trees = dict()
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

    def get_k_initial_points(self, dm, num_k, alpha):
        assert dm is not None
        vipnos = list(sorted(self.FTC_trees.keys()))
        n_vipno = dm.shape[0]
        ipts = list()

        row_index = [i for i in range(n_vipno)]
        np.random.shuffle(row_index)
        for row in row_index:
            vtree = self.FTC_trees[vipnos[row]]
            if len(vtree.root.children) > 12:
                start_pt = row
                ipts.append(start_pt)
                break

        ratio = (1 - alpha) * num_k / n_vipno
        for k in range(1, num_k):
            sub_dm = dm[ipts]
            dist_to_centroid = np.sum(sub_dm, 0)
            dist_to_centroid[ipts] = -1
            # print(dist_to_centroid)
            new_cols = np.where(dist_to_centroid > max(dist_to_centroid) * (alpha + ratio))

            good_col, lst_child_num = new_cols[0][0], 0
            biggest = -1
            # print("new_cols:", new_cols)
            for col in new_cols[0]:
                vtree = self.FTC_trees[vipnos[col]]
                if np.sum(dm[col]) >= (n_vipno - 1) * alpha * (1-(num_k-2)/(n_vipno-2)) and len(vtree.root.children) >= lst_child_num:
                    good_col = col
                    biggest = np.sum(dm[col])
            # print("good_col:", np.sum(dm[good_col]))
            ipts.append(good_col)

        #print(ipts, dm[ipts[0], ipts[1]])
        #print(len(self.FTC_trees[vipnos[ipts[0]]].root.children))
        #print(len(self.FTC_trees[vipnos[ipts[1]]].root.children))
        return ipts

    def transaction_kmediod(self, dm, k):
        # k-means聚类算法
        # k       - 指定分簇数量
        # dm      - distance_matrix
        n_sample, n_feature = dm.shape  # m：样本数量，n：每个样本的属性值个数
        result = np.empty(n_sample, dtype=np.int)  # m个样本的聚类结果

        cores = self.get_k_initial_points(dm, k, alpha=0.7)
        # cores = get_k_initial_points2(dm, k)

        # 无放回抽取质心
        max_iter = 40
        while True: # 迭代计算
            distance = dm[:, cores]
            index_min = np.argmin(distance, axis=1)  # 每个样本距离最近的质心索引序号

            if (index_min == result).all():
                return result, cores

            result = index_min  # 更新聚类结果
            for i in range(k):  # 遍历质心集

                # 获得这个簇里的点在所有点里的索引
                cluster_index = np.array([s for s in range(n_sample)])[result == i]
                # 找出对应当前质心的子样本集
                items = dm[result == i]
                items = items[:, result == i]

                med_index = cluster_index[np.argmin(np.sum(items, axis=1))]
                cores[i] = med_index

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
        total = 0
        for i in range(n_vipno):
            for j in range(i + 1):
                tree1 = self.FTC_trees[vipnos[i]]
                tree2 = self.FTC_trees[vipnos[j]]
                dm[i, j] = dm[j, i] = tree1.distance(tree2)
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


def get_compactness(dm, result, cores):
    n_sample = dm.shape[0]
    n_cluster = len(cores)
    cts = np.empty((len(cores), 1))
    total_compactness = 0
    for i in range(len(cores)):
        dists = dm[cores[i], np.where(result == i)]
        avg_dist = np.sum(dists) / dists.shape[1]
        total_compactness = total_compactness + avg_dist
    return total_compactness / n_cluster


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
    # with open('ftct_dist_mat.txt','rb') as f:
    #     dm = pickle.load(f)
    dm[np.diag_indices_from(dm)] = 0

    total_sc, total_cp = dict(), dict()
    for k in range(2, 51):
        scs, cps = list(), list()
        for attempt in range(1):
            results, cores = solution.transaction_kmediod(dm, k)
            sc = silhouette_score(dm, results, metric="precomputed")
            cp = get_compactness(dm, results, cores)
            scs.append(sc)
            cps.append(cp)
        total_sc[k] = np.mean(scs)
        total_cp[k] = np.mean(cps)
        print("{}-mediod SC:{} CP:{}".format(k, np.mean(scs), np.mean(cps)))

    with open('careful_sc.pkl', 'wb') as f:
        pickle.dump(total_sc, f)
    with open('careful_cp.pkl', 'wb') as f:
        pickle.dump(total_cp, f)
