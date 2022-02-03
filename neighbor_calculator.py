import numpy as np
import math
from hamiltonian_builder import *
from pprint import pprint
import itertools as it
import functools as ft
#from numba import jit


def get_all_sets_with_odd_overlap(non_zero_neighbor_sets, overlap, d):
    assert d < 10
    assert 0 in overlap
    ret_sets = []
    for v in overlap:
        for s in non_zero_neighbor_sets:
            np_s = np.array(s)
            if v != 0:
                if np_s.size == 2 and np.all(np_s == np.array([0, d])):
                    continue
                mod_s = np.sort(np_s * (np_s != d) + v * (9*(np_s > 0)+1) * (np_s != d))
                ret_sets.append(mod_s)
            else:
                ret_sets.append(np_s)
    return ret_sets


def do_thing():
    ret_set = []
    for i in range(1, 5):
        # edges
        ret_set.append([0,i])

        # 4 type 1 quads
        ret_set.append([j for j in range(5) if j != i])

        for j in range(1, 4):
            ret_set.append([0, i * 10 + j])
            ret_set.append([0, i] + [i*10 + k for k in range(1, 4) if k != j])

        # type 2 quads
        ret_set.append([0, i*10 + 1, i*10 + 2, i*10 + 3])

    return ret_set


#@jit(nopython=True)
def reduce_odds(set_if_sets):
    flattened = [val for set_elem in set_if_sets for val in set_elem]
    active_nodes = np.remainder(np.bincount(flattened), 2).nonzero()[0]
    return active_nodes


#@jit(nopython=True)
def reduce_odds_sct(set_if_sets):
    flattened = [val for set_elem in set_if_sets for val in set_elem[0]]
    active_nodes = np.remainder(np.bincount(flattened), 2).nonzero()[0]
    return active_nodes


#@jit(nopython=True)
def get_subsets_that_sum_to(all_set, target):
    ret_sets = []
    n = len(all_set)
    np_target = np.array(target)
    for i in range(1 << n):
        subset = [j for j in range(n) if (i & (1 << j))]

        flattened = [val for set_ind in subset for val in all_set[set_ind]]
        active_nodes = np.remainder(np.bincount(flattened), 2).nonzero()[0]

        if active_nodes.size == np_target.size and (active_nodes-np_target == 0).all():
            ret_sets.append([all_set[set_ind] for set_ind in subset])

    return ret_sets


#@jit(nopython=True)
def get_subsets_that_sum_to_sct(all_set, target):
    ret_sets = []
    n = len(all_set)
    np_target = np.array(target)
    for i in range(1 << n):
        subset = [j for j in range(n) if (i & (1 << j))]

        flattened = [val for set_ind in subset for val in all_set[set_ind][0]]
        active_nodes = np.remainder(np.bincount(flattened), 2).nonzero()[0]

        if active_nodes.size == np_target.size and (active_nodes-np_target == 0).all():
            ret_sets.append([all_set[set_ind] for set_ind in subset])

    return ret_sets


def set_remove_add_elem(s, e):
    if e in s:
        s.remove(e)
    else:
        s.append(e)

# note, the center vertex is v0 or just 0

# d = 2
# for a vertex v, the possible combinations of neighbors with non-zero weights on the Zs are
# 0,1; 0,2;   1,2

# d = 3
# 0,1; 0,2; 0,3;   0,1,2,3

# d = 4
# 0,1; 0,2; 0,3; 0,4;   1,2; 1,3; 1,4; 2,3; 2,4; 3;4   0,1,2,3; 0,1,2,4; 0,1,3,4; 0,2,3,4;   1,2,3,4

# K is the set that needs to be oddly counted, everything else needs to be
# evenly counted


def d_3_main():
    neighbors_sets = get_subsets_that_sum_to(
        [[0, 1], [0, 2], [0, 3], [1, 11], [1, 12], [2, 21], [2, 22], [3, 31], [3, 32],
         [0, 1, 2, 3], [0, 1, 11, 12], [0, 2, 21, 22], [0, 3, 31, 32]], [0, 1, 2, 3])
    print(f"len(neighbors_sets) = {len(neighbors_sets)}")
    for comb in neighbors_sets:
        pprint(comb)
        print("")

    all_set_count = 0
    all_sets = []
    type_counter = {}
    for ns in neighbors_sets:
            assert np.all(reduce_odds(ns) == [0, 1, 2, 3])
            # all_sets.append(ng)
            all_set_count += 1

            num_edges = ft.reduce(lambda s, l: s + (1 if len(l) == 2 else 0), ns, 0)
            num_quads = ft.reduce(lambda s, l: s + (1 if len(l) == 4 else 0), ns, 0)

            if (num_edges, num_quads) not in type_counter:
                type_counter[(num_edges, num_quads)] = 0
            type_counter[(num_edges, num_quads)] += 1

    print(f"all_set_count={all_set_count}")

    latex_str = ""
    tot_edges = 9
    tot_quads = 10

    for k,v in type_counter.items():
        num_edges,num_quads = k
        neg = ((num_edges + 3*num_quads + 3)//2) % 2
        latex_str += f" - {v}" if neg == 1 else f" + {v}"
        if num_edges != 0:
            latex_str += f"\\sin^{{{num_edges}}}\\left(\\gamma\\right)"
        if num_quads != 0:
            latex_str += f"\\sin^{{{num_quads}}}\\left(\\frac{{\\gamma}}{{2}}\\right)"
        if tot_edges-num_edges != 0:
            latex_str += f"\\cos^{{{tot_edges-num_edges}}}\\left(\\gamma\\right)"
        if tot_quads-num_quads != 0:
            latex_str += f"\\cos^{{{tot_quads-num_quads}}}\\left(\\frac{{\\gamma}}{{2}}\\right)"

    print(latex_str)


def d_4_main():
    # print(get_all_sets_with_odd_overlap([[0,1], [0,2], [1,2]], [0], 2))
    # print(get_all_sets_with_odd_overlap([[0,1], [0,2], [1,2]], [0,1], 2))
    #
    #
    # print(get_all_sets_with_odd_overlap([[0,1], [0,2], [0,3], [0,1,2,3]], [0], 3))
    # print(get_all_sets_with_odd_overlap([[0,1], [0,2], [0,3], [0,1,2,3]], [0,1], 3))
    # print(get_all_sets_with_odd_overlap([[0,1], [0,2], [0,3], [0,1,2,3]], [0,1,2,3], 3))

    # sets = do_thing()
    # sets.sort(key=lambda e: sort_meth(e, 5))
    # print(sets)
    # print(len(sets))
    just_neighbors_sets = get_subsets_that_sum_to(
        [[0, 1], [0, 2], [0, 3], [0, 4], [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4]], [0, 1])
    print(f"len(just_neighbors_sets) = {len(just_neighbors_sets)}")
    for comb in just_neighbors_sets:
        pprint(comb)
        print("")
    neighbors_group_sets = get_subsets_that_sum_to(
        [[0, 1], [0, 11], [0, 12], [0, 13], [0, 1, 11, 12], [0, 1, 11, 13], [0, 1, 12, 13], [0, 11, 12, 13]], [])
    print(f"len(neighbors_group_sets) = {len(neighbors_group_sets)}")
    for comb in neighbors_group_sets:
        pprint(comb)
        print("")
    # pprint(get_subsets_that_sum_to([[0,1],[0,2],[0,3],[0,4], [0,1,2,3],[0,1,2,4],[0,1,3,4],[0,2,3,4]], [0,1]))

    all_set_count = 0
    all_sets = []
    type_counter = {}
    for ns in just_neighbors_sets:
        for edge_neighbor_sets in it.product(neighbors_group_sets, neighbors_group_sets, neighbors_group_sets,
                                             neighbors_group_sets):
            ng = ns.copy()
            for i in range(len(edge_neighbor_sets)):
                ng_tmp = [[elem + 10 * (i) if elem >= 10 else elem * (i + 1) for elem in K] for K in
                          edge_neighbor_sets[i]]
                if [0, i + 1] in ns:
                    ng.remove([0, i + 1])
                    set_remove_add_elem(ng_tmp, [0, i + 1])
                ng += ng_tmp
            # print(ng)
            assert np.all(reduce_odds(ng) == [0, 1])
            # all_sets.append(ng)
            all_set_count += 1

            num_edges = ft.reduce(lambda s, l: s + (1 if len(l) == 2 and np.sum(np.array(l) < 10) == 2 else 0), ng, 0)
            num_t2_pairs = ft.reduce(lambda s, l: s + (1 if len(l) == 2 and np.sum(np.array(l) < 10) == 1 else 0), ng,
                                     0)
            num_t1_quads = ft.reduce(lambda s, l: s + (1 if len(l) == 4 and np.sum(np.array(l) < 10) >= 2 else 0), ng,
                                     0)
            num_t2_quads = ft.reduce(lambda s, l: s + (1 if len(l) == 4 and np.sum(np.array(l) < 10) == 1 else 0), ng,
                                     0)

            if (num_edges, num_t2_pairs, num_t1_quads, num_t2_quads) not in type_counter:
                type_counter[(num_edges, num_t2_pairs, num_t1_quads, num_t2_quads)] = 0
            type_counter[(num_edges, num_t2_pairs, num_t1_quads, num_t2_quads)] += 1

    print(f"all_set_count={all_set_count}")


if __name__ == "__main__":
    #d_3_main()
    d_4_main()
