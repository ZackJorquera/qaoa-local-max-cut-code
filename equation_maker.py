import numpy as np
import math
import sys
import time
from hamiltonian_builder import *
from neighbor_calculator import *
from pprint import pprint
import itertools as it
import more_itertools as mit
import functools as ft
from collections import Counter
# from numba import jit
# import numba


#@jit(nopython=True)
def odd_overlap(a, b):
    return len(set(a).intersection(set(b))) % 2 == 1


#@jit(nopython=True)
def sin_cos_str(sin_cos_type, power, weight=(1, 1), str_format="latex"):
    sin_cos = sin_cos_type.lower()
    weight_str = "" if weight[0] == 1 else str(weight[0]), str(weight[1])
    if str_format == "latex" or (power == 1 and str_format == "desmos"):
        substr = f"\\frac{{{weight_str[0]}\\gamma}}{{{weight_str[1]}}}" if weight[1] != 1 else f"{weight_str[0]}\\gamma"
        power_substring = f"^{{{power}}}" if power != 1 else ""
        return f"\\{sin_cos}{power_substring}\\left({substr}\\right)"
    elif str_format == "mathematica":
        sin_cos = 'Sin' if sin_cos == 'sin' else 'Cos'
        power_substring = f"^({power})" if power != 1 else ""
        return f"{sin_cos}[{weight[0]}/{weight[1]} \\[Gamma]]{power_substring}"
    elif str_format == "desmos":
        substr = f"\\frac{{{weight_str[0]}\\gamma}}{{{weight_str[1]}}}" if weight[1] != 1 else f"{weight_str[0]}\\gamma"
        return f"\\left(\\{sin_cos}\\left({substr}\\right)\\right)^{{{power}}}"
    elif str_format == "python":
        substr = f" * {weight[0]}/{weight[1]}" if weight[0] != 1 or weight[1] != 1 else ""
        power_substring = f"**({power})" if power != 1 else ""
        return f" * {sin_cos}(gamma{substr}){power_substring}"


#@jit(nopython=True)
def make_equation_for(K, d, str_format="latex"):
    str_format = str_format.lower()
    assert str_format in ["latex", "mathematica", "desmos", "python"]
    eq_str = ""
    subsets = [[K[j] for j in range(len(K)) if (i & (1 << j))] for i in range(1 << len(K))]
    # TODO: create neighbors_sets from K and d, don't just hard code them
    all_neighbor_sets_d2 = [[0, 1], [0,12], [1, 12], [1, 122], [12, 122],
                            [0, 2], [0,22], [2, 22], [2, 222], [22, 222],]
    all_neighbor_sets_d3 = [[0, 1], [0, 2], [0, 3], [0, 1, 2, 3],
                         [1, 12], [1, 13], [0, 1, 12, 13], [1, 12, 122, 123], [1, 13, 132, 133],
                         [2, 22], [2, 23], [0, 2, 22, 23], [2, 22, 222, 223], [2, 23, 232, 233],
                         [3, 32], [3, 33], [0, 3, 32, 33], [3, 32, 322, 323], [3, 33, 332, 333]]
    all_neighbor_sets_d4 = [] # TODO

    if d == 2:
        all_neighbor_sets = all_neighbor_sets_d2
    elif d == 3:
        all_neighbor_sets = all_neighbor_sets_d3
    else:
        assert d == 2 or d == 3

    for L in subsets:
        all_neighbor_sets_with_overlap = [ns for ns in all_neighbor_sets if odd_overlap(ns, L)]
        neighbors_sets_that_sum_to_K = get_subsets_that_sum_to(all_neighbor_sets_with_overlap, K)
        # print(L, all_neighbor_sets_with_overlap, neighbors_sets_that_sum_to_K)

        if d == 3:
            all_set_count = 0
            type_counter = {}
            for ns in neighbors_sets_that_sum_to_K:
                assert np.all(reduce_odds(ns) == K)
                all_set_count += 1

                if d == 3:
                    num_edges = ft.reduce(lambda s, l: s + (1 if len(l) == 2 else 0), ns, 0)
                    num_quads = ft.reduce(lambda s, l: s + (1 if len(l) == 4 else 0), ns, 0)
                else:
                    assert d == 3

                if (num_edges, num_quads) not in type_counter:
                    type_counter[(num_edges, num_quads)] = 0
                type_counter[(num_edges, num_quads)] += 1

            # print(f"all_set_count={all_set_count}")

            eq_str_part = ""
            tot_edges = 9
            tot_quads = 10
            tot_edges = ft.reduce(lambda s, l: s + (1 if len(l) == 2 else 0), all_neighbor_sets_with_overlap, 0)
            tot_quads = ft.reduce(lambda s, l: s + (1 if len(l) == 4 else 0), all_neighbor_sets_with_overlap, 0)

            for k, v in type_counter.items():
                num_edges, num_quads = k
                neg = ((num_edges + 3 * num_quads + len(L)) // 2) % 2
                eq_str_part += f" - {v} " if neg == 1 else f" + {v} "
                if num_edges != 0:
                    eq_str_part += sin_cos_str('sin', num_edges, str_format=str_format)
                if num_quads != 0:
                    eq_str_part += sin_cos_str('sin', num_quads, (1,2), str_format=str_format)
                if tot_edges - num_edges != 0:
                    eq_str_part += sin_cos_str('cos', tot_edges - num_edges, str_format=str_format)
                if tot_quads - num_quads != 0:
                    eq_str_part += sin_cos_str('cos', tot_quads - num_quads, (1,2), str_format=str_format)

            if eq_str_part != "":
                if str_format == "latex":
                    eq_str += f"+ \\cos^{{{len(K) - len(L)}}}\\left(2 \\beta\\right) \\sin^{{{len(L)}}}\\left(2 \\beta\\right) \\left({eq_str_part}\\right)"
                elif str_format == "mathematica":
                    eq_str += f"+ Cos[2 \\[Beta]]^({len(K) - len(L)}) Sin[2 \\[Beta]]^({len(L)}) ({eq_str_part})"
                elif str_format == "desmos":
                    eq_str += f"+ \\left(\\cos\\left(2 \\beta\\right)\\right)^{{{len(K) - len(L)}}} \\left(\\sin\\left(2 \\beta\\right)\\right)^{{{len(L)}}} \\left({eq_str_part}\\right)"
                elif str_format == "python":
                    eq_str += f"+ cos(2 * beta)**({len(K) - len(L)}) * sin(2 * beta)**({len(L)}) * ({eq_str_part})"

    return eq_str


# @jit(nopython=True)
def count_list(list: list[tuple[list[int], int]]):
     return np.bincount(np.fromiter(map(lambda t: t[1], list), dtype=np.int32), minlength=4)


#@jit(nopython=True)
def make_equation_for_d4(K, str_format="latex"):
    str_format = str_format.lower()
    assert str_format in ["latex", "mathematica", "desmos", "python"]
    eq_str = ""
    assert K == [0, 1]
    subsets = [[K[j] for j in range(len(K)) if (i & (1 << j))] for i in range(1 << len(K))]

    # TODO: create neighbors_sets from K and d, don't just hard code them
    # 0: edge, 1: t2 pair, 2: t1 quad, 3: t2 quad
    just_neighbor_sets_L_eq_0 = [([0, 1], 0), ([0, 2], 0), ([0, 3], 0), ([0, 4], 0),
                                 ([0, 1, 2, 3], 2), ([0, 1, 2, 4], 2), ([0, 1, 3, 4], 2), ([0, 2, 3, 4], 2)]
    just_neighbor_sets_L_eq_01 = [([0, 2], 0), ([0, 3], 0), ([0, 4], 0), ([1, 11], 0), ([1, 12], 0), ([1, 13], 0),
                                  ([0, 11], 1), ([0, 12], 1), ([0, 13], 1), ([1, 2], 1), ([1, 3], 1), ([1, 4], 1),
                                  ([0, 2, 3, 4], 2), ([1, 11, 12, 13], 2), ([0, 11, 12, 13], 3), ([1, 2, 3, 4], 3)]

    neighbor_branch_sets = [([0, 1], 0), ([0, 11], 1), ([0, 12], 1), ([0, 13], 1),
                            ([0, 1, 11, 12], 2), ([0, 1, 11, 13], 2), ([0, 1, 12, 13], 2), ([0, 11, 12, 13], 3)]
    neighbor_branch_subsets_that_sum_to_emptyset = get_subsets_that_sum_to_sct(neighbor_branch_sets, []) # : list[tuple[list[int], int]]
    neighbor_branch_subsets_that_sum_to_01 = get_subsets_that_sum_to_sct(neighbor_branch_sets, [0,1])

    for L in subsets:
        print(f"K = {np.array(K)}, L = {np.array(L)}")
        all_set_count = 0
        type_counter = Counter()
        mult_factor = 1
        num_neighbor_branchs = 0

        if L == [0]:
            mult_factor = 2
            num_neighbor_branchs = 4

            just_neighbor_sets_L_eq_0_with_overlap = [ns for ns in just_neighbor_sets_L_eq_0 if odd_overlap(ns[0], L)]
            just_neighbors_sets_L_eq_0_that_sum_to_K = get_subsets_that_sum_to_sct(just_neighbor_sets_L_eq_0_with_overlap, K)
            just_neighbor_sets = just_neighbor_sets_L_eq_0_with_overlap

            j, N = 0, len(just_neighbors_sets_L_eq_0_that_sum_to_K)
            for ns in just_neighbors_sets_L_eq_0_that_sum_to_K:
                start_time = time.time()
                ng: list[tuple[list[int], int]] = ns.copy()
                edge_neighbor_sets_builder = []
                edge_neighbor_sets_builder_counts = []
                for i in range(4):
                    if ([0, i + 1], 0) in ns:
                        nbs_tmp = [[([elem + 10 * (i) if elem >= 10 else elem * (i + 1) for elem in K[0]], K[1]) for K in nbs]
                                  for nbs in neighbor_branch_subsets_that_sum_to_01]
                        ng.remove(([0, i + 1], 0))
                    else:
                        nbs_tmp = [[([elem + 10 * (i) if elem >= 10 else elem * (i + 1) for elem in K[0]], K[1]) for K in nbs]
                                  for nbs in neighbor_branch_subsets_that_sum_to_emptyset]
                    edge_neighbor_sets_builder.append(nbs_tmp)
                    edge_neighbor_sets_builder_counts.append([count_list(nb_tmp) for nb_tmp in nbs_tmp])

                ng_count = count_list(ng)
                type_counter += Counter(map(lambda ens: tuple(
                    sum(ens) + ng_count), it.product(*edge_neighbor_sets_builder_counts)))
                elp_time = time.time() - start_time

                j += 1
                sys.stdout.write(f"\r{j}/{N} : {elp_time:.2}s ")
                sys.stdout.flush()

            # print(f"all_set_count={all_set_count}")
        elif L == [1]:
            # this will just give us the same result as for L == [0] so all we do it add a times 2 to the L == [0] result
            continue
        elif L == [0, 1]:
            mult_factor = 1
            num_neighbor_branchs = 6

            just_neighbor_sets_L_eq_01_with_overlap = [ns for ns in just_neighbor_sets_L_eq_01 if odd_overlap(ns[0], L)]
            just_neighbors_sets_L_eq_01_that_sum_to_K = get_subsets_that_sum_to_sct(just_neighbor_sets_L_eq_01_with_overlap, K)
            just_neighbor_sets = just_neighbor_sets_L_eq_01_with_overlap

            j, N = 0, len(just_neighbors_sets_L_eq_01_that_sum_to_K)
            for ns in just_neighbors_sets_L_eq_01_that_sum_to_K:
                start_time = time.time()
                ng = ns.copy()
                edge_neighbor_sets_builder = []
                edge_neighbor_sets_builder_counts = []
                for edge in [[0, 2], [0, 3], [0, 4], [1, 11], [1, 12], [1, 13]]:
                    if (edge, 0) in ns:
                        nbs_tmp = [[([edge[0] if elem == 0 else elem + 10 * (edge[1]-1) if elem >= 10 else elem * edge[1] for elem in K[0]], K[1]) for K in nbs]
                                   for nbs in neighbor_branch_subsets_that_sum_to_01]
                        ng.remove((edge, 0))
                    else:
                        nbs_tmp = [[([edge[0] if elem == 0 else elem + 10 * (edge[1]-1) if elem >= 10 else elem * edge[1] for elem in K[0]], K[1]) for K in nbs]
                                   for nbs in neighbor_branch_subsets_that_sum_to_emptyset]
                    edge_neighbor_sets_builder.append(nbs_tmp)
                    edge_neighbor_sets_builder_counts.append([count_list(nb_tmp) for nb_tmp in nbs_tmp])

                ng_count = count_list(ng)
                type_counter += Counter(map(lambda ens: tuple(
                    sum(ens) + ng_count), it.product(*edge_neighbor_sets_builder_counts)))
                elp_time = time.time() - start_time

                j += 1
                sys.stdout.write(f"\r{j}/{N} : {elp_time:.5}s ")
                sys.stdout.flush()
        else:
            continue

        type_counter = dict(type_counter)
        print(f" : {len(type_counter)}")
        eq_str_part = ""
        tot_edges = 4
        tot_t2_pairs = 12
        tot_t1_quads = 16
        tot_t2_quads = 4
        tot_edges = ft.reduce(lambda s, l: s + (l[1] == 0), just_neighbor_sets, 0)
        tot_t2_pairs = ft.reduce(lambda s, l: s + (l[1] == 1), just_neighbor_sets, 0) \
                       + num_neighbor_branchs * ft.reduce(lambda s, l: s + (l[1] == 1), neighbor_branch_sets, 0)
        tot_t1_quads = ft.reduce(lambda s, l: s + (l[1] == 2), just_neighbor_sets, 0) \
                       + num_neighbor_branchs * ft.reduce(lambda s, l: s + (l[1] == 2), neighbor_branch_sets, 0)
        tot_t2_quads = ft.reduce(lambda s, l: s + (l[1] == 3), just_neighbor_sets, 0) \
                       + num_neighbor_branchs * ft.reduce(lambda s, l: s + (l[1] == 3), neighbor_branch_sets, 0)

        for k, v in type_counter.items():
            num_edges, num_t2_pairs, num_t1_quads, num_t2_quads = k
            neg = ((num_edges + num_t2_pairs + 3 * num_t1_quads + 3 * num_t2_quads + len(L)) // 2) % 2
            eq_str_part += f" - {v} " if neg == 1 else f" + {v} "
            if num_edges != 0:
                eq_str_part += sin_cos_str('sin', num_edges, (3, 4), str_format=str_format)
            if num_t2_pairs + num_t1_quads != 0:
                eq_str_part += sin_cos_str('sin', num_t2_pairs + num_t1_quads, (1, 8), str_format=str_format)
            if num_t2_quads != 0:
                eq_str_part += sin_cos_str('sin', num_t2_quads, (3, 8), str_format=str_format)
            if tot_edges - num_edges != 0:
                eq_str_part += sin_cos_str('cos', tot_edges - num_edges, (3, 4), str_format=str_format)
            if tot_t2_pairs - num_t2_pairs + tot_t1_quads - num_t1_quads != 0:
                eq_str_part += sin_cos_str('cos', tot_t2_pairs - num_t2_pairs + tot_t1_quads - num_t1_quads, (1, 8),
                                           str_format=str_format)
            if tot_t2_quads - num_t2_quads != 0:
                eq_str_part += sin_cos_str('cos', tot_t2_quads - num_t2_quads, (3, 8), str_format=str_format)

        if eq_str_part != "":
            # The 2 multiplier is there because we don't want to run L == [1] when it will just give us that same result
            if str_format == "latex":
                eq_str += f"+ {mult_factor} \\cos^{{{len(K) - len(L)}}}\\left(2 \\beta\\right) \\sin^{{{len(L)}}}\\left(2 \\beta\\right) \\left({eq_str_part}\\right)"
            elif str_format == "mathematica":
                eq_str += f"+ {mult_factor} Cos[2 \\[Beta]]^({len(K) - len(L)}) Sin[2 \\[Beta]]^({len(L)}) ({eq_str_part})"
            elif str_format == "desmos":
                eq_str += f"+ {mult_factor} \\left(\\cos\\left(2 \\beta\\right)\\right)^{{{len(K) - len(L)}}} \\left(\\sin\\left(2 \\beta\\right)\\right)^{{{len(L)}}} \\left({eq_str_part}\\right)"
            elif str_format == "python":
                eq_str += f"+ {mult_factor} * cos(2 * beta)**({len(K) - len(L)}) * sin(2 * beta)**({len(L)}) * ({eq_str_part})"
    return eq_str


#@jit(nopython=True)
def make_expected_val_equation_string_for_d_4(str_format):
    latex_eq_str1: str = make_equation_for_d4([0, 1], str_format)
    latex_eq_str2 = "0"  # make_equation_for_d4([0, 1, 2, 3], str_format)
    latex_eq_str3 = "0"  #make_equation_for_d4([0, 1, 2, 3], str_format)
    latex_eq_str4 = "0"  #make_equation_for_d4([0, 1, 2, 3], str_format)

    if str_format == 'mathematica':
        return f"11n/16 - 3n/4 ({latex_eq_str1}) - 3n/8 ({latex_eq_str2}) + n/4 ({latex_eq_str3}) + 3n/16 ({latex_eq_str4})"
    elif str_format == "latex" or str_format == "desmos":
        return f"\\frac{{11n}}{{16}} - \\frac{{3n}}{{4}}\\left({latex_eq_str1}\\right) - \\frac{{3n}}{{8}}\\left({latex_eq_str2}\\right) + \\frac{{n}}{{4}}\\left({latex_eq_str3}\\right) + \\frac{{3n}}{{16}}\\left({latex_eq_str4}\\right)"
    elif str_format == 'python':
        return f"(11*n/16) - (3*n/4) * ({latex_eq_str1}) - (3*n/8) * ({latex_eq_str2}) + (n/4) * ({latex_eq_str3}) + (3*n/16) * ({latex_eq_str4})"


# @jit(nopython=True)
def make_expected_val_equation_string_for_d_3(str_format):
        latex_eq_str1 = make_equation_for([0, 1], 3, str_format)
        latex_eq_str2 = make_equation_for([0, 1, 2, 3], 3, str_format)

        if str_format == 'mathematica':
            return f"n/2 - 3n/4 ({latex_eq_str1}) + n/4 ({latex_eq_str2})"
        elif str_format == "latex" or str_format == "desmos":
            return f"\\frac{{n}}{{2}} - \\frac{{3n}}{{4}}\\left({latex_eq_str1}\\right) + \\frac{{n}}{{4}}\\left({latex_eq_str2}\\right)"
        elif str_format == 'python':
            return f"(n/2) - (3*n/4) * ({latex_eq_str1}) + (n/4) * ({latex_eq_str2})"


if __name__ == '__main__':
    print("running")
#     h_out_code = f"""
# def h_out(n, gamma, beta):
#     from numpy import sin, cos
#     return {make_expected_val_equation_string_for_d_3('python')}
#
# print(f">>> {{h_out(1, 0.635848, 0.345952)}}")
# """
#     print(h_out_code)
#     exec(h_out_code)

    h_out_code = f"""
def h_out(n, gamma, beta):
    from numpy import sin, cos
    return {make_expected_val_equation_string_for_d_4('python')}

print(f">>> {{h_out(1, 0.635848, 0.345952)}}")
"""
    print(h_out_code)
    exec(h_out_code)
