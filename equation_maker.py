import numpy as np
import math
from hamiltonian_builder import *
from neighbor_calculator import *
from pprint import pprint
import itertools as it
import functools as ft


def odd_overlap(a, b):
    return len(set(a).intersection(set(b))) % 2 == 1


def make_equation_for(K, d, str_format="latex"):
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
            all_sets = []
            type_counter = {}
            for ns in neighbors_sets_that_sum_to_K:
                assert np.all(reduce_odds(ns) == K)
                # all_sets.append(ng)
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


def make_expected_val_equation_string_for_d_3(str_format):
    latex_eq_str1 = make_equation_for([0, 1], 3, str_format)
    latex_eq_str2 = make_equation_for([0, 1, 2, 3], 3, str_format)

    if str_format == 'mathematica':
        return f"n/2 - 3n/4 ({latex_eq_str1}) + n/4 ({latex_eq_str2})"
    elif str_format == "latex" or str_format == "desmos":
        return f"\\frac{{n}}{{2}} - \\frac{{3n}}{{4}}\\left({latex_eq_str1}\\right) + \\frac{{n}}{{4}}\\left({latex_eq_str2}\\right)"
    elif str_format == 'python':
        return f"(n/2) - (3*n/4) * ({latex_eq_str1}) + (n/4) * ({latex_eq_str2})"


def h_out(n, gamma, beta):
    from numpy import sin, cos
    return (n/2) - (3*n/4) * (+ cos(2 * beta)**(1) * sin(2 * beta)**(1) * ( - 1  * sin(gamma) * cos(gamma)**(2) * cos(gamma * 1/2)**(4) - 1  * sin(gamma)**(2) * sin(gamma * 1/2) * cos(gamma) * cos(gamma * 1/2)**(3))+ cos(2 * beta)**(1) * sin(2 * beta)**(1) * ( - 1  * sin(gamma) * cos(gamma)**(2) * cos(gamma * 1/2)**(4) - 1  * sin(gamma)**(2) * sin(gamma * 1/2) * cos(gamma) * cos(gamma * 1/2)**(3))) + (n/4) * (+ cos(2 * beta)**(3) * sin(2 * beta)**(1) * ( + 1  * sin(gamma)**(3) * cos(gamma * 1/2)**(4) + 1  * sin(gamma * 1/2) * cos(gamma)**(3) * cos(gamma * 1/2)**(3))+ cos(2 * beta)**(3) * sin(2 * beta)**(1) * ( + 1  * sin(gamma * 1/2) * cos(gamma)**(3) * cos(gamma * 1/2)**(3) - 1  * sin(gamma)**(3) * sin(gamma * 1/2)**(2) * cos(gamma * 1/2)**(2))+ cos(2 * beta)**(3) * sin(2 * beta)**(1) * ( + 1  * sin(gamma * 1/2) * cos(gamma)**(3) * cos(gamma * 1/2)**(3) - 1  * sin(gamma)**(3) * sin(gamma * 1/2)**(2) * cos(gamma * 1/2)**(2))+ cos(2 * beta)**(1) * sin(2 * beta)**(3) * ( - 1  * sin(gamma * 1/2) * cos(gamma)**(5) * cos(gamma * 1/2)**(5))+ cos(2 * beta)**(3) * sin(2 * beta)**(1) * ( + 1  * sin(gamma * 1/2) * cos(gamma)**(3) * cos(gamma * 1/2)**(3) - 1  * sin(gamma)**(3) * sin(gamma * 1/2)**(2) * cos(gamma * 1/2)**(2))+ cos(2 * beta)**(1) * sin(2 * beta)**(3) * ( - 1  * sin(gamma * 1/2) * cos(gamma)**(5) * cos(gamma * 1/2)**(5))+ cos(2 * beta)**(1) * sin(2 * beta)**(3) * ( - 1  * sin(gamma * 1/2) * cos(gamma)**(5) * cos(gamma * 1/2)**(5))+ cos(2 * beta)**(1) * sin(2 * beta)**(3) * ( - 1  * sin(gamma)**(3) * cos(gamma)**(6) * cos(gamma * 1/2)**(10) - 1  * sin(gamma * 1/2) * cos(gamma)**(9) * cos(gamma * 1/2)**(9) - 3  * sin(gamma)**(4) * sin(gamma * 1/2) * cos(gamma)**(5) * cos(gamma * 1/2)**(9) + 3  * sin(gamma)**(3) * sin(gamma * 1/2)**(2) * cos(gamma)**(6) * cos(gamma * 1/2)**(8) - 3  * sin(gamma)**(5) * sin(gamma * 1/2)**(2) * cos(gamma)**(4) * cos(gamma * 1/2)**(8) - 4  * sin(gamma)**(6) * sin(gamma * 1/2)**(3) * cos(gamma)**(3) * cos(gamma * 1/2)**(7) + 1  * sin(gamma)**(9) * sin(gamma * 1/2)**(4) * cos(gamma * 1/2)**(6)))


if __name__ == '__main__':
    print(make_expected_val_equation_string_for_d_3('mathematica'))

    print(h_out(1, 0.635848, 0.345952))
