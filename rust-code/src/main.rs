#![allow(non_snake_case)]

use std::fs::File;
use std::hash::Hash;
use std::ops::Rem;
use std::collections::HashSet;
use std::time::SystemTime;
use ndarray::prelude::*;
use counter::Counter;
use itertools::Itertools;
use std::io::Write;

fn powerset<T>(s: &[T]) -> Vec<Vec<T>> where T: Clone {
    (0..2usize.pow(s.len() as u32)).map(|i| {
         s.iter().enumerate().filter(|&(t, _)| (i >> t) % 2 == 1)
                             .map(|(_, element)| element.clone())
                             .collect()
     }).collect()
}

// len(set(a).intersection(set(b))) % 2 == 1
fn odd_overlap<T, T1, T2>(a: T1, b: T2) -> bool 
where 
    T: Eq + Hash, 
    T1: IntoIterator<Item = T>, 
    T2: IntoIterator<Item = T>
{
    a.into_iter().collect::<HashSet<_>>().intersection(&b.into_iter().collect::<_>()).count() % 2 == 1
}

fn get_subsets_that_sum_to_sct(all_set: &[(Vec<i32>, i32)], target: &[i32]) -> Vec<Vec<(Vec<i32>, i32)>> {
    let mut ret_set: Vec<Vec<(Vec<i32>, i32)>> = vec![];
    let np_target = Array1::from_iter(target.iter().cloned());

    for subset in powerset(all_set) {
        let bins = subset.iter().map(|e| e.0.to_owned()).flatten().fold(vec![], |mut bins, num| {
            if bins.len() <= num as usize {
                bins.extend_from_slice(&vec![0; num as usize - bins.len() + 1]);
            }
            bins[num as usize] += 1;
            bins
        });
        let active_nodes = Array::from_iter(Array::from_vec(bins).rem(2i32)
            .indexed_iter()
            .filter_map(|(i, v)| (*v != 0).then(|| i as i32)));

        if np_target.shape() == active_nodes.shape() && np_target.eq(&active_nodes) {
            ret_set.push(subset)
        }
    }
    ret_set
}


fn bincount(list: &Array1<usize>, len: usize) -> Array1<usize>{
    let mut bins = Array1::zeros(len);
    list.iter().for_each(|elem| bins[*elem] += 1);
    bins
}

fn count_list<>(list: &[(Vec<i32>, i32)]) -> Array1<usize> {
    bincount(&Array::from_iter(list.iter().map(|e| e.1 as usize)), 4)
}

fn sin_cos_str(sin_cos_type: &str, power: i32, weight: (i32, i32), str_format: &str) -> String {
    let sin_cos = sin_cos_type.to_lowercase();
    let weight_str0 = if weight.0 == 1 {String::from("")} else {weight.0.to_string()};

    if str_format == "latex" || (power == 1 && str_format == "desmos") {
        let substr = if weight.1 != 1 {format!("\\frac{{{}\\gamma}}{{{}}}", weight_str0, weight.1)} else {format!("{}\\gamma", weight_str0)};
        let power_substring = if power != 1 {format!("^{{{}}}", power)} else {String::from("")};
        format!("\\{}{}\\left({}\\right)", sin_cos, power_substring, substr)
    }
    else if str_format == "mathematica" {
        let sin_cos = if sin_cos == "sin" {"Sin"} else {"Cos"};
        let power_substring = if power != 1 {format!("^({})", power)} else {String::from("")};
        format!("{}[{}/{} \\[Gamma]]{}", sin_cos, weight.0, weight.1, power_substring)
    }
    else if str_format == "desmos" {
        let substr = if weight.1 != 1 {format!("\\frac{{{}\\gamma}}{{{}}}", weight_str0, weight.1)} else {format!("{}\\gamma", weight_str0)};
        format!("\\left(\\{}\\left({}\\right)\\right)^{{{}}}", sin_cos, substr, power)
    }
    else if str_format == "python" {
        let substr = if weight.0 != 1 || weight.1 != 1 {format!(" * {}/{}", weight.0, weight.1)} else {String::from("")};
        let power_substring = if power != 1 {format!("**({})", power)} else {String::from("")};
        format!(" * {}(gamma{}){}", sin_cos, substr, power_substring)
    } else {
        unimplemented!()
    }
}

fn make_equation_for_d4(K: &[i32], str_format: &str) -> String {
    let str_format = str_format.to_lowercase();
    assert!(str_format == "latex" || str_format ==  "mathematica" || str_format ==  "desmos" || str_format ==  "python");
    let mut eq_str = String::new();
    assert!(K == [0, 1] || K == [0, 11]);

    let subsets = powerset(K);

    // TODO: create neighbors_sets from K and d, don't just hard code them
    // 0: edge, 1: t2 pair, 2: t1 quad, 3: t2 quad
    let just_neighbor_sets_L_eq_0 = vec![(vec![0, 1], 0), (vec![0, 2], 0), (vec![0, 3], 0), (vec![0, 4], 0),
                                         (vec![0, 1, 2, 3], 2), (vec![0, 1, 2, 4], 2), (vec![0, 1, 3, 4], 2), (vec![0, 2, 3, 4], 2)];
    let just_neighbor_sets_L_eq_01 = vec![(vec![0, 2], 0), (vec![0, 3], 0), (vec![0, 4], 0), (vec![1, 11], 0), (vec![1, 12], 0), (vec![1, 13], 0),
                                          (vec![0, 11], 1), (vec![0, 12], 1), (vec![0, 13], 1), (vec![1, 2], 1), (vec![1, 3], 1), (vec![1, 4], 1),
                                          (vec![0, 2, 3, 4], 2), (vec![1, 11, 12, 13], 2), (vec![0, 11, 12, 13], 3), (vec![1, 2, 3, 4], 3)];
    let just_neighbor_sets_L_eq_011 = vec![(vec![0, 1], 0), (vec![0, 2], 0), (vec![0, 3], 0), (vec![0, 4], 0), (vec![1, 11], 0), (vec![11, 111], 0),
                                           (vec![11, 112], 0), (vec![11, 113], 0), (vec![0, 12], 1), (vec![0, 13], 1), (vec![11, 12], 1), (vec![11, 13], 1),
                                           (vec![0, 2, 3, 4], 2), (vec![0, 1, 3, 4], 2), (vec![0, 1, 2, 4], 2), (vec![0, 1, 2, 3], 2), (vec![0, 1, 12, 13], 2),
                                           (vec![11, 111, 112, 113], 2), (vec![1, 11, 111, 112], 2), (vec![1, 11, 111, 113], 2), (vec![1, 11, 112, 113], 2),
                                           (vec![1, 11, 12, 13], 2)];

    let neighbor_branch_sets = vec![(vec![0, 1], 0), (vec![0, 11], 1), (vec![0, 12], 1), (vec![0, 13], 1),
                            (vec![0, 1, 11, 12], 2), (vec![0, 1, 11, 13], 2), (vec![0, 1, 12, 13], 2), (vec![0, 11, 12, 13], 3)];
    let neighbor_branch_subsets_that_sum_to_emptyset = get_subsets_that_sum_to_sct(&neighbor_branch_sets, &[]);
    let neighbor_branch_subsets_that_sum_to_01 = get_subsets_that_sum_to_sct(&neighbor_branch_sets, &[0,1]);

    for L in subsets {
        println!("K = {:?}, L = {:?}", K, L);
        let mult_factor;
        let num_neighbor_branchs;
        let just_neighbor_sets;
        let mut type_counter = Counter::new();

        if L == [0] && K == [0, 1] {
            mult_factor = 2;
            num_neighbor_branchs = 4;
            let just_neighbor_sets_L_eq_0_with_overlap = just_neighbor_sets_L_eq_0.iter().cloned().unique()
                .filter(|ns| odd_overlap(&ns.0, &L)).collect::<Vec<_>>();
            let just_neighbors_sets_L_eq_0_that_sum_to_K = get_subsets_that_sum_to_sct(&just_neighbor_sets_L_eq_0_with_overlap, K);
            just_neighbor_sets = just_neighbor_sets_L_eq_0_with_overlap.clone();

            let mut j = 0;
            let N = just_neighbors_sets_L_eq_0_that_sum_to_K.len();

            for ns in just_neighbors_sets_L_eq_0_that_sum_to_K {
                let start = SystemTime::now();
                let mut ng = ns.clone();
                let mut edge_neighbor_sets_builder = vec![];
                let mut edge_neighbor_sets_builder_counts = vec![];
                for i in 0..4 {
                    let nbs_tmp = if let Some(ind) = ng.iter().position(|e| *e == (vec![0, i + 1], 0)) {
                        ng.remove(ind);
                        neighbor_branch_subsets_that_sum_to_01.iter().cloned().map(|nbs| 
                            nbs.into_iter().map(|k| (k.0.into_iter().map(|elem| 
                                if elem >= 10 {elem + 10 * (i)} else {elem * (i + 1)}).collect::<Vec<_>>(), k.1)).collect())
                            .collect::<Vec<Vec<_>>>()
                    } else {
                        neighbor_branch_subsets_that_sum_to_emptyset.iter().cloned().map(|nbs| 
                            nbs.into_iter().map(|k| (k.0.into_iter().map(|elem| 
                                if elem >= 10 {elem + 10 * (i)} else {elem * (i + 1)}).collect::<Vec<_>>(), k.1)).collect())
                            .collect::<Vec<Vec<_>>>()
                    };
                    edge_neighbor_sets_builder_counts.push(nbs_tmp.iter().map(|nb_tmp| count_list(nb_tmp)).collect::<Vec<_>>());
                    edge_neighbor_sets_builder.push(nbs_tmp);
                }

                let ng_count = count_list(&ng);
                type_counter += edge_neighbor_sets_builder_counts.into_iter()
                    .multi_cartesian_product().map(|ens| {
                        let sum = &ng_count + &ens[0] + &ens[1] + &ens[2] + &ens[3];
                        (sum[0], sum[1], sum[2], sum[3])
                    }).collect::<Counter<_>>();
                let elp_time = start.elapsed().unwrap().as_secs_f32();

                j += 1;
                print!("\r{}/{} : {:.5}s ", j, N, elp_time);
                std::io::stdout().flush().unwrap();
            }
        } 
        else if L == [0] && K == [0, 11] {
            mult_factor = 2;
            num_neighbor_branchs = 3;
            let just_neighbor_sets_L_eq_0_with_overlap = just_neighbor_sets_L_eq_0.iter().cloned()
                .chain(neighbor_branch_sets.iter().cloned()).unique()
                .filter(|ns| odd_overlap(&ns.0, &L)).collect::<Vec<_>>();
            let just_neighbors_sets_L_eq_0_that_sum_to_K = get_subsets_that_sum_to_sct(&just_neighbor_sets_L_eq_0_with_overlap, K);
            just_neighbor_sets = just_neighbor_sets_L_eq_0_with_overlap.clone();

            let mut j = 0;
            let N = just_neighbors_sets_L_eq_0_that_sum_to_K.len();

            for ns in just_neighbors_sets_L_eq_0_that_sum_to_K {
                let start = SystemTime::now();
                let mut ng = ns.clone();
                let mut edge_neighbor_sets_builder = vec![];
                let mut edge_neighbor_sets_builder_counts = vec![];
                for i in 1..4 {
                    let nbs_tmp = if let Some(ind) = ng.iter().position(|e| *e == (vec![0, i + 1], 0)) {
                        ng.remove(ind);
                        neighbor_branch_subsets_that_sum_to_01.iter().cloned().map(|nbs| 
                            nbs.into_iter().map(|k| (k.0.into_iter().map(|elem| 
                                if elem >= 10 {elem + 10 * (i)} else {elem * (i + 1)}).collect::<Vec<_>>(), k.1)).collect())
                            .collect::<Vec<Vec<_>>>()
                    } else {
                        neighbor_branch_subsets_that_sum_to_emptyset.iter().cloned().map(|nbs| 
                            nbs.into_iter().map(|k| (k.0.into_iter().map(|elem| 
                                if elem >= 10 {elem + 10 * (i)} else {elem * (i + 1)}).collect::<Vec<_>>(), k.1)).collect())
                            .collect::<Vec<Vec<_>>>()
                    };
                    edge_neighbor_sets_builder_counts.push(nbs_tmp.iter().map(|nb_tmp| count_list(nb_tmp)).collect::<Vec<_>>());
                    edge_neighbor_sets_builder.push(nbs_tmp);
                }

                let ng_count = count_list(&ng);
                type_counter += edge_neighbor_sets_builder_counts.into_iter()
                    .multi_cartesian_product().map(|ens| {
                        let sum = &ng_count + &ens[0] + &ens[1] + &ens[2];
                        (sum[0], sum[1], sum[2], sum[3])
                    }).collect::<Counter<_>>();
                let elp_time = start.elapsed().unwrap().as_secs_f32();

                j += 1;
                print!("\r{}/{} : {:.5}s ", j, N, elp_time);
                std::io::stdout().flush().unwrap();
            }
        } 
        else if L == [1] || L == [11] {
            continue;
        }
        else if L == [0, 1] || L == [0, 11] {
            mult_factor = 1;
            num_neighbor_branchs = 6;
            let just_neighbors_sets_that_sum_to_K;
            let edges;
            if L == [0, 1] {
                let just_neighbor_sets_L_eq_01_with_overlap = just_neighbor_sets_L_eq_01.iter().cloned().unique()
                    .filter(|ns| odd_overlap(&ns.0, &L)).collect::<Vec<_>>();
                just_neighbors_sets_that_sum_to_K = get_subsets_that_sum_to_sct(&just_neighbor_sets_L_eq_01_with_overlap, K);
                just_neighbor_sets = just_neighbor_sets_L_eq_01_with_overlap.clone();
                edges = [[0, 2], [0, 3], [0, 4], [1, 11], [1, 12], [1, 13]];
            } else if L == [0, 11] {
                let just_neighbor_sets_L_eq_011_with_overlap = just_neighbor_sets_L_eq_011.iter().cloned().unique()
                    .filter(|ns| odd_overlap(&ns.0, &L)).collect::<Vec<_>>();
                just_neighbors_sets_that_sum_to_K = get_subsets_that_sum_to_sct(&just_neighbor_sets_L_eq_011_with_overlap, K);
                just_neighbor_sets = just_neighbor_sets_L_eq_011_with_overlap.clone();
                edges = [[0, 2], [0, 3], [0, 4], [11, 111], [11, 112], [11, 113]];
            } else { unimplemented!(); }

            let mut j = 0;
            let N = just_neighbors_sets_that_sum_to_K.len();

            for ns in just_neighbors_sets_that_sum_to_K {
                let start = SystemTime::now();
                let mut ng = ns.clone();
                let mut edge_neighbor_sets_builder = vec![];
                let mut edge_neighbor_sets_builder_counts = vec![];
                for ref edge in edges {
                    let nbs_tmp = if let Some(ind) = ng.iter().position(|e| *e == (vec![edge[0], edge[1]], 0)) {
                        ng.remove(ind);
                        neighbor_branch_subsets_that_sum_to_01.iter().cloned().map(|nbs| 
                            nbs.into_iter().map(|k| (k.0.into_iter().map(|elem| 
                                if elem == 0 {edge[0]} else if elem >= 10 {elem + 10 * (edge[1]-1)} else {elem * edge[1]})
                                .collect::<Vec<_>>(), k.1)).collect())
                            .collect::<Vec<Vec<_>>>()
                    } else {
                        neighbor_branch_subsets_that_sum_to_emptyset.iter().cloned().map(|nbs| 
                            nbs.into_iter().map(|k| (k.0.into_iter().map(|elem| 
                                if elem == 0 {edge[0]} else if elem >= 10 {elem + 10 * (edge[1]-1)} else {elem * edge[1]})
                                .collect::<Vec<_>>(), k.1)).collect())
                            .collect::<Vec<Vec<_>>>()
                    };
                    edge_neighbor_sets_builder_counts.push(nbs_tmp.iter().map(|nb_tmp| count_list(nb_tmp)).collect::<Vec<_>>());
                    edge_neighbor_sets_builder.push(nbs_tmp);
                }

                let ng_count = count_list(&ng);
                type_counter += edge_neighbor_sets_builder_counts.into_iter()
                    .multi_cartesian_product().map(|ens| {
                        let sum = &ng_count + &ens[0] + &ens[1] + &ens[2] + &ens[3] + &ens[4] + &ens[5];
                        (sum[0], sum[1], sum[2], sum[3])
                    }).collect::<Counter<_>>();
                let elp_time = start.elapsed().unwrap().as_secs_f32();

                j += 1;
                print!("\r{}/{} : {:.5}s ", j, N, elp_time);
                std::io::stdout().flush().unwrap();
            }
        } 
        else {
            continue;
        }

        let type_counter = type_counter.into_map();
        println!(" : {}", type_counter.len());
        let mut eq_str_part = String::new();
        let mut tot_vals_1 = count_list(&just_neighbor_sets).into_iter();
        let mut tot_vals_2 = count_list(&neighbor_branch_sets).into_iter();
        let _ = tot_vals_2.next();
        let tot_edges = tot_vals_1.next().unwrap();
        let tot_t2_pairs = tot_vals_1.next().unwrap() + num_neighbor_branchs * tot_vals_2.next().unwrap();
        let tot_t1_quads = tot_vals_1.next().unwrap() + num_neighbor_branchs * tot_vals_2.next().unwrap();
        let tot_t2_quads = tot_vals_1.next().unwrap() + num_neighbor_branchs * tot_vals_2.next().unwrap();

        for (k, v) in type_counter {
            let (num_edges, num_t2_pairs, num_t1_quads, num_t2_quads) = k;
            let neg = ((num_edges + num_t2_pairs + 3 * num_t1_quads + 3 * num_t2_quads + L.len()) / 2) % 2 == 1;
            eq_str_part += &if neg {format!(" - {} ", v)} else {format!(" + {} ", v)};
            if num_edges != 0 {
                eq_str_part += &sin_cos_str("sin", num_edges as i32, (3, 4), &str_format);
            }
            if num_t2_pairs + num_t1_quads != 0 {
                eq_str_part += &sin_cos_str("sin", (num_t2_pairs + num_t1_quads) as i32, (1, 8), &str_format);
            }
            if num_t2_quads != 0 {
                eq_str_part += &sin_cos_str("sin", num_t2_quads as i32, (3, 8), &str_format);
            }
            if tot_edges - num_edges != 0 {
                eq_str_part += &sin_cos_str("cos", (tot_edges - num_edges) as i32, (3, 4), &str_format);
            }
            if tot_t2_pairs - num_t2_pairs + tot_t1_quads - num_t1_quads != 0 {
                eq_str_part += &sin_cos_str("cos", (tot_t2_pairs - num_t2_pairs + tot_t1_quads - num_t1_quads) as i32, (1, 8), &str_format);
            }
            if tot_t2_quads - num_t2_quads != 0 {
                eq_str_part += &sin_cos_str("cos", (tot_t2_quads - num_t2_quads) as i32, (3, 8), &str_format);
            }
        }

        if !eq_str_part.is_empty() {
            eq_str += &if str_format == "latex" {
                format!("+ {} \\cos^{{{}}}\\left(2 \\beta\\right) \\sin^{{{}}}\\left(2 \\beta\\right) \\left({}\\right)", mult_factor, K.len() - L.len(), L.len(), eq_str_part)
            }
            else if str_format == "mathematica" {
                format!("+ {} Cos[2 \\[Beta]]^({}) Sin[2 \\[Beta]]^({}) ({})", mult_factor, K.len() - L.len(), L.len(), eq_str_part)
            }
            else if str_format == "desmos" {
                format!("+ {} \\left(\\cos\\left(2 \\beta\\right)\\right)^{{{}}} \\left(\\sin\\left(2 \\beta\\right)\\right)^{{{}}} \\left({}\\right)", mult_factor, K.len() - L.len(), L.len(), eq_str_part)
            }
            else if str_format == "python" {
                format!("+ {} * cos(2 * beta)**({}) * sin(2 * beta)**({}) * ({})", mult_factor, K.len() - L.len(), L.len(), eq_str_part)
            }
            else {
                unimplemented!()
            }
        }
        
    }

    eq_str
}

fn make_expected_val_equation_string_for_d_4(str_format: &str) -> String {
    let eq_str1 = make_equation_for_d4(&[0, 1], str_format);
    let eq_str2 = make_equation_for_d4(&[0, 11], str_format);
    let eq_str3 = String::from("0");
    let eq_str4 = String::from("0");

    if str_format == "mathematica" {
        format!("11n/16 - 3n/4 ({}) - 3n/8 ({}) + n/4 ({}) + 3n/16 ({})", eq_str1, eq_str2, eq_str3, eq_str4)
    }
    else if str_format == "latex" || str_format == "desmos" {
        format!("\\frac{{11n}}{{16}} - \\frac{{3n}}{{4}}\\left({}\\right) - \\frac{{3n}}{{8}}\\left({}\\right) + \\frac{{n}}{{4}}\\left({}\\right) + \\frac{{3n}}{{16}}\\left({}\\right)", eq_str1, eq_str2, eq_str3, eq_str4)
    }
    else if str_format == "python" {
        format!("(11*n/16) - (3*n/4) * ({}) - (3*n/8) * ({}) + (n/4) * ({}) + (3*n/16) * ({})", eq_str1, eq_str2, eq_str3, eq_str4)
    } else {
        unimplemented!()
    }
}

fn main() {
    let eq_str = make_expected_val_equation_string_for_d_4("mathematica");

    let path = "eq_str_d_eq_4.txt";
    let mut output = File::create(path).unwrap();
    write!(output, "{}", eq_str).unwrap();

    println!("\n\n{}", eq_str);
}
