from copy import deepcopy
from core.Environment import post_fix_to_tree

def simplify_0_op(pf):
    if len(pf) <=1:
        return pf
    newpf = []
    i = 0
    while i <= len(pf) - 2:
        if pf[i] in ['0'] and pf[i + 1] in ['+', '-']:
            i += 2 #empty operation
            break
        else:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf


def simplify_0_function(pf):
    if len(pf) <=1:
        return pf
    newpf = []
    i = 0
    while i <= len(pf) - 2:
        if pf[i] in ['0'] and pf[i + 1] in ['np.sin(', 'np.tan(', 'np.arcsin(', 'np.arctan(', 'np.arcsinh(', 'np.arctanh(', 'np.sinh(', 'np.tanh(']:
            newpf.append('0')
            i += 2 #empty operation
            break
        elif pf[i] in ['0'] and pf[i + 1] in ['np.cos(', 'np.exp(', 'np.cosh(']:
            newpf.append('1')
            i += 2
            break
        elif pf[i] in ['1'] and pf[i + 1] in ['np.log(', 'np.arccos(', 'np.arccosh(']:
            newpf.append('0')
            i += 2
            break
        else:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf

def simplify_function_function(pf):
    if len(pf) <=1:
        return pf
    #like log(exp(x)) = x
    newpf = []
    i = 0
    while i <= len(pf) - 2:
        if pf[i] in ['np.log('] and pf[i + 1] in ['np.exp('] :
            i += 2
            break
        elif pf[i] in ['np.exp('] and pf[i + 1] in ['np.log('] :
            i += 2
            break
        else:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf

def simplify_1_op(pf):
    if len(pf) <=1:
        return pf
    newpf = []
    i = 0
    while i <= len(pf) - 2:
        if pf[i] in ['1'] and pf[i + 1] in ['*', '/']:
            i += 2 #empty operation
            break
        else:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf

def simplify_aa_op(pf):
    if len(pf) <=2:
        return pf
    newpf = []
    i = 0
    while i <= len(pf) - 3:
        if pf[i] == 'A' and pf[i + 1] == 'A' and pf[i + 2] in ['+', '-', '*', '/', '**']:
            newpf.append('A')
            i += 3
            break
        else:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf

def simplify_a_function(pf, function_list):
    if len(pf) <=1:
        return pf
    i = 0
    newpf = []
    while i <= len(pf) - 2:
        if pf[i] == 'A' and pf[i + 1] in function_list:
            #print('happen', i, pf[i], pf[i + 1], 'current', newpf)
            newpf.append('A')
            i += 2
            break
        else:
            newpf.append(pf[i])
            i += 1
    #print('final i', i, 'current', newpf)
    newpf += pf[i:]
    #print('ducoup', newpf)
    return newpf


def simplify_a_power(pf, power_integers):
    if len(pf) <=2:
        return pf
    i = 0
    newpf = []
    while i <= len(pf) - 3:
        if pf[i] == 'A' and pf[i + 1] in power_integers and pf[i + 2] == '**':
            #print('happen', i, pf[i], pf[i + 1], 'current', newpf)
            newpf.append('A')
            i += 3
            break
        else:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf

def simplify_1_0_power(pf):
    if len(pf) <= 1:
        return pf
    i = 0
    newpf = []
    while i <= len(pf) - 2:
        if pf[i] == '1' and pf[i + 1] == '**':
            i += 2
            break
        elif pf[i] == '0' and pf[i + 1] == '**':
            newpf.append('1')
            i += 2
            break
        else:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf

def simplify_var_op_var(pf, var_list):
    if len(pf) <=2:
        return pf
    i = 0
    newpf = []
    while i <= len(pf) - 3:

        # removed since it breaks dimensional analysis
        #if pf[i] in var_list and pf[i + 1] == pf[i] and pf[i + 2] in ['-']:
        #    newpf.append('0')
        #    i += 3
        #    break
        if pf[i] in var_list and pf[i + 1] == pf[i] and pf[i + 2] in ['/']:
            newpf.append('1')
            i += 3
            break
        else:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf

def simplify_sqrt_pow_2(pf):
    #'2', '**', 'np.sqrt(' to none
    if len(pf) <=2:
        return pf
    i = 0
    newpf = []
    while i <= len(pf) - 3:
        if pf[i] == '2' and pf[i + 1] == '**' and pf[i + 2] == 'np.sqrt(':
            i += 3
            break
        else:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf

def simplify_a_op_a_op(pf):
    if len(pf) <=3:
        return pf
    i = 0
    newpf = []
    while i <= len(pf) - 4:
        simplified = False
        for op in ['+', '-', '*', '/']:
            if pf[i] == 'A' and pf[i + 1] == op and pf[i + 2] =='A' and pf[i + 3] == op:
                newpf.append('A')
                newpf.append(op)
                i += 4
                simplified = True
                break

        if not simplified:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf

def simplify_a_op_a_op_bis(pf):
    if len(pf) <=3:
        return pf
    i = 0
    newpf = []
    while i <= len(pf) - 4:
        simplified = False
        for ops in [['+', '-'], ['-', '+'], ['*', '/'], ['/', '*']]:
            if pf[i] == 'A' and pf[i + 1] == ops[0] and pf[i + 2] =='A' and pf[i + 3] == ops[1]:
                newpf.append('A')
                newpf.append('+') if '+' in ops else newpf.append('*')
                i += 4
                simplified = True
                break

        if not simplified:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf


def simplify_a_var_op_a_op(pf, var_list):
    if len(pf) <=4:
        return pf
    i = 0
    newpf = []
    while i <= len(pf) - 5:
        simplified = False
        for ops in [['+', '-'], ['*', '/']]:
            if pf[i] == 'A' and pf[i+1] in var_list and pf[i + 2] in ops and pf[i + 3] =='A' and pf[i + 4] in ops:
                newpf.append('A')
                newpf.append(pf[i+1])
                newpf.append(pf[i+2])
                i += 5
                simplified = True
                break

        if not simplified:
            newpf.append(pf[i])
            i += 1
    newpf += pf[i:]
    return newpf



def simplify(pf, var_list, function_list, power_integers):
    try:
        local_pf = deepcopy(pf)
        local_pf = simplify_a_function(local_pf, function_list)
        local_pf = simplify_0_op(local_pf)
        local_pf = simplify_0_function(local_pf)
        local_pf = simplify_1_op(local_pf)
        local_pf = simplify_function_function(local_pf)
        local_pf = simplify_aa_op(local_pf)
        local_pf = simplify_var_op_var(local_pf, var_list)
        local_pf = simplify_a_power(local_pf, power_integers)
        local_pf = simplify_a_op_a_op(local_pf)
        local_pf = simplify_a_op_a_op_bis(local_pf)
        local_pf = simplify_a_var_op_a_op(local_pf, var_list)
        local_pf = simplify_sqrt_pow_2(local_pf)
        return local_pf
    except:
        print('error in simplify', pf)
        return pf


def simplify_one_tree(tree, vocabulary):
    new_pf = deepcopy(tree.postfix_formula)
    #print('here global start', tree.infix_formula)
    previous_len = len(new_pf)
    done = False
    cnt = 0
    while not done and cnt < 10:
        new_pf = simplify(new_pf, vocabulary.variables,
                                 vocabulary.arity_1_symbols_with_sqrt,
                                 vocabulary.integers_for_power)

        if len(new_pf) == previous_len:
            done = True
        previous_len = len(new_pf)
        cnt += 1
    # if done and use floats :
    new_pf = [x if x not in ['0', '1'] else 'A' for x in new_pf] #replace 0 and 1 by scalars A and simplify again
    new_pf = simplify(new_pf, vocabulary.variables,
                             vocabulary.arity_1_symbols_with_sqrt,
                             vocabulary.integers_for_power)
    previous_len = len(new_pf)
    done = False
    cnt = 0
    while not done and cnt < 10:
        new_pf = simplify(new_pf, vocabulary.variables,
                                 vocabulary.arity_1_symbols_with_sqrt,
                                 vocabulary.integers_for_power)
        if len(new_pf) == previous_len:
            done = True
        previous_len = len(new_pf)
        cnt += 1
    tree = post_fix_to_tree(new_pf, vocabulary)
    return tree


def simplify_pool(pool, vocabulary):
    simplified_pool_as_trees = []
    for tree in pool:
        new_tree = simplify_one_tree(tree, vocabulary)
        simplified_pool_as_trees.append(new_tree)
    return simplified_pool_as_trees

