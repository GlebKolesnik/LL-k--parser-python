# Необходимые импорты
import networkx as nx
import matplotlib.pyplot as plt


# 1. LL(1)-анализатор

# Вспомогательные функции
def is_terminal(symbol, grammar):
    for rule in grammar:
        if rule == symbol:
            return False
    return True


def compute_first(symbol, grammar, computed_firsts={}):
    if symbol in computed_firsts:
        return computed_firsts[symbol]
    if is_terminal(symbol, grammar):
        return {symbol}
    firsts = set()
    for production in grammar[symbol]:
        if production[0] == 'ε':
            firsts.add('ε')
        else:
            for s in production:
                s_first = compute_first(s, grammar, computed_firsts)
                firsts.update(s_first - {'ε'})
                if 'ε' not in s_first:
                    break
            else:
                firsts.add('ε')
    computed_firsts[symbol] = firsts
    return firsts


def compute_follow(symbol, grammar, computed_follows={}):
    if symbol in computed_follows:
        return computed_follows[symbol]
    follows = set()
    if symbol == list(grammar.keys())[0]:  # If start symbol
        follows.add('$')
    for rule, productions in grammar.items():
        for production in productions:
            for i, s in enumerate(production):
                if s == symbol:
                    if i < len(production) - 1:
                        beta = production[i + 1:]
                        first_beta = set()
                        for b in beta:
                            first_b = compute_first(b, grammar, {})
                            first_beta.update(first_b - {'ε'})
                            if 'ε' not in first_b:
                                break
                        follows.update(first_beta)
                    if i == len(production) - 1 or 'ε' in first_beta:
                        follows.update(compute_follow(rule, grammar, computed_follows))
    computed_follows[symbol] = follows
    return follows


def build_ll1_parsing_table(grammar):
    table = {}
    for non_terminal in grammar:
        table[non_terminal] = {}
        for production in grammar[non_terminal]:
            first_symbols = compute_first(production[0], grammar)
            for symbol in first_symbols:
                if symbol != 'ε':
                    table[non_terminal][symbol] = production
            if 'ε' in first_symbols:
                follow_symbols = compute_follow(non_terminal, grammar)
                for symbol in follow_symbols:
                    table[non_terminal][symbol] = production
    return table


def ll1_parser(tokens, table, start_symbol):
    stack = ['$', start_symbol]
    idx = 0
    while stack:
        top = stack[-1]
        if idx == len(tokens):
            lookahead = '$'
        else:
            lookahead = tokens[idx]
        if top == lookahead:
            stack.pop()
            idx += 1
        else:
            try:
                production = table[top][lookahead]
                stack.pop()
                if production[0] != 'ε':
                    for symbol in reversed(production):
                        stack.append(symbol)
            except KeyError:
                return False
    return idx == len(tokens) and not stack


# 2. Вспомогательные функции
# (Они были предоставлены выше и используются в LL(1)-анализаторе)

# 3. LL(2)-анализатор
# Примечание: этот раздел был опущен из-за сложностей

# 4. Анализатор методом рекурсивного спуска

def S(tokens, idx):
    if idx < len(tokens) and tokens[idx] == 'a':
        idx, ast_node = A(tokens, idx + 1)
        return idx, {'name': 'S', 'children': [{'name': 'a'}, ast_node]}
    elif idx < len(tokens) and tokens[idx] == 'b':
        return idx + 1, {'name': 'S', 'children': [{'name': 'b'}]}
    return idx, None


def A(tokens, idx):
    if idx < len(tokens) and tokens[idx] == 'a':
        idx, ast_node = A(tokens, idx + 1)
        return idx, {'name': 'A', 'children': [{'name': 'a'}, ast_node]}
    elif idx < len(tokens) and tokens[idx] == 'b':
        idx, ast_node = B(tokens, idx + 1)
        return idx, {'name': 'A', 'children': [{'name': 'b'}, ast_node]}
    return idx, None


def B(tokens, idx):
    if idx < len(tokens) and tokens[idx] == 'b':
        return idx + 1, {'name': 'B', 'children': [{'name': 'b'}]}
    return idx, None


def recursive_descent(tokens):
    idx, ast = S(tokens, 0)
    if idx == len(tokens):
        return True, ast
    return False, None


# 5. Графическая визуализация AST

def draw_ast(ast):
    if not ast:
        print("AST is empty. Nothing to visualize.")
        return

    G = nx.DiGraph()
    node_count = [0]

    def add_nodes(node):
        if node is None:
            return

        current_node = node_count[0]
        G.add_node(current_node, label=node['name'])
        if 'children' in node:
            for child in node['children']:
                node_count[0] += 1
                G.add_edge(current_node, node_count[0])
                add_nodes(child)

    add_nodes(ast)

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    labels = {i: G.nodes[i]["label"] for i in G.nodes()}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color="skyblue")
    plt.show()


# Тестирование
if __name__ == "__main__":
    tokens_ll1 = ["a", "b", "c"]
    result_ll1 = ll1_parser(tokens_ll1, build_ll1_parsing_table({
        'S': [['a', 'A', 'c'], ['b', 'B', 'c'], ['ε']],
        'A': [['d'], ['ε']],
        'B': [['e'], ['ε']]
    }), 'S')
    print(f"LL(1) parser result for {tokens_ll1}: {result_ll1}")

    tokens_recursive = ["a", "a", "a", "b"]
    result_recursive, ast_recursive = recursive_descent(tokens_recursive)
    print(f"Recursive descent parser result for {tokens_recursive}: {result_recursive}")
    draw_ast(ast_recursive)
