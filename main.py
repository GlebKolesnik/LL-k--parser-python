import networkx as nx
import matplotlib.pyplot as plt
def parse_grammar_from_text(grammar_text):
    """
    Parses a grammar from its textual representation into a dictionary.
    Format:
    A -> aBc | d
    B -> e | ε
    """
    grammar = {}
    lines = grammar_text.strip().split('\n')

    for line in lines:
        left, right = line.split('->')
        left = left.strip()
        productions = right.strip().split('|')
        grammar[left] = [prod.strip() for prod in productions]

    return grammar

def ll1_parse(input_string, grammar, parsing_table):
    stack = ['$','E']
    pointer = 0
    while len(stack) > 0:
        top = stack[-1]
        if top in grammar:
            if (pointer < len(input_string) and input_string[pointer] in parsing_table[top]) or ('ε' in parsing_table[top]):
                production = parsing_table[top].get(input_string[pointer]) or parsing_table[top].get('ε')
                if not production:
                    return False
                stack.pop()
                if production[0] != 'ε':
                    for symbol in reversed(production[0].split()):
                        stack.append(symbol)
            else:
                return False
        else:
            stack.pop()
            pointer += 1

    if pointer != len(input_string):
        return False
    return True

def find_epsilon_producing_non_terminals(grammar):
    """
    Finds the non-terminals which produce epsilon in the given grammar.
    """
    epsilon_producers = set()

    # Initial scan for direct producers
    for non_terminal, productions in grammar.items():
        for production in productions:
            if production == 'ε' or production == '':
                epsilon_producers.add(non_terminal)

    # Iterative scan for indirect producers
    size_before = 0
    while size_before != len(epsilon_producers):
        size_before = len(epsilon_producers)

        for non_terminal, productions in grammar.items():
            for production in productions:
                if all(symbol in epsilon_producers for symbol in production):
                    epsilon_producers.add(non_terminal)
                    break

    return epsilon_producers


# Test the functions
sample_grammar_text = """
A -> aBc | d
B -> e | ε
"""

grammar = parse_grammar_from_text(sample_grammar_text)
epsilon_producers = find_epsilon_producing_non_terminals(grammar)

grammar, epsilon_producers


def compute_first(grammar, epsilon_producers):
    """
    Computes the FIRST set for each non-terminal in the grammar.
    """
    first = {non_terminal: set() for non_terminal in grammar.keys()}

    # Initialization for terminals
    for non_terminal, productions in grammar.items():
        for production in productions:
            if production[0] not in grammar.keys():  # if it's a terminal
                first[non_terminal].add(production[0])
            elif production == 'ε':
                first[non_terminal].add('ε')

    # Iteratively compute FIRST sets
    changed = True
    while changed:
        changed = False
        for non_terminal, productions in grammar.items():
            for production in productions:
                for symbol in production:
                    # If symbol is terminal
                    if symbol not in grammar.keys():
                        break
                    # If symbol is non-terminal
                    first[non_terminal].update(first[symbol])
                    if symbol not in epsilon_producers:
                        break
                    if symbol == production[-1] and symbol in epsilon_producers:
                        first[non_terminal].add('ε')

                new_items_count = len(first[non_terminal])
                if new_items_count > len(first[non_terminal]):
                    changed = True

    return first


def compute_follow(grammar, first, epsilon_producers):
    """
    Computes the FOLLOW set for each non-terminal in the grammar.
    """
    follow = {non_terminal: set() for non_terminal in grammar.keys()}
    follow[next(iter(grammar.keys()))].add('$')  # Add $ to the start symbol

    changed = True
    while changed:
        changed = False
        for non_terminal, productions in grammar.items():
            for production in productions:
                for i, symbol in enumerate(production):
                    if symbol in grammar.keys():  # If symbol is a non-terminal
                        # All but the last symbol
                        if i < len(production) - 1:
                            next_symbol = production[i + 1]
                            if next_symbol in grammar.keys():
                                follow[symbol].update(first[next_symbol] - {'ε'})
                                if next_symbol in epsilon_producers:
                                    follow[symbol].update(follow[non_terminal])
                            else:
                                follow[symbol].add(next_symbol)
                        # If the symbol is the last one in the production
                        else:
                            follow[symbol].update(follow[non_terminal])

                new_items_count = len(follow[non_terminal])
                if new_items_count > len(follow[non_terminal]):
                    changed = True

    return follow


# Test the functions
first_sets = compute_first(grammar, epsilon_producers)
follow_sets = compute_follow(grammar, first_sets, epsilon_producers)

first_sets, follow_sets


def build_parsing_table(grammar, first, follow, epsilon_producers):
    """
    Builds the LL(1) parsing table.
    """
    table = {}
    for non_terminal in grammar.keys():
        table[non_terminal] = {}

        for production in grammar[non_terminal]:
            first_symbols = set()

            for symbol in production:
                if symbol in grammar.keys():
                    first_symbols.update(first[symbol])
                    if symbol not in epsilon_producers:
                        break
                else:
                    first_symbols.add(symbol)
                    break

            for symbol in first_symbols:
                if symbol != 'ε':
                    if symbol not in table[non_terminal]:
                        table[non_terminal][symbol] = []
                    table[non_terminal][symbol].append(production)

            if 'ε' in first_symbols or production == 'ε':
                for symbol in follow[non_terminal]:
                    if symbol not in table[non_terminal]:
                        table[non_terminal][symbol] = []
                    table[non_terminal][symbol].append(production)

    return table


# Test the function
parsing_table = build_parsing_table(grammar, first_sets, follow_sets, epsilon_producers)
parsing_table


def ast_visualization(grammar, parsing_table, input_string):
    """
    Builds and visualizes the Abstract Syntax Tree (AST) for the given input string using the provided grammar
    and parsing table.
    """
    stack = ['$']
    start_symbol = next(iter(grammar.keys()))
    stack.append(start_symbol)

    graph = nx.DiGraph()
    graph.add_node(start_symbol)

    prev_node = start_symbol
    idx = 0
    while stack:
        top = stack[-1]

        if idx >= len(input_string):
            break

        if top in parsing_table and input_string[idx] in parsing_table[top]:
            production = parsing_table[top][input_string[idx]][0]
            stack.pop()

            for symbol in reversed(production):
                if symbol != 'ε':
                    stack.append(symbol)
                    graph.add_node(symbol)
                    graph.add_edge(prev_node, symbol)
                    prev_node = symbol

            if production == 'ε':
                epsilon_node = 'ε' + str(idx)
                graph.add_node(epsilon_node, label='ε')
                graph.add_edge(prev_node, epsilon_node)
                prev_node = epsilon_node

        else:
            if top == input_string[idx]:
                stack.pop()
                idx += 1
            else:
                raise ValueError(f"Parsing error: Unexpected token '{input_string[idx]}' at position {idx}.")

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=1500, node_color='pink')
    plt.title("Abstract Syntax Tree (AST)")
    plt.show()


# Test the function with the sample input string "aec"
ast_visualization(grammar, parsing_table, "aec")

def parse_grammar_from_text_k(grammar_text_k):
    grammar_k = {}
    lines = grammar_text_k.strip().split('\n')
    for line in lines:
        left, right = line.split('->')
        left = left.strip()
        productions = right.strip().split('|')
        grammar_k[left] = [prod.strip() for prod in productions]
    return grammar_k
def find_epsilon_producing_non_terminals_k(grammar_k):
    epsilon_producers_k = set()
    for non_terminal, productions in grammar_k.items():
        for production in productions:
            if production == 'ε' or production == '':
                epsilon_producers_k.add(non_terminal)
    size_before = 0
    while size_before != len(epsilon_producers_k):
        size_before = len(epsilon_producers_k)
        for non_terminal, productions in grammar_k.items():
            for production in productions:
                if all(symbol in epsilon_producers_k for symbol in production):
                    epsilon_producers_k.add(non_terminal)
                    break
    return epsilon_producers_k
def compute_first_k(grammar_k, k=2):
    first_k = {non_terminal: set() for non_terminal in grammar_k.keys()}
    for non_terminal, productions in grammar_k.items():
        for production in productions:
            if len(production) < k:
                first_k[non_terminal].add(production)
            else:
                first_k[non_terminal].add(production[:k])
    changed = True
    while changed:
        changed = False
        for non_terminal, productions in grammar_k.items():
            for production in productions:
                prefix = ""
                for i, symbol in enumerate(production):
                    if symbol not in grammar_k.keys():
                        prefix += symbol
                        if len(prefix) == k:
                            break
                    else:
                        possible_prefixes = [prefix + item for item in first_k[symbol]]
                        for p in possible_prefixes:
                            if len(p) > k:
                                p = p[:k]
                            if p not in first_k[non_terminal]:
                                first_k[non_terminal].add(p)
                                changed = True
                        if len(prefix) == k:
                            break
    return first_k
def compute_follow_k(grammar_k, first_k, k=2):
    follow_k = {non_terminal: set() for non_terminal in grammar_k.keys()}
    start_symbol = next(iter(grammar_k.keys()))
    follow_k[start_symbol].add('$' * k)
    changed = True
    while changed:
        changed = False
        for non_terminal, productions in grammar_k.items():
            for production in productions:
                for i, symbol in enumerate(production):
                    if symbol in grammar_k.keys():
                        suffix = production[i+1:i+1+k]
                        while len(suffix) < k:
                            for follow_item in follow_k[non_terminal]:
                                suffix += follow_item
                                if len(suffix) > k:
                                    suffix = suffix[:k]
                                    break
                        if suffix not in follow_k[symbol]:
                            follow_k[symbol].add(suffix)
                            changed = True
    return follow_k
def build_parsing_table_k(grammar_k, first_k, follow_k, k=2):
    table_k = {}
    for non_terminal in grammar_k.keys():
        table_k[non_terminal] = {}
        for production in grammar_k[non_terminal]:
            first_symbols = set()
            temp_symbols = []
            for symbol in production:
                if symbol in grammar_k:
                    for first_symbol in first_k[symbol]:
                        temp_symbols.append(first_symbol)
                else:
                    temp_symbols.append(symbol)
            for i in range(len(temp_symbols) - k + 1):
                combined = "".join(temp_symbols[i:i+k])
                first_symbols.add(combined)
            for entry in first_symbols:
                if entry not in table_k[non_terminal]:
                    table_k[non_terminal][entry] = []
                table_k[non_terminal][entry].append(production)
    return table_k

class RecursiveDescentParserWithAST:
    def __init__(self, input_string):
        self.input = input_string
        self.index = 0
        self.ast = nx.DiGraph()
        self.node_count = 0

    def lookahead(self):
        return self.input[self.index] if self.index < len(self.input) else None

    def consume(self, char):
        if self.lookahead() == char:
            self.index += 1
            return True
        return False

    def E(self):
        node = self.new_node('E')
        if self.T():
            child1 = self.last_node
            if self.consume('+'):
                child2 = self.new_node('+')
                if self.E():
                    child3 = self.last_node
                    self.ast.add_edges_from([(node, child1), (node, child2), (node, child3)])
                    self.last_node = node
                    return True
            elif self.consume('-'):
                child2 = self.new_node('-')
                if self.E():
                    child3 = self.last_node
                    self.ast.add_edges_from([(node, child1), (node, child2), (node, child3)])
                    self.last_node = node
                    return True
            else:
                self.ast.add_edge(node, child1)
                self.last_node = node
                return True
        return False

    def T(self):
        node = self.new_node('T')
        if self.F():
            child1 = self.last_node
            if self.consume('*'):
                child2 = self.new_node('*')
                if self.T():
                    child3 = self.last_node
                    self.ast.add_edges_from([(node, child1), (node, child2), (node, child3)])
                    self.last_node = node
                    return True
            elif self.consume('/'):
                child2 = self.new_node('/')
                if self.T():
                    child3 = self.last_node
                    self.ast.add_edges_from([(node, child1), (node, child2), (node, child3)])
                    self.last_node = node
                    return True
            else:
                self.ast.add_edge(node, child1)
                self.last_node = node
                return True
        return False

    def F(self):
        node = self.new_node('F')
        if self.consume('('):
            child1 = self.new_node('(')
            if self.E():
                child2 = self.last_node
                if self.consume(')'):
                    child3 = self.new_node(')')
                    self.ast.add_edges_from([(node, child1), (node, child2), (node, child3)])
                    self.last_node = node
                    return True
        elif self.lookahead().isalnum():
            child1 = self.new_node(self.lookahead())
            self.index += 1
            self.ast.add_edge(node, child1)
            self.last_node = node
            return True
        return False

    def new_node(self, label):
        self.node_count += 1
        self.ast.add_node(self.node_count, label=label)
        return self.node_count

    def parse(self):
        result = self.E() and self.index == len(self.input)
        return result, self.ast if result else None

# Test the recursive descent parser
parser = RecursiveDescentParserWithAST("a+b*c")
success, ast = parser.parse()

# Visualize the AST
if success:
    pos = nx.spring_layout(ast)
    labels = {node: ast.nodes[node]['label'] for node in ast.nodes()}
    nx.draw(ast, pos, labels=labels, with_labels=True, node_size=3000, node_color="skyblue", font_size=15)
else:
    print("Parsing failed.")


def main():
    input_string = input("Enter the string to be parsed: ")

    print("\n=== LL(1) PARSER ===")

    # LL(1) Grammar
    grammar_text = """
    E -> T E'
    E' -> + T E' | ε
    T -> F T'
    T' -> * F T' | ε
    F -> ( E ) | id
    """
    print("Grammar:\n", grammar_text)

    # Parse Grammar
    grammar = parse_grammar_from_text(grammar_text)
    print("Parsed Grammar:", grammar)

    # Find Epsilon Producing Non-terminals
    epsilon_producers = find_epsilon_producing_non_terminals(grammar)
    print("Epsilon Producing Non-terminals:", epsilon_producers)

    # Compute FIRST sets
    first_sets = compute_first(grammar, epsilon_producers)
    print("FIRST sets:", first_sets)

    # Compute FOLLOW sets
    follow_sets = compute_follow(grammar, first_sets, epsilon_producers)
    print("FOLLOW sets:", follow_sets)

    # Build Parsing Table
    parsing_table = build_parsing_table(grammar, first_sets, follow_sets, epsilon_producers)
    print("Parsing Table:", parsing_table)

    # LL(1) Parsing
    success = ll1_parse(input_string, grammar, parsing_table)
    print(f"LL(1) Parsing of '{input_string}': {'Success' if success else 'Failure'}")

    print("\n=== LL(k) PARSER (k=2) ===")

    # LL(k) Grammar
    grammar_text_k = """
    S -> aabA | aacB
    A -> d
    B -> d
    """
    print("Grammar:\n", grammar_text_k)

    # Parse Grammar
    grammar_k = parse_grammar_from_text_k(grammar_text_k)
    print("Parsed Grammar:", grammar_k)

    # Compute FIRST_2 sets
    first_sets_k = compute_first_k(grammar_k, k=2)
    print("FIRST_2 sets:", first_sets_k)

    # Compute FOLLOW_2 sets
    follow_sets_k = compute_follow_k(grammar_k, first_sets_k, k=2)
    print("FOLLOW_2 sets:", follow_sets_k)

    # Build Parsing Table for LL(2)
    parsing_table_k = build_parsing_table_k(grammar_k, first_sets_k, follow_sets_k, k=2)
    print("Parsing Table LL(2):", parsing_table_k)

    print("\n=== RECURSIVE DESCENT PARSER ===")

    # Grammar is implicit in the RecursiveDescentParserWithAST implementation
    print("Grammar is for simple arithmetic expressions: E -> T + E | T - E | T, T -> F * T | F / T | F, F -> (E) | id")

    # Recursive Descent Parsing
    # В функции main после Recursive Descent Parsing
    parser = RecursiveDescentParserWithAST(input_string)
    success, ast = parser.parse()
    print(f"Recursive Descent Parsing of '{input_string}': {'Success' if success else 'Failure'}")

    # Добавим визуализацию дерева разбора в случае успеха
    if success:
        pos = nx.spring_layout(ast)
        labels = {node: ast.nodes[node]['label'] for node in ast.nodes()}
        nx.draw(ast, pos, labels=labels, with_labels=True, node_size=3000, node_color="skyblue", font_size=15)
        plt.show()

    return success

main()
