from graphviz import Digraph
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz\\bin'


# Повторно определяем все функции
def compute_first(grammar):
    first_set = {key: set() for key in grammar}

    def first(symbol):
        if symbol not in grammar:
            return {symbol}
        if first_set[symbol]:
            return first_set[symbol]
        for production in grammar[symbol]:
            if production == "ε":
                first_set[symbol].add("ε")
            else:
                for s in production:
                    first_set[symbol] |= (first(s) - {"ε"})
                    if "ε" not in first(s):
                        break
                else:
                    first_set[symbol].add("ε")
        return first_set[symbol]

    for non_terminal in grammar:
        first(non_terminal)

    return first_set


def compute_follow(grammar, first_set):
    follow_set = {key: set() for key in grammar}
    follow_set[next(iter(grammar))] = {"$"}

    def first_of_string(string):
        result = set()
        for s in string:
            result |= (first_set.get(s, {s}) - {"ε"})
            if "ε" not in first_set.get(s, {s}):
                break
        else:
            result.add("ε")
        return result

    changes_made = True
    while changes_made:
        changes_made = False
        for non_terminal, productions in grammar.items():
            current_follow = follow_set[non_terminal].copy()
            for production in productions:
                for i, symbol in enumerate(production):
                    if symbol in grammar:
                        follow_set[symbol] |= (first_of_string(production[i + 1:]) - {"ε"})
                        if "ε" in first_of_string(production[i + 1:]):
                            follow_set[symbol] |= follow_set[non_terminal]
            if current_follow != follow_set[non_terminal]:
                changes_made = True

    return follow_set


def construct_parsing_table(grammar, first_set, follow_set):
    parsing_table = {}

    def first_of_string(string):
        result = set()
        for s in string:
            result |= (first_set.get(s, {s}) - {"ε"})
            if "ε" not in first_set.get(s, {s}):
                break
        else:
            result.add("ε")
        return result

    for non_terminal, productions in grammar.items():
        for production in productions:
            first_of_production = first_of_string(production)
            for terminal in first_of_production:
                if terminal != "ε":
                    parsing_table[(non_terminal, terminal)] = production
            if "ε" in first_of_production:
                for terminal in follow_set[non_terminal]:
                    parsing_table[(non_terminal, terminal)] = production

    return parsing_table


def ll1_parser(input_string, grammar, parsing_table, start_symbol):
    stack = [start_symbol]
    input_string += "$"
    pointer = 0
    output = []

    while stack:
        top = stack[-1]
        current_input = input_string[pointer]
        if top in grammar:
            try:
                production = parsing_table[(top, current_input)]
                output.append(f"{top} -> {production}")
                stack.pop()
                for symbol in reversed(production):
                    if symbol != "ε":
                        stack.append(symbol)
            except KeyError:
                return False, [f"Error: No rule found for ({top}, {current_input})"]
        elif top == current_input:
            stack.pop()
            pointer += 1
        else:
            return False, [f"Error: Mismatch found with top of stack {top} and current input {current_input}"]

    if pointer != len(input_string):
        return False, [f"Error: Input not fully parsed. Remaining input: {input_string[pointer:]}"]

    return True, output


# Пример использования:
grammar_example = {
    "S": ["aAB", "bBA"],
    "A": ["a", "ε"],
    "B": ["b"]
}
first_set_example = compute_first(grammar_example)
follow_set_example = compute_follow(grammar_example, first_set_example)
parsing_table_example = construct_parsing_table(grammar_example, first_set_example, follow_set_example)
input_test = "aabb"
success, output = ll1_parser(input_test, grammar_example, parsing_table_example, "S")
success, output
# Добавляем команды вывода для демонстрации результатов

print("Grammar:", grammar_example)
print("\nFIRST sets:")
for non_terminal, first in first_set_example.items():
    print(f"FIRST({non_terminal}) = {first}")

print("\nFOLLOW sets:")
for non_terminal, follow in follow_set_example.items():
    print(f"FOLLOW({non_terminal}) = {follow}")

print("\nParsing Table:")
for (non_terminal, terminal), production in parsing_table_example.items():
    print(f"Table[{non_terminal}, {terminal}] = {production}")

print("\nParsing Result for input:", input_test)
if success:
    print("Parsing Successful!")
    for step in output:
        print(step)
else:
    print("Parsing Failed!")
    for error in output:
        print(error)

class Node:
    def __init__(self, type, children=None, value=None):
        self.type = type
        self.children = children or []
        self.value = value
class RecursiveDescentParserASTVerbose:
    def __init__(self, input_string):
        self.input = input_string
        self.pointer = 0

    def peek(self, k=1):
        return self.input[self.pointer:self.pointer + k]

    def consume(self):
        self.pointer += 1

    def E(self):
        print(f"Parsing E at {self.peek()}")
        children = []
        t_node = self.T()
        if t_node:
            children.append(t_node)
            if self.peek() == "+":
                print(f"Matched + at {self.peek()}")
                self.consume()
                e_node = self.E()
                if e_node:
                    children.append(Node("PLUS"))
                    children.append(e_node)
                    return Node("E", children)
            return Node("E", children)
        return None

    def T(self):
        print(f"Parsing T at {self.peek()}")
        children = []
        f_node = self.F()
        if f_node:
            children.append(f_node)
            if self.peek() == "*":
                print(f"Matched * at {self.peek()}")
                self.consume()
                t_node = self.T()
                if t_node:
                    children.append(Node("MULTIPLY"))
                    children.append(t_node)
                    return Node("T", children)
            return Node("T", children)
        return None

    def F(self):
        print(f"Parsing F at {self.peek()}")
        if self.peek() == "(":
            print(f"Matched ( at {self.peek()}")
            self.consume()
            e_node = self.E()
            if e_node and self.peek() == ")":
                print(f"Matched ) at {self.peek()}")
                self.consume()
                return Node("F", [e_node])
        elif self.peek().isalpha() or self.peek().isdigit():
            value = self.peek()
            print(f"Matched id at {self.peek()}")
            self.consume()
            return Node("F", value=value)
        return None

    def parse(self):
        root = self.E()
        if root and self.pointer == len(self.input):
            print("Parsing Successful!")
            return root
        print("Parsing Failed!")
        return None

# Test the verbose recursive descent parser to produce AST
parser_ast_verbose = RecursiveDescentParserASTVerbose("a+b*c")
ast_root_verbose = parser_ast_verbose.parse()

def visualize_ast(node, graph=None):
    if graph is None:
        graph = Digraph('AST', node_attr={'shape': 'box'})
        graph.attr(size='10,10')
    if node.value:
        graph.node(str(id(node)), label=f"{node.type}\n{node.value}")
    else:
        graph.node(str(id(node)), label=f"{node.type}")
    for child in node.children:
        graph.edge(str(id(node)), str(id(child)))
        visualize_ast(child, graph)
    return graph

# Для визуализации дерева
ast_graph = visualize_ast(ast_root_verbose)
ast_graph.view()  # Это откроет файл .pdf с визуализацией дерева