def parse_grammar(grammar_str):
    """
    Parses a grammar from its string representation.

    :param grammar_str: A string representation of the grammar.
    :return: A dictionary representation of the grammar.
    """
    grammar = {}

    # Split the grammar string by lines to get individual rules
    rules = grammar_str.strip().split("\n")

    for rule in rules:
        # Split the rule into left and right parts
        lhs, rhs = rule.split("->")
        lhs = lhs.strip()

        # Split the right hand side by the "|" symbol to get alternatives
        alternatives = [alt.strip() for alt in rhs.split("|")]

        grammar[lhs] = alternatives

    return grammar


# Test the function
grammar_str = """
S -> aBc
B -> b|ε
"""
parse_grammar(grammar_str)


def first(grammar, symbol, visited=None):
    """
    Compute the First set for a given symbol.

    :param grammar: The grammar in dictionary format.
    :param symbol: The symbol (non-terminal or terminal) to compute the First set for.
    :param visited: A set of already visited symbols to avoid infinite recursion.
    :return: The First set for the given symbol.
    """
    if visited is None:
        visited = set()

    # If the symbol is a terminal or epsilon, return it
    if symbol not in grammar or symbol == "ε":
        return {symbol}

    first_set = set()

    # If the symbol is a non-terminal, compute the First set for each production
    for production in grammar[symbol]:
        if production[0] not in visited:
            visited.add(production[0])
            for char in production:
                # Recursively compute the First set for each character in the production
                char_first = first(grammar, char, visited)

                first_set.update(char_first)

                # If the First set of the character does not contain epsilon, break
                if "ε" not in char_first:
                    break

    return first_set


# Test the function
grammar = parse_grammar(grammar_str)
first(grammar, 'S')


def follow(grammar, symbol, visited=None):
    """
    Compute the Follow set for a given non-terminal.

    :param grammar: The grammar in dictionary format.
    :param symbol: The non-terminal to compute the Follow set for.
    :param visited: A set of already visited symbols to avoid infinite recursion.
    :return: The Follow set for the given non-terminal.
    """
    if visited is None:
        visited = set()

    if symbol not in grammar:
        return set()

    follow_set = set()

    # Start symbol always has $ (end of input) in its Follow set
    if symbol == list(grammar.keys())[0]:
        follow_set.add("$")

    for lhs, productions in grammar.items():
        for production in productions:
            # Find the position of the symbol in the production
            if symbol in production:
                position = production.index(symbol)

                # Add everything from First set of the next symbol to the Follow set of the current symbol
                if position < len(production) - 1:
                    next_symbol = production[position + 1]
                    follow_set.update(first(grammar, next_symbol))

                    # If the First set of the next symbol contains epsilon, add Follow set of the LHS to the Follow set
                    # of the current symbol
                    if "ε" in first(grammar, next_symbol) and lhs not in visited:
                        visited.add(lhs)
                        follow_set.update(follow(grammar, lhs, visited))
                # If the symbol is the last in the production, add Follow set of the LHS to the Follow set of the symbol
                else:
                    if lhs not in visited:
                        visited.add(lhs)
                        follow_set.update(follow(grammar, lhs, visited))

    # Remove epsilon from the Follow set, as it should not be there
    follow_set.discard("ε")

    return follow_set


# Test the function
follow(grammar, 'B')


def build_parsing_table(grammar):
    """
    Build the LL(1) parsing table for the given grammar.

    :param grammar: The grammar in dictionary format.
    :return: The LL(1) parsing table in dictionary format.
    """
    table = {}

    for non_terminal, productions in grammar.items():
        table[non_terminal] = {}

        for production in productions:
            # For each terminal in First(production), add the production to the table
            first_set = first(grammar, production[0])
            for terminal in first_set:
                if terminal != "ε":
                    table[non_terminal][terminal] = production

            # If First(production) contains epsilon, add the production to the table for each terminal in Follow(non-terminal)
            if "ε" in first_set:
                follow_set = follow(grammar, non_terminal)
                for terminal in follow_set:
                    table[non_terminal][terminal] = production

    return table


# Test the function
parsing_table = build_parsing_table(grammar)
parsing_table


def ll1_parse(input_str, grammar, table):
    """
    Parse an input string using the LL(1) parsing table.

    :param input_str: The input string to parse.
    :param grammar: The grammar in dictionary format.
    :param table: The LL(1) parsing table.
    :return: A tuple (success, stack_trace, input_trace) indicating whether the parse was successful,
             and the traces of the stack and input for debugging.
    """
    # Initialize the stack with the start symbol and end of input symbol
    stack = ["$", list(grammar.keys())[0]]
    # Append end of input symbol to input
    input_str = list(input_str) + ["$"]

    # Lists to hold the trace of stack and input for debugging
    stack_trace = [list(stack)]
    input_trace = [list(input_str)]

    # While there are symbols left in the input and stack
    while input_str and stack:
        # Current symbols
        current_input = input_str[0]
        stack_top = stack[-1]

        # If the top of the stack is a terminal or end of input, match against input
        if stack_top not in grammar or stack_top == "$":
            if stack_top == current_input:
                stack.pop()
                input_str.pop(0)
            else:
                # Mismatch between input and stack
                return False, stack_trace, input_trace
        else:
            # Top of the stack is a non-terminal
            if current_input in table[stack_top]:
                production = table[stack_top][current_input]
                stack.pop()
                # Push the production onto the stack in reverse order
                if production != "ε":
                    for symbol in reversed(production):
                        stack.append(symbol)
            else:
                # No rule found in the table for the current input and stack top
                return False, stack_trace, input_trace

        # Append the current state of the stack and input to the traces
        stack_trace.append(list(stack))
        input_trace.append(list(input_str))

    # If both the input and the stack are empty, the parse is successful
    success = not stack and not input_str
    return success, stack_trace, input_trace


# Test the function
input_str = "abc"
ll1_parse(input_str, grammar, parsing_table)
