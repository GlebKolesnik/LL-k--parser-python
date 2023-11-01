import ply.lex as lex
import ply.yacc as yacc

# ---- LEXER PART ----

# int x = 5; if (x == 5) { x = 10; } while (x > 0) { x = x - 1; }
# List of token names
tokens = (
    'NUMBER',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'LPAREN',
    'RPAREN',
    'ID',
    'EQUALS',
    'IF',
    'ELSE',
    'WHILE',
    'FOR',
    'INT',
    'CHAR',
    'FLOAT',
    'LBRACE',
    'RBRACE',
    'SEMICOLON',
    'EQUALTO',
    'GT',
    'LT',
    'COMMA'
)

# Regular expression rules for simple tokens
t_PLUS    = r'\+'
t_MINUS   = r'-'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_EQUALS  = r'='
t_LBRACE  = r'{'
t_RBRACE  = r'}'
t_SEMICOLON = r';'
t_COMMA    = r','
t_EQUALTO = r'=='
t_GT     = r'>'
t_LT     = r'<'

# Regular expressions for ID and reserved words
reserved = {
    'if'    : 'IF',
    'else'  : 'ELSE',
    'while' : 'WHILE',
    'for'   : 'FOR',
    'int'   : 'INT',
    'char'  : 'CHAR',
    'float' : 'FLOAT'
}

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')  # Check for reserved words
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'

# Error handling rule
def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

# ---- PARSER PART ----

precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE'),
    ('right', 'UMINUS'),
)

def p_program(t):
    'program : statements'
    t[0] = t[1]

def p_statements(t):
    '''statements : statement
                  | statement statements'''
    if len(t) == 2:
        t[0] = [t[1]]
    else:
        t[0] = [t[1]] + t[2]

def p_declaration(t):
    '''declaration : type_specifier ID SEMICOLON
                   | type_specifier ID EQUALS expression SEMICOLON'''
    if len(t) == 4:
        t[0] = ('declare', t[1], t[2])
    else:
        t[0] = ('declare_assign', t[1], t[2], t[4])

def p_type_specifier(t):
    '''type_specifier : INT
                      | CHAR
                      | FLOAT'''
    t[0] = t[1]

def p_statement_assign(t):
    'statement : ID EQUALS expression SEMICOLON'
    t[0] = ('assign', t[1], t[3])

def p_statement_expr(t):
    'statement : expression SEMICOLON'
    t[0] = t[1]

def p_statement_if(t):
    '''statement : IF LPAREN expression RPAREN LBRACE statements RBRACE
                 | IF LPAREN expression RPAREN LBRACE statements RBRACE ELSE LBRACE statements RBRACE'''
    if len(t) == 8:
        t[0] = ('if', t[3], t[6])
    else:
        t[0] = ('if-else', t[3], t[6], t[10])

def p_statement_while(t):
    'statement : WHILE LPAREN expression RPAREN LBRACE statements RBRACE'
    t[0] = ('while', t[3], t[6])

def p_statement_for(t):
    'statement : FOR LPAREN expression SEMICOLON expression SEMICOLON expression RPAREN LBRACE statements RBRACE'
    t[0] = ('for', t[3], t[5], t[7], t[10])

def p_expression_binop(t):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression'''
    t[0] = (t[2], t[1], t[3])

def p_expression_uminus(t):
    'expression : MINUS expression %prec UMINUS'
    t[0] = ('uminus', t[2])

def p_expression_group(t):
    'expression : LPAREN expression RPAREN'
    t[0] = t[2]

def p_expression_number(t):
    'expression : NUMBER'
    t[0] = t[1]

def p_expression_name(t):
    'expression : ID'
    t[0] = t[1]

def p_expression_binop(t):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression EQUALTO expression
                  | expression GT expression
                  | expression LT expression'''
    t[0] = (t[2], t[1], t[3])

def p_statement_decl_assign(t):
    'statement : type_specifier ID EQUALS expression SEMICOLON'
    t[0] = ('declare_assign', t[1], t[2], t[4])



def p_error(t):
    if t:
        print(f"Syntax error at '{t.value}'")
    else:
        print("Syntax error at EOF")


# Build the parser
parser = yacc.yacc()
