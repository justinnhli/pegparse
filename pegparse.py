#!/usr/bin/env python3

import re
from os.path import dirname, join as join_path
from textwrap import indent

EBNF_GRAMMAR = join_path(dirname(__file__), 'ebnf.ebnf')

EBNF_DEFS = {
    'Syntax': ('CONJUNCT', ('ONEORMORE', ('CONJUNCT', ('ZEROORMORE', ('CONJUNCT', 'EmptyLine')), 'Definition', 'newline')), ('ZEROORMORE', ('CONJUNCT', 'EmptyLine'))),
    'Definition': ('CONJUNCT', 'Identifier', 'Whitespace', '"= "', 'Expression', '";"'),
    'Expression': ('DISJUNCT', 'Disjunct', 'Except', 'Conjunct'),
    'Conjunct': ('CONJUNCT', 'Item', ('ZEROORMORE', ('CONJUNCT', '" "', 'Item'))),
    'Disjunct': ('CONJUNCT', 'Atom', ('ONEORMORE', ('CONJUNCT', 'newline', 'Whitespace', '"| "', 'Atom'))),
    'Except': ('CONJUNCT', 'Atom', ('ONEORMORE', ('CONJUNCT', 'newline', 'Whitespace', '"- "', 'Atom'))),
    'Item': ('DISJUNCT', 'Repetition', 'Atom'),
    'Repetition': ('DISJUNCT', 'ZeroOrMore', 'ZeroOrOne', 'OneOrMore'),
    'ZeroOrMore': ('CONJUNCT', '"( "', 'Conjunct', '" )*"'),
    'ZeroOrOne': ('CONJUNCT', '"( "', 'Conjunct', '" )?"'),
    'OneOrMore': ('CONJUNCT', '"( "', 'Conjunct', '" )+"'),
    'Atom': ('DISJUNCT', 'Identifier', 'Reserved', 'Literal'),
    'Identifier': ('CONJUNCT', ('ONEORMORE', ('CONJUNCT', 'upper', ('ZEROORMORE', ('CONJUNCT', 'lower'))))),
    'Reserved': ('CONJUNCT', ('ONEORMORE', ('CONJUNCT', 'lower'))),
    'Literal': ('DISJUNCT', 'DString', 'SString'),
    'DString': ('CONJUNCT', '\'"\'', ('ZEROORMORE', ('CONJUNCT', 'NoDQuote')), '\'"\''),
    'SString': ('CONJUNCT', '"\'"', ('ZEROORMORE', ('CONJUNCT', 'NoSQuote')), '"\'"'),
    'NoDQuote': ('EXCEPT', 'print', '\'"\''),
    'NoSQuote': ('EXCEPT', 'print', '"\'"'),
    'Whitespace': ('CONJUNCT', ('ONEORMORE', ('CONJUNCT', 'blank'))),
    'EmptyLine': ('CONJUNCT', ('ZEROORONE', ('CONJUNCT', '"#"', ('ZEROORMORE', ('CONJUNCT', 'print')))), 'newline'),
}


def create_parser_from_file(filepath):
    with open(filepath) as fd:
        ebnf = fd.read()
    return create_parser(ebnf)


def create_parser(ebnf):
    return PEGParser(EBNFWalker().parse(ebnf))


def one_line_format(string):
    string = re.sub(r'\t', r'\\t', string)
    if '\n' in string:
        string = string[:string.index('\n')]
    return string


class ASTNode:

    def __init__(self, term, children, string, start_pos, end_pos):
        self.term = term
        self.children = children
        self.string = string
        self.start_pos = start_pos
        self.end_pos = end_pos

    @property
    def match(self):
        return self.string[self.start_pos:self.end_pos]

    @property
    def line_num(self):
        return self.string.count('\n', 0, self.start_pos) + 1

    @property
    def column(self):
        prev_newline = self.string.rfind('\n', 0, self.start_pos)
        if prev_newline == -1:
            column = 0
        else:
            column = self.start_pos - prev_newline
        return column + 1

    def first_descendant(self, path=None):
        path = ('*' if path is None else path).split('/')
        result = self
        for term in path:
            if term == '*':
                result = result.children[0]
            else:
                children = tuple(
                    child for child in result.children
                    if child.term == term
                )
                if children:
                    result = children[0]
                else:
                    return ()
        return result

    def descendants(self, path=None):
        path = ('*' if path is None else path).split('/')
        cur_gen = (self, )
        for term in path:
            next_gen = []
            for adult in cur_gen:
                next_gen.extend(adult.children)
            if term == '*':
                cur_gen = tuple(next_gen)
            else:
                cur_gen = tuple(child for child in next_gen if child.term == term)
        return cur_gen

    def pretty_print(self, indent_level=0):
        print('{}{}: {}'.format(indent_level * 4 * ' ', self.term, one_line_format(self.match)))
        for child in self.children:
            child.pretty_print(indent_level + 1)


class PEGParser:
    CORE_DEFS = {
        'empty': r'',
        'blank': r'[ \t]',
        'digit': r'[0-9]',
        'upper': r'[A-Z]',
        'lower': r'[a-z]',
        'alpha': r'[A-Za-z]',
        'alnum': r'[0-9A-Za-z]',
        'punct': r"[-!\"#$%&'()*+,./:;<=>?@[\\\]^_`{|}~]",
        'print': r'[ -~]',
        'unicode': r'[^\x00-\x7F]',
        'newline': r'\n',
        'tab': r'\t',
    }

    def __init__(self, syntax, debug=False):
        self.custom_defs = syntax
        self.debug = debug
        self.cache = {}
        self.depth = 0
        self.trace = []
        self.max_position = 0

    def parse(self, string, term):
        ast, parsed = self._partial_parse(string, term)
        if ast and parsed == len(string):
            return ast
        trace = []
        for position, term in self.trace:
            trace.append('Failed to match {} at position {}'.format(term, position))
            trace.append('  ' + one_line_format(string[position:]))
        message = 'only parsed {} of {} characters:\n'.format(parsed, len(string)) + indent('\n'.join(trace), '  ')
        raise SyntaxError(message)

    def parse_file(self, filepath, term):
        with open(filepath) as fd:
            return self.parse(fd.read(), term)

    def _partial_parse(self, string, term):
        self.cache = {}
        self.depth = 0
        self.trace = []
        self.max_position = 0
        ast, parsed = self._dispatch(string, term, 0)
        self.trace = list(reversed(self.trace[1:]))
        return ast, parsed

    def _dispatch(self, string, term, position=0):
        if isinstance(term, tuple) and hasattr(self, '_match_{}'.format(term[0].lower())):
            return getattr(self, '_match_{}'.format(term[0].lower()))(string, term, position)
        elif isinstance(term, str):
            ast, pos = self._get_cached(term, position)
            if ast:
                return ast, pos
            elif term in self.custom_defs:
                return self._match_custom(string, term, position)
            elif term in PEGParser.CORE_DEFS:
                return self._match_core(string, term, position)
            elif re.match(r"^'[^']*'$", term) or re.match(r'^"[^"]*"$', term):
                return self._match_literal(string, term, position)
            else:
                raise NameError('Unknown terminal {}'.format(term))
        self._debug_print('unknown non-terminal: {}'.format(term))
        return self._fail(term, position)

    def _match_zeroormore(self, string, terms, position):
        terms = terms[1]
        children = []
        last_pos = position
        ast, pos = self._dispatch(string, terms, position)
        while ast:
            children.extend(ast.children)
            last_pos = pos
            ast, pos = self._dispatch(string, terms, pos)
        return ASTNode('ZEROORMORE', children, string, position, last_pos), last_pos

    def _match_zeroorone(self, string, terms, position):
        terms = terms[1]
        ast, pos = self._dispatch(string, terms, position)
        if ast:
            return ast, pos
        return self._dispatch(string, 'empty', position)

    def _match_oneormore(self, string, terms, position):
        terms = terms[1]
        ast, pos = self._dispatch(string, terms, position)
        if not ast:
            return ast, pos
        children = ast.children
        last_pos = pos
        ast, pos = self._dispatch(string, terms, pos)
        while ast:
            children.extend(ast.children)
            last_pos = pos
            ast, pos = self._dispatch(string, terms, pos)
        return ASTNode('ONEORMORE', children, string, position, last_pos), last_pos

    def _match_conjunct(self, string, terms, position):
        children = []
        pos = position
        for term in terms[1:]:
            child_ast, child_pos = self._dispatch(string, term, pos)
            if child_ast:
                if isinstance(term, tuple) and (term[0] in ('ZEROORONE', 'ZEROORMORE', 'ONEORMORE')):
                    children.extend(child_ast.children)
                else:
                    children.append(child_ast)
                pos = child_pos
            else:
                return None, child_pos
        return ASTNode('AND', children, string, position, pos), pos

    def _match_disjunct(self, string, terms, position):
        for term in terms[1:]:
            ast, pos = self._dispatch(string, term, position)
            if ast:
                return ast, pos
        return None, position

    def _match_except(self, string, terms, position):
        ast, pos = self._dispatch(string, terms[1], position)
        if not ast:
            return self._fail(terms[1], position)
        for term in terms[2:]:
            nast = self._dispatch(string, term, position)[0]
            if nast and ast.match == nast.match:
                return self._fail(term, position)
        return ast, pos

    def _match_custom(self, string, term, position):
        expression = self.custom_defs[term]
        self._debug_print('parse called at position {} with {} >>>{}'.format(
            position, term, one_line_format(string[position:position+32])
        ))
        max_position = self.max_position
        self.depth += 1
        ast = self._dispatch(string, expression, position)[0]
        self.depth -= 1
        if self.max_position >= max_position and (not ast or len(self.trace) > 1):
            self.trace.append((position, term))
        if not ast:
            return self._fail(term, position)
        if isinstance(expression, tuple) and expression[0] == 'DISJUNCT':
            ast = ASTNode(term, [ast], string, position, position + len(ast.match))
        else:
            ast.term = term
        return self._cache_and_return(term, position, ast)

    def _match_core(self, string, term, position):
        match = re.match(PEGParser.CORE_DEFS[term], string[position:])
        if match:
            ast = ASTNode(term, [], string, position, position + len(match.group(0)))
            return self._cache_and_return(term, position, ast)
        return self._fail(term, position)

    def _match_literal(self, string, term, position):
        if string[position:].find(term[1:-1]) == 0:
            ast = ASTNode(term, [], string, position, position + len(term[1:-1]))
            return self._cache_and_return(term, position, ast)
        return self._fail(term, position)

    def _fail(self, term, position):
        if term in self.custom_defs:
            self._debug_print('failed to match {} at position {}'.format(term, position))
        return None, position

    def _cache_and_return(self, term, position, ast):
        self.cache[(term, position)] = ast
        return self._get_cached(term, position)

    def _get_cached(self, term, position):
        if (term, position) in self.cache:
            if term in self.custom_defs:
                self._debug_print('matched {} at position {}'.format(term, position))
            ast = self.cache[(term, position)]
            new_position = position + len(ast.match)
            if new_position > self.max_position:
                self.max_position = new_position
                self.trace = [(position, term), ]
            return ast, new_position
        return None, position

    def _debug_print(self, obj):
        if self.debug:
            print('    ' * self.depth + str(obj))


class ASTWalker:

    class EmptySentinel:
        pass

    def __init__(self, parser, term):
        self.parser = parser
        self.term = term
        self._terms_to_expand = set(term[6:] for term in dir(self) if term.startswith('parse_'))
        noskips = list(self._terms_to_expand)
        while noskips:
            noskip = noskips.pop()
            for term, definition in self.parser.custom_defs.items():
                if term in self._terms_to_expand:
                    continue
                if ASTWalker.term_in_definition(noskip, definition):
                    noskips.append(term)
                    self._terms_to_expand.add(term)

    def _postorder_traversal(self, ast, depth=0):
        results = []
        for child in ast.descendants('*'):
            if child.term not in self._terms_to_expand:
                continue
            result, parsed = self._postorder_traversal(child, depth=depth + 1)
            if not isinstance(result, ASTWalker.EmptySentinel):
                if parsed:
                    results.append(result)
                else:
                    results.extend(result)
        function = 'parse_' + ast.term
        if hasattr(self, function):
            return getattr(self, function)(ast, tuple(results)), True
        elif results:
            return tuple(results), False
        else:
            return ASTWalker.EmptySentinel(), False

    def parse(self, text, term=None):
        if term is None:
            term = self.term
        ast = self.parser.parse(text, term)
        return self.parse_ast(ast)

    def parse_file(self, filepath, term=None):
        with open(filepath) as fd:
            return self.parse(fd.read(), term)

    def parse_ast(self, ast):
        return self._postorder_traversal(ast)[0]

    @staticmethod
    def term_in_definition(term, definition):
        return any(
            (term == element or (isinstance(element, tuple) and ASTWalker.term_in_definition(term, element)))
            for element in definition
        )

class EBNFWalker(ASTWalker):

    def __init__(self):
        super().__init__(PEGParser(EBNF_DEFS), 'Syntax')

    def flatten(self, ast, results):
        return tuple((ast.term.upper(), *results))

    def parse_Syntax(self, ast, results):
        return dict(results)

    def parse_Definition(self, ast, results):
        return tuple(results)

    def parse_Disjunct(self, ast, results):
        return self.flatten(ast, results)

    def parse_Except(self, ast, results):
        return self.flatten(ast, results)

    def parse_Conjunct(self, ast, results):
        return self.flatten(ast, results)

    def parse_Repetition(self, ast, results):
        return self.flatten(ast.first_descendant('*'), results)

    def parse_Reserved(self, ast, results):
        return ast.match

    def parse_Identifier(self, ast, results):
        return ast.match

    def parse_Literal(self, ast, results):
        return ast.match


def main():
    from argparse import ArgumentParser
    from fileinput import input as fileinput
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-e', dest='expression', help='starting expression; if omitted, first defined term is used')
    arg_parser.add_argument('-g', dest='grammar', default=EBNF_GRAMMAR, help='EBNF grammar file')
    arg_parser.add_argument('-v', dest='verbose', action='store_true', help='show what the parser is doing')
    arg_parser.add_argument('file', default='-', nargs='?', help='text file to be parsed')
    args = arg_parser.parse_args()
    grammar = ''
    with open(args.grammar, 'r') as fd:
        grammar = fd.read()
    parser = create_parser(grammar, debug=args.verbose)
    if args.expression:
        term = args.expression
    else:
        term = PEGParser(EBNF_DEFS).parse(grammar, 'Syntax').first_descendant('Definition/Identifier').match
    contents = ''.join(fileinput(files=args.file))
    parser.parse(contents, term).pretty_print()


if __name__ == '__main__':
    main()
