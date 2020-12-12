#!/usr/bin/env python3
"""A Pack Rat Parsing Expression Grammer parser."""

# pylint: disable = too-many-lines

import re
from argparse import ArgumentParser
from collections import namedtuple
from fileinput import input as fileinput
from pathlib import Path
from textwrap import indent
from typing import Any, Optional, Union, Literal, Tuple, List, Dict

PEGAtom = str
PEGOperator = Literal['CHOICE', 'SEQUENCE', 'ZERO_OR_MORE', 'ZERO_OR_ONE', 'ONE_OR_MORE', 'AND', 'NOT'] # pylint: disable = unsubscriptable-object
PEGExpression = Tuple[Any, ...] # should be Union[Tuple[PEGOperator, PEGExpression, ...], PEGAtom]
PEGRules = Dict[str, PEGExpression]

PEG_GRAMMAR = Path(__file__).parent / 'peg.peg'

# pylint: disable=line-too-long
PEG_DEFS = {
    'syntax': ('CHOICE', ('SEQUENCE', 'opt_space', ('ZERO_OR_MORE', ('CHOICE', ('SEQUENCE', 'definition', 'opt_space'))))),
    'definition': ('CHOICE', ('SEQUENCE', 'identifier', 'opt_space', '"="', 'opt_space', 'expression', 'opt_space', '";"')),
    'expression': ('CHOICE', ('SEQUENCE', 'choice')),
    'choice': ('CHOICE', ('SEQUENCE', ('ZERO_OR_ONE', ('CHOICE', ('SEQUENCE', '"|"', 'opt_space'))), 'sequence', ('ZERO_OR_MORE', ('CHOICE', ('SEQUENCE', 'opt_space', '"|"', 'opt_space', 'sequence'))))),
    'sequence': ('CHOICE', ('SEQUENCE', 'item', ('ZERO_OR_MORE', ('CHOICE', ('SEQUENCE', 'req_space', 'item'))))),
    'item': ('CHOICE', ('SEQUENCE', 'zero_or_more'), ('SEQUENCE', 'zero_or_one'), ('SEQUENCE', 'one_or_more'), ('SEQUENCE', 'and_predicate'), ('SEQUENCE', 'not_predicate'), ('SEQUENCE', 'term')),
    'zero_or_more': ('CHOICE', ('SEQUENCE', 'term', 'opt_space', '"*"')),
    'zero_or_one': ('CHOICE', ('SEQUENCE', 'term', 'opt_space', '"?"')),
    'one_or_more': ('CHOICE', ('SEQUENCE', 'term', 'opt_space', '"+"')),
    'and_predicate': ('CHOICE', ('SEQUENCE', '"&"', 'opt_space', 'term')),
    'not_predicate': ('CHOICE', ('SEQUENCE', '"!"', 'opt_space', 'term')),
    'term': ('CHOICE', ('SEQUENCE', 'paren'), ('SEQUENCE', 'atom')),
    'paren': ('CHOICE', ('SEQUENCE', '"("', 'opt_space', 'expression', 'opt_space', '")"')),
    'atom': ('CHOICE', ('SEQUENCE', 'identifier'), ('SEQUENCE', 'builtin'), ('SEQUENCE', 'literal')),
    'identifier': ('CHOICE', ('SEQUENCE', ('ONE_OR_MORE', ('CHOICE', ('SEQUENCE', 'LOWER'))), ('ZERO_OR_MORE', ('CHOICE', ('SEQUENCE', '"_"', ('ONE_OR_MORE', ('CHOICE', ('SEQUENCE', 'LOWER')))))))),
    'builtin': ('CHOICE', ('SEQUENCE', ('ONE_OR_MORE', ('CHOICE', ('SEQUENCE', 'UPPER'))))),
    'literal': ('CHOICE', ('SEQUENCE', 'd_string'), ('SEQUENCE', 's_string')),
    'd_string': ('CHOICE', ('SEQUENCE', '\'"\'', ('ZERO_OR_MORE', ('CHOICE', ('SEQUENCE', ('NOT', ('CHOICE', ('SEQUENCE', '\'"\''))), 'PRINT'))), '\'"\'')),
    's_string': ('CHOICE', ('SEQUENCE', '"\'"', ('ZERO_OR_MORE', ('CHOICE', ('SEQUENCE', ('NOT', ('CHOICE', ('SEQUENCE', '"\'"'))), 'PRINT'))), '"\'"')),
    'opt_space': ('CHOICE', ('SEQUENCE', ('ZERO_OR_MORE', ('CHOICE', ('SEQUENCE', 'space'))))),
    'req_space': ('CHOICE', ('SEQUENCE', ('ONE_OR_MORE', ('CHOICE', ('SEQUENCE', 'space'))))),
    'space': ('CHOICE', ('SEQUENCE', '"#"', ('ZERO_OR_MORE', ('CHOICE', ('SEQUENCE', 'PRINT'))), 'NEWLINE'), ('SEQUENCE', 'BLANK'), ('SEQUENCE', 'NEWLINE')),
}


def create_parser_from_file(filepath, debug=False):
    # type: (Path, bool) -> PEGParser
    """Create parser from a PEG grammar file.

    Parameters:
        filepath (str): Path to PEG file.
        debug (bool): Print debugging information. Defaults to False.

    Returns:
        PEGParser: A parser for the grammar.
    """
    with open(filepath) as fd:
        peg = fd.read()
    return create_parser(peg, debug=debug)


def create_parser(peg, debug=False):
    # type: (str, bool) -> PEGParser
    """Create parser from a PEG grammar.

    Parameters:
        peg (str): A PEG grammar.
        debug (bool): Print debugging information. Defaults to False.

    Returns:
        PEGParser: A parser for the grammar.
    """
    return PEGParser(PEGWalker().parse(peg), debug=debug)


def one_line_format(string):
    # type: (str) -> str
    """Escape tabs and spaces in a string.

    Parameters:
        string (str): The string to escape.

    Returns:
        str: The escaped string.
    """
    string = re.sub(r'\t', r'\\t', string)
    if '\n' in string:
        string = string[:string.index('\n')]
    return string


def index_to_line_col(string, index):
    # type: (str, int) -> Tuple[int, int]
    """Convert an index in a string to line and column number.

    Parameters:
        string (str): The string.
        index (int): The index.

    Returns:
        int: The line number of that index.
        int: The column number of that index.
    """
    line_num = string.count('\n', 0, index) + 1
    prev_newline = string.rfind('\n', 0, index)
    if prev_newline == -1:
        column = index
    else:
        column = index - prev_newline
    return line_num, column


TraceItem = namedtuple('TraceItem', 'depth, term, position')


class ASTNode:
    """Abstract Syntax Tree (AST) node."""

    def __init__(self, term, children, string, start_pos, end_pos):
        # type: (str, List[ASTNode], str, int, int) -> None
        """Initialize the ASTNode.

        The string, start_pos, and end_pos arguments matches the arguments to
        range(), ie. the substring
            string[start_pos:end_pos]

        Parameters:
            term (str): The term this node matches.
            children (list[ASTNode]): Children nodes of this term in the grammar.
            string (str): The complete string being parsed.
            start_pos (int): Index of the first character matched by this node.
            end_pos (int): Index of the last character matched by this node.
        """
        self.term = term
        self.children = children
        self.string = string
        self.start_pos = start_pos
        self.end_pos = end_pos

    @property
    def match(self):
        # type: () -> str
        """Return the substring matched by this node."""
        return self.string[self.start_pos:self.end_pos]

    @property
    def line_num(self):
        # type: () -> int
        """Return the starting line number of the substring matched by this node."""
        return index_to_line_col(self.string, self.start_pos)[0]

    @property
    def column(self):
        # type: () -> int
        """Return the starting column of the substring matched by this node."""
        return index_to_line_col(self.string, self.start_pos)[1]

    def first_descendant(self, path='*'):
        # type: (str) -> ASTNode
        """Get the first ASTNode descendant that matches the path.

        See the docstring for descendants() for a description of the path
        argument.

        Parameters:
            path (str): The path to the desired descendant.

        Returns:
            ASTNode: The node representing the descendant.
        """
        result = self
        for term in path.split('/'):
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
                    return None
        return result

    def descendants(self, path='*'):
        # type: (str) -> Tuple[ASTNode, ...]
        """Get all ASTNode descendants that match the path.

        The path describes the term of each descendant separated by a '/'.
        Where the term is irrelevant, it can either be represented by '*' or
        left empty. For example, given the following grammar:

            Expression = Operand ( Operator Operand )*;
            Operand = ParenExpression
                    | Number;
            ParenExpression = "(" Expression ")";
            Operator = "+"
                     | "-"
                     | "*"
                     | "/";
            Number = ( digit )+;

        And an ASTNode representing the following Expression:

            (1+(2*3))-5

        The call `.descendants('Operand/ParenExpression/Expression/Operand')`
        would return the Operand ASTNodes with for '1' and '(2*3)'. Both of the
        paths 'Operand' and 'Operand/*' would give the ASTNodes for '(1+(2*3))'
        and '5', but while the first path would return two Operand ASTNodes,
        the second path would return a ParenExpression and a Number ASTNode.

        Parameters:
            path (str): The path to the desired descendant.

        Returns:
            ASTNode: All descendant ASTNodes that match the path.
        """
        cur_gen = [self,] # type: List[ASTNode]
        for term in path.split('/'):
            next_gen = []
            for adult in cur_gen:
                next_gen.extend(adult.children)
            if term == '*':
                cur_gen = next_gen
            else:
                cur_gen = [child for child in next_gen if child.term == term]
        return tuple(cur_gen)

    def pretty_print(self, indent_level=0):
        # type: (int) -> None
        """Print the ASTNode using indentation to denote ancestry."""
        print('{}{}: {}'.format(indent_level * 4 * ' ', self.term, one_line_format(self.match)))
        for child in self.children:
            child.pretty_print(indent_level + 1)


class PEGParser:
    """Parser for Parsing Expression Grammars (PEGs).

    A fairly standard packrat parser. The core definitions are stored as
    constants, while the custom definitions are provided to the constructor.
    """

    CORE_DEFS = {
        'EMPTY': r'',
        'BLANK': r'[ \t]',
        'DIGIT': r'[0-9]',
        'UPPER': r'[A-Z]',
        'LOWER': r'[a-z]',
        'ALPHA': r'[A-Za-z]',
        'ALNUM': r'[0-9A-Za-z]',
        'PUNCT': r"[-!\"#$%&'()*+,./:;<=>?@[\\\]^_`{|}~]",
        'PRINT': r'[ -~]',
        'UNICODE': r'[^\x00-\x7F]',
        'NEWLINE': r'\n',
        'TAB': r'\t',
    }

    def __init__(self, syntax, debug=False):
        # type: (PEGRules, bool) -> None
        """Initialize the Parser.

        Parameters:
            syntax (dict[str]): Dictionary of term definitions. This is usually
                produced by an PEGWalker instance.
            debug (bool): Whether to print parsing information.
                Defaults to False.
        """
        self.custom_defs = syntax
        self.debug = debug
        self.cache = {} # type: Dict[Tuple[str, int], ASTNode]
        self.depth = 0
        self.trace = [] # type: List[TraceItem]
        self.max_trace_index = 0

    def parse_file(self, filepath, term):
        # type: (Path, str) -> ASTNode
        """Parse the contents of a file as a given term.

        Parameters:
            filepath (str): The path to the file.
            term (str): The term to parse the string as.

        Returns:
            ASTNode: The root node of the AST.
        """
        with open(filepath) as fd:
            return self.parse(fd.read(), term)

    def parse(self, string, term):
        # type: (str, str) -> ASTNode
        """Parse a string as a given term.

        Parameters:
            string (str): The string to parse.
            term (str): The term to parse the string as.

        Returns:
            ASTNode: The root node of the AST.
        """
        ast = self.parse_partial(string, term)
        if not ast:
            self._fail_parse(string, 0)
        elif ast.end_pos != len(string):
            self._fail_parse(string, ast.end_pos)
        return ast

    def parse_partial(self, string, term):
        # type: (str, str) -> Optional[ASTNode]
        """Parse a string as a given term.

        Parameters:
            string (str): The string to parse.
            term (str): The term to parse the string as.

        Returns:
            ASTNode: The root node of the AST.
            int: The number of characters parsed
        """
        self.cache = {}
        self.depth = 0
        self.trace = []
        self.max_trace_index = 0
        ast = self._match(string, term, 0)
        if ast:
            return ast
        else:
            return None

    def _add_trace(self, term, position):
        # type: (str, int) -> None
        """Log the parse for error messages.

        Parameters:
            term (str): The term to parse as.
            position (int): The position from which to parse.
        """
        trace_item = TraceItem(self.depth, term, position)
        self.trace.append(trace_item)
        if len(self.trace) == 1:
            return
        max_position = self.trace[self.max_trace_index].position
        index = len(self.trace) - 2
        while (
                index >= 0
                and trace_item.depth <= self.trace[index].depth
                and (
                    trace_item.position == self.trace[index].position
                    or self.trace[index].position < max_position
                )
        ):
            del self.trace[index]
            index -= 1
        if trace_item.position >= max_position:
            self.max_trace_index = len(self.trace) - 1

    def _fail_parse(self, string, parsed):
        # type: (str, int) -> None
        """Fail a parse by raising SyntaxError with a trace.

        Parameters:
            string (str): The string to parse.
            parsed (int): The number of characters successfully parsed.

        Raises:
            SyntaxError: If the term is not defined, or if no parse was found
                before the end of the string.
        """
        trace = []
        for _, term, position in self.trace[:self.max_trace_index]:
            line, col = index_to_line_col(string, position)
            trace.append('\n'.join([
                'Failed to match {} at line {} column {} (position {})'.format(
                    term, line, col, position
                ),
                '  ' + string.splitlines()[line - 1].replace('\t', ' '),
                '  ' + (col - 1) * '-' + '^',
            ]))
        raise SyntaxError(
            'only parsed {} of {} characters:\n'.format(parsed, len(string))
            + indent('\n'.join(trace), '  ')
        )

    def _match(self, string, term, position=0):
        # type: (str, Union[PEGExpression, str], int) -> Optional[ASTNode]
        """Dispatch the parsing to specialized functions.

        Parameters:
            string (str): The string to parse.
            term (str): The term to parse the string as.
            position (int): The position which with to start the parse.
                Defaults to 0.

        Returns:
            ASTNode: The root node of the AST.
            int: The number of characters successfully parsed.

        Raises:
            NameError: If the terminal is unknown.
        """
        if isinstance(term, tuple) and hasattr(self, '_match_{}'.format(term[0].lower())):
            return getattr(self, '_match_{}'.format(term[0].lower()))(string, term, position)
        elif isinstance(term, str):
            ast = self._get_cached(term, position)
            if ast:
                return ast
            elif term in self.custom_defs:
                return self._match_custom(string, term, position)
            elif term in PEGParser.CORE_DEFS:
                return self._match_core(string, term, position)
            elif re.match(r"^'[^']*'$", term) or re.match(r'^"[^"]*"$', term):
                return self._match_literal(string, term, position)
            else:
                raise NameError('Unknown terminal {}'.format(term))
        raise NameError('Unknown meta-terminal {}'.format(term))

    def _match_choice(self, string, terms, position):
        # type: (str, PEGExpression, int) -> Optional[ASTNode]
        """Parse the disjunction of/any of multiple terms.

        Parameters:
            string (str): The string to parse.
            terms (list[str]): The terms to attempt to parse as.
            position (int): The position which with to start the parse.

        Returns:
            ASTNode: The root node of this abstract syntax sub-tree.
            int: The index of the last character parsed.
        """
        for term in terms[1:]:
            ast = self._match(string, term, position)
            if ast:
                return ast
        return None

    def _match_sequence(self, string, terms, position):
        # type: (str, PEGExpression, int) -> Optional[ASTNode]
        """Parse the concatenation of multiple terms.

        Parameters:
            string (str): The string to parse.
            terms (list[str]): The terms that are concatenated.
            position (int): The position which with to start the parse.

        Returns:
            ASTNode: The root node of this abstract syntax sub-tree.
            int: The index of the last character parsed.
        """
        children = []
        pos = position
        for term in terms[1:]:
            child_ast = self._match(string, term, pos)
            if child_ast:
                if isinstance(term, tuple):
                    children.extend(child_ast.children)
                else:
                    children.append(child_ast)
                pos = child_ast.end_pos
            else:
                return None
        return ASTNode('SEQUENCE', children, string, position, pos)

    def _match_zero_or_more(self, string, terms, position):
        # type: (str, PEGExpression, int) -> Optional[ASTNode]
        """Parse zero-or-more of a term (the * operator).

        Parameters:
            string (str): The string to parse.
            terms (list[str]): The terms to repeat, as a syntax definition.
            position (int): The position which with to start the parse.

        Returns:
            ASTNode: The root node of this abstract syntax sub-tree.
            int: The index of the last character parsed.
        """
        terms = terms[1]
        last_pos = position
        children = []
        ast = self._match(string, terms, last_pos)
        while ast:
            last_pos = ast.end_pos
            children.extend(ast.children)
            ast = self._match(string, terms, last_pos)
        return ASTNode('ZERO_OR_MORE', children, string, position, last_pos)

    def _match_zero_or_one(self, string, terms, position):
        # type: (str, PEGExpression, int) -> Optional[ASTNode]
        """Parse zero-or-one of a term (the ? operator).

        Parameters:
            string (str): The string to parse.
            terms (list[str]): The terms to repeat, as a syntax definition.
            position (int): The position which with to start the parse.

        Returns:
            ASTNode: The root node of this abstract syntax sub-tree.
            int: The index of the last character parsed.
        """
        terms = terms[1]
        ast = self._match(string, terms, position)
        if ast:
            return ast
        return self._match(string, 'EMPTY', position)

    def _match_one_or_more(self, string, terms, position):
        # type: (str, PEGExpression, int) -> Optional[ASTNode]
        """Parse one-or-more of a term (the + operator).

        Parameters:
            string (str): The string to parse.
            terms (list[str]): The terms to repeat, as a syntax definition.
            position (int): The position which with to start the parse.

        Returns:
            ASTNode: The root node of this abstract syntax sub-tree.
            int: The index of the last character parsed.
        """
        terms = terms[1]
        ast = self._match(string, terms, position)
        if not ast:
            return None
        last_pos = ast.end_pos
        children = ast.children
        ast = self._match(string, terms, last_pos)
        while ast:
            last_pos = ast.end_pos
            children.extend(ast.children)
            ast = self._match(string, terms, last_pos)
        return ASTNode('ONE_OR_MORE', children, string, position, last_pos)

    def _match_and(self, string, terms, position):
        # type: (str, PEGExpression, int) -> Optional[ASTNode]
        """Parse the negation of a term.

        Parameters:
            string (str): The string to parse.
            terms (list[str]): The first item (index 0) is the term to match;
                all subsequent items are terms to *not* match. FIXME
            position (int): The position which with to start the parse.

        Returns:
            ASTNode: The root node of this abstract syntax sub-tree.
            int: The index of the last character parsed.
        """
        ast = self._match(string, terms[1], position)
        if not ast:
            return None
        return self._match(string, 'EMPTY', position)

    def _match_not(self, string, terms, position):
        # type: (str, PEGExpression, int) -> Optional[ASTNode]
        """Parse the negation of a term.

        Parameters:
            string (str): The string to parse.
            terms (list[str]): The first item (index 0) is the term to match;
                all subsequent items are terms to *not* match. FIXME
            position (int): The position which with to start the parse.

        Returns:
            ASTNode: The root node of this abstract syntax sub-tree.
            int: The index of the last character parsed.
        """
        ast = self._match(string, terms[1], position)
        if ast:
            return None
        return self._match(string, 'EMPTY', position)

    def _match_custom(self, string, term, position):
        # type: (str, PEGAtom, int) -> Optional[ASTNode]
        """Dispatch a parse to the custom syntax definition.

        Parameters:
            string (str): The string to parse.
            term (str): The term to parse the string as.
            position (int): The position which with to start the parse.

        Returns:
            ASTNode: The root node of this abstract syntax sub-tree.
            int: The index of the last character parsed.
        """
        expression = self.custom_defs[term]
        self._debug_print('parsing {} at position {} >>>{}'.format(
            term, position, one_line_format(string[position:position+32])
        ))
        self._add_trace(term, position)
        self.depth += 1
        ast = self._match(string, expression, position)
        self.depth -= 1
        if ast:
            ast.term = term
            return self._cache_and_return(term, position, ast)
        else:
            self._debug_print('failed to match {} at position {}'.format(term, position))
            return self._cache_and_return(term, position, None)

    def _match_core(self, string, term, position):
        # type: (str, PEGAtom, int) -> Optional[ASTNode]
        """Parse a core syntax definition.

        Parameters:
            string (str): The string to parse.
            term (str): The term to parse the string as.
            position (int): The position which with to start the parse.

        Returns:
            ASTNode: The root node of this abstract syntax sub-tree.
            int: The index of the last character parsed.
        """
        match = re.match(PEGParser.CORE_DEFS[term], string[position:])
        if match:
            ast = ASTNode(term, [], string, position, position + len(match.group(0)))
            return self._cache_and_return(term, position, ast)
        return self._cache_and_return(term, position, None)

    def _match_literal(self, string, term, position):
        # type: (str, PEGAtom, int) -> Optional[ASTNode]
        """Parse a literal.

        Parameters:
            string (str): The string to parse.
            term (str): The literal to parse the string as.
            position (int): The position which with to start the parse.

        Returns:
            ASTNode: The root node of this abstract syntax sub-tree.
            int: The index of the last character parsed.
        """
        if string[position:].startswith(term[1:-1]):
            ast = ASTNode(term, [], string, position, position + len(term[1:-1]))
            return self._cache_and_return(term, position, ast)
        return self._cache_and_return(term, position, None)

    def _cache_and_return(self, term, position, ast):
        # type: (str, int, Optional[ASTNode]) -> Optional[ASTNode]
        """Cache a successful parse and return the result.

        Parameters:
            term (str): The literal to parse the string as.
            position (int): The position which with to start the parse.
            ast (ASTNode): The root of the parsed abstract syntax sub-tree.

        Returns:
            None: The absence of an ASTNode.
            int: The index of the last character parsed.
        """
        self.cache[(term, position)] = ast
        return ast

    def _get_cached(self, term, position):
        # type: (str, int) -> Optional[ASTNode]
        """Retrieve a parse from cache, if it exists.

        Parameters:
            term (str): The literal to parse the string as.
            position (int): The position which with to start the parse.

        Returns:
            None: The absence of an ASTNode.
            int: The index of the last character parsed.
        """
        if (term, position) in self.cache:
            if term in self.custom_defs:
                self._debug_print('matched {} at position {}'.format(term, position))
            ast = self.cache[(term, position)]
            return ast
        return None

    def _debug_print(self, message):
        # type: (str) -> None
        """Print debugging information with indentation.

        Parameters:
            message (str): The message to print.
        """
        if self.debug:
            print(self.depth * '    ' + message)


class ASTWalker:
    """A traversal of an AST.

    This is a base class for any processing that requires the bottom-up
    building of structures from an AST. Subclass functions with the name
    `parse_Term` - where Term is the name of the node in the grammar - are
    called when a that term is encountered on the way back up the tree, ie.
    in a post-order traversal. Each such function takes two arguments:

    * ast (ASTNode): The AST rooted at that node.
    * results (list[any]): The results from the descendants of the node.
    """

    PARSE_FUNCTION_PREFIX = '_parse_'

    class EmptySentinel:
        """Sentinel to indicate that no processing was done."""

    def __init__(self, parser, root_term):
        # type: (PEGParser, str) -> None
        """Initialize the traversal.

        Parameters:
            parser (PEGParser): The parser to use.
            root_term (str): The term to start parsing on.
        """
        self.parser = parser
        self.root_term = root_term
        self._dispatch_terms = set(
            term[len(self.PARSE_FUNCTION_PREFIX):] for term in dir(self)
            if term.startswith(self.PARSE_FUNCTION_PREFIX)
        )
        self._terms_to_expand = set()
        noskips = list(self._dispatch_terms)
        while noskips:
            noskip = noskips.pop()
            self._terms_to_expand.add(noskip)
            for term, definition in self.parser.custom_defs.items():
                if term in self._terms_to_expand:
                    continue
                if ASTWalker.term_in_definition(noskip, definition):
                    noskips.append(term)
                    self._terms_to_expand.add(term)

    def _postorder_traversal(self, ast, depth=0):
        # type: (ASTNode, int) -> Tuple[Any, bool]
        """Traverses the AST in post-order.

        Parameters:
            ast (ASTNode): The AST to traverse.
            depth (int): The current depth, for printing purposes.
                Defaults to 0.

        Returns:
            any: Whatever the parse_* functions return, or an EmptySentinel.
            bool: Whether a parse occurred.
        """
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
        function = self.PARSE_FUNCTION_PREFIX + ast.term
        if hasattr(self, function):
            return getattr(self, function)(ast, tuple(results)), True
        elif results:
            return tuple(results), False
        else:
            return ASTWalker.EmptySentinel(), False

    def parse_file(self, filepath, term=None):
        # type: (Path, Optional[str]) -> Any
        """Parse a file with the traversal.

        Parameters:
            filepath (str): The path to the file.
            term (str): The term to start parsing on. Defaults to the term from
                the constructor.

        Returns:
            any: Whatever the parse_* functions return.
        """
        with open(filepath) as fd:
            return self.parse(fd.read(), term)

    def parse(self, text, term=None):
        # type: (str, Optional[str]) -> Any
        """Parse a complete string as the term.

        Parameters:
            text (str): The text to parse.
            term (str): The term to start parsing on. Defaults to the term from
                the constructor.

        Returns:
            any: Whatever the parse_* functions return.
        """
        if term is None:
            term = self.root_term
        ast = self.parser.parse(text, term)
        return self.parse_ast(ast)

    def parse_partial(self, text, term=None):
        # type: (str, Optional[str]) -> Tuple[Any, int]
        """Parse as much of a string as possible as the term.

        Parameters:
            text (str): The text to parse.
            term (str): The term to start parsing on. Defaults to the term from
                the constructor.

        Returns:
            any: Whatever the parse_* functions return.
            int: The number of characters parsed
        """
        if term is None:
            term = self.root_term
        ast = self.parser.parse_partial(text, term)
        return self.parse_ast(ast), ast.end_pos

    def parse_ast(self, ast):
        # type: (ASTNode) -> Any
        """Parse an AST.

        Parameters:
            ast (ASTNode): The AST to parse.

        Returns:
            any: Whatever the parse_* functions return.
        """
        if ast is None:
            return None
        result = self._postorder_traversal(ast)[0]
        if isinstance(result, ASTWalker.EmptySentinel) or ast.term in self._dispatch_terms:
            return result
        else:
            return result[0]

    @staticmethod
    def term_in_definition(term, definition):
        # type: (str, Union[PEGExpression, str]) -> bool
        """Determine if a definition could ever expand to include a term.

        Parameters:
            term (str): The term to find.
            definition (dict[str]): Dictionary of term definitions.

        Returns:
            bool: Whether the term could be in the definition.
        """
        return term == definition or any(
            (
                term == element
                or (
                    isinstance(element, tuple)
                    and ASTWalker.term_in_definition(term, element)
                )
            )
            for element in definition
        )


class PEGWalker(ASTWalker):
    # pylint: disable=invalid-name,no-self-use,unused-argument
    """A traversal of the PEG grammar to build up term definitions."""

    def __init__(self):
        # type: () -> None
        """Initialize the traversal."""
        super().__init__(PEGParser(PEG_DEFS), 'syntax')

    def _parse_syntax(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> PEGRules
        """Parse a Syntax node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            dict[str]: Dictionary of term definitions.
        """
        return {result[0]: result[1] for result in results}

    def _parse_definition(self, ast, results):
        # type: (ASTNode, Tuple[str, PEGExpression]) -> Tuple[str, PEGExpression]
        """Parse a Definition node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            tuple[str, tuple[str]]: A term definition.
        """
        return results

    def _parse_choice(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> PEGExpression
        """Parse a Choice node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            tuple[str]: A choice definition.
        """
        return ('CHOICE', *results)

    def _parse_sequence(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> PEGExpression
        """Parse a Sequence node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            tuple[str]: A choice definition.
        """
        return ('SEQUENCE', *results)

    def _parse_zero_or_more(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> PEGExpression
        """Parse a zero-or-more predicate node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            tuple[str]: A repetition definition.
        """
        return ('ZERO_OR_MORE', *results)

    def _parse_zero_or_one(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> PEGExpression
        """Parse a zero-or-one predicate node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            tuple[str]: A repetition definition.
        """
        return ('ZERO_OR_ONE', *results)

    def _parse_one_or_more(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> PEGExpression
        """Parse a one-or-more predicate node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            tuple[str]: A repetition definition.
        """
        return ('ONE_OR_MORE', *results)

    def _parse_and_predicate(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> PEGExpression
        """Parse an And predicate node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            tuple[str]: A predicate definition.
        """
        return ('AND', *results)

    def _parse_not_predicate(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> PEGExpression
        """Parse an Not predicate node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            tuple[str]: A predicate definition.
        """
        return ('NOT', *results)

    def _parse_identifier(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> str
        """Parse an Identifier node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            str: A custom term.
        """
        return ast.match

    def _parse_builtin(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> str
        """Parse a Reserved node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            str: A builtin (core term).
        """
        return ast.match

    def _parse_literal(self, ast, results):
        # type: (ASTNode, Tuple[Any, ...]) -> str
        """Parse a Literal node.

        Parameters:
            ast (ASTNode): The AST term to head the tuple.
            results (list[any]): The results from descendants.

        Returns:
            str: The literal.
        """
        return ast.match


def main():
    # type: () -> None
    """Parse a grammar."""
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        '-e', dest='expression',
        help='starting expression; if omitted, first defined term is used',
    )
    arg_parser.add_argument('-g', dest='grammar', default=PEG_GRAMMAR, help='PEG grammar file')
    arg_parser.add_argument(
        '-v', dest='verbose', action='store_true',
        help='show what the parser is doing',
    )
    arg_parser.add_argument('file', default='-', nargs='?', help='text file to be parsed')
    args = arg_parser.parse_args()
    if args.expression:
        term = args.expression
    else:
        with open(args.grammar, 'r') as fd:
            grammar_ast = PEGParser(PEG_DEFS).parse(fd.read(), 'syntax')
            term = grammar_ast.first_descendant('definition/identifier').match
    parser = create_parser_from_file(args.grammar, debug=args.verbose)
    parser.parse(''.join(fileinput(files=args.file)), term).pretty_print()


if __name__ == '__main__':
    main()
