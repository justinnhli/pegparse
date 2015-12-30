#!/usr/bin/env python3

import re

# TODO
# have ASTNodes dynamically generate the match; saves memory from storing a string multiple times
#   this can be done by storing the full string via variable capture in a function which takes substring start and end positions

EBNF_DEFS = {
    'Syntax': ('AND', ('ONE-OR-MORE', ('AND', ('ZERO-OR-MORE', ('AND', 'EmptyLine')), 'Definition', 'newline')), ('ZERO-OR-MORE', ('AND', 'EmptyLine'))),
    'Definition': ('AND', 'Identifier', 'Whitespace', '"= "', 'Expression', '";"'),
    'Expression': ('OR', 'Disjunct', 'Except', 'Conjunct'),
    'Conjunct': ('AND', 'Item', ('ZERO-OR-MORE', ('AND', '" "', 'Item'))),
    'Disjunct': ('AND', 'Atom', ('ONE-OR-MORE', ('AND', 'newline', 'Whitespace', '"| "', 'Atom'))),
    'Except': ('AND', 'Atom', ('ONE-OR-MORE', ('AND', 'newline', 'Whitespace', '"- "', 'Atom'))),
    'Item': ('OR', 'ZeroOrMore', 'ZeroOrOne', 'OneOrMore', 'Atom'),
    'ZeroOrMore': ('AND', '"( "', 'Conjunct', '" )*"'),
    'ZeroOrOne': ('AND', '"( "', 'Conjunct', '" )?"'),
    'OneOrMore': ('AND', '"( "', 'Conjunct', '" )+"'),
    'Atom': ('OR', 'Identifier', 'Reserved', 'Literal'),
    'Identifier': ('AND', ('ONE-OR-MORE', ('AND', 'upper', ('ZERO-OR-MORE', ('AND', 'lower'))))),
    'Reserved': ('AND', ('ONE-OR-MORE', ('AND', 'lower'))),
    'Literal': ('OR', 'DString', 'SString'),
    'DString': ('AND', '\'"\'', ('ZERO-OR-MORE', ('AND', 'NoDQuote')), '\'"\''),
    'SString': ('AND', '"\'"', ('ZERO-OR-MORE', ('AND', 'NoSQuote')), '"\'"'),
    'NoDQuote': ('NOT', 'print', '\'"\''),
    'NoSQuote': ('NOT', 'print', '"\'"'),
    'Whitespace': ('AND', ('ONE-OR-MORE', ('AND', 'blank'))),
    'EmptyLine': ('AND', ('ZERO-OR-ONE', ('AND', '"#"', ('ZERO-OR-MORE', ('AND', 'print')))), 'newline'),
}

DESIGNATOR_MAPPING = {
    "Conjunct"   : "AND",
    "Disjunct"   : "OR",
    "Except"     : "NOT",
    "ZeroOrMore" : "ZERO-OR-MORE",
    "ZeroOrOne"  : "ZERO-OR-ONE",
    "OneOrMore"  : "ONE-OR-MORE",
}

def create_parser_from_file(file):
    with open(file) as fd:
        ebnf = fd.read()
    return create_parser(ebnf)

def create_parser(bnf):
    return PEGParser(EBNFWalker().parse(bnf))

class ASTNode:
    def __init__(self, term=None, children=None, match=None):
        self.term = ("" if term is None else term)
        self.children = children
        self.match = match
    def __bool__(self):
        return self.term != ""
    def first_descendant(self, descentry=None):
        descentry = ("*" if descentry is None else descentry).split("/")
        result = self
        for term in descentry:
            if term == "*":
                result = result.children[0]
            else:
                children = [child for child in result.children if child.term == term]
                if children:
                    result = children[0]
                else:
                    return []
        return result
    def descendants(self, descentry=None):
        descentry = ("*" if descentry is None else descentry).split("/")
        cur_gen = [self]
        for term in descentry:
            next_gen = []
            for adult in cur_gen:
                next_gen.extend(adult.children)
            if term == "*":
                cur_gen = next_gen
            else:
                cur_gen = [child for child in next_gen if child.term == term]
        return cur_gen
    def pretty_print(self, indent=0):
        print("{}{}: {}".format("    " * indent, self.term, re.sub(r"\n", r"\\n", str(self.match))))
        for child in self.children:
            child.pretty_print(indent + 1)

class PEGParser:
    CORE_DEFS = {
        "empty"   : r"",
        "blank"   : r"[ \t]",
        "digit"   : r"[0-9]",
        "upper"   : r"[A-Z]",
        "lower"   : r"[a-z]",
        "alpha"   : r"[A-Za-z]",
        "alnum"   : r"[0-9A-Za-z]",
        "punct"   : r"[-!\"#$%&'()*+,./:;<=>?@[\\\]^_`{|}~]",
        "print"   : r"[ -~]",
        "unicode" : r"[^\x00-\x7F]",
        "newline" : r"\n",
        "tab"     : r"\t",
    }
    def __init__(self, syntax):
        self.custom_defs = syntax
        self.debug = False
        self.cache = {}
        self.indent = 0
        self.syntax_map = {
            "AND"          : self.match_and,
            "OR"           : self.match_or,
            "NOT"          : self.match_not,
            "ZERO-OR-ONE"  : self.match_zero_or_one,
            "ZERO-OR-MORE" : self.match_zero_or_more,
            "ONE-OR-MORE"  : self.match_one_or_more,
        }
    def parse(self, string, term):
        self.cache = {}
        self.indent = 0
        ast, parsed = self.dispatch(string, term, 0)
        if ast and parsed == len(string):
            return ast
        else:
            raise SyntaxError('only parsed {} of {} characters in:\n{}\n'.format(parsed, len(string), string))
    def partial_parse(self, string, term):
        self.cache = {}
        self.indent = 0
        return self.dispatch(string, term, 0)
    def dispatch(self, string, term, position=0):
        if not isinstance(term, tuple):
            self.debug_print("parse called at position {} with {} >>>{}".format(position, term, re.sub(r"\n", r"\\n", string[position:position+32])))
        if isinstance(term, tuple) and term[0] in self.syntax_map:
            return self.syntax_map[term[0]](string, term, position)
        elif isinstance(term, str):
            ast, pos = self.get_cached(term, position)
            if ast:
                return ast, pos
            elif term in self.custom_defs:
                return self.match_custom(string, term, position)
            elif term in PEGParser.CORE_DEFS:
                return self.match_core(string, term, position)
            elif re.match(r"^'[^']*'$", term) or re.match(r'^"[^"]*"$', term):
                return self.match_literal(string, term, position)
        self.debug_print("unknown non-terminal: " + term)
        return self.fail(term, position)
    def match_zero_or_more(self, string, terms, position):
        terms = terms[1]
        children = []
        last_pos = position
        ast, pos = self.dispatch(string, terms, position)
        while ast:
            children.extend(ast.children)
            last_pos = pos
            ast, pos = self.dispatch(string, terms, pos)
        return ASTNode("ZERO-OR-MORE", children, string[position:last_pos]), last_pos
    def match_zero_or_one(self, string, terms, position):
        terms = terms[1]
        ast, pos = self.dispatch(string, terms, position)
        if ast:
            return ast, pos
        return self.dispatch(string, "empty", position)
    def match_one_or_more(self, string, terms, position):
        terms = terms[1]
        ast, pos = self.dispatch(string, terms, position)
        if not ast:
            return ast, pos
        else:
            children = ast.children
            last_pos = pos
            ast, pos = self.dispatch(string, terms, pos)
            while ast:
                children.extend(ast.children)
                last_pos = pos
                ast, pos = self.dispatch(string, terms, pos)
            return ASTNode("ONE-OR-MORE", children, string[position:last_pos]), last_pos
    def match_and(self, string, terms, position):
        children = []
        pos = position
        for term in terms[1:]:
            child_ast, child_pos = self.dispatch(string, term, pos)
            if child_ast:
                if isinstance(term, tuple) and (term[0] in ["ZERO-OR-ONE", "ZERO-OR-MORE", "ONE-OR-MORE"]):
                    children.extend(child_ast.children)
                else:
                    children.append(child_ast)
                pos = child_pos
            else:
                return self.fail(term, child_pos)
        return ASTNode("AND", children, string[position:pos]), pos
    def match_or(self, string, terms, position):
        for term in terms[1:]:
            ast, pos = self.dispatch(string, term, position)
            if ast:
                return ast, pos
        return self.fail(terms[-1], position)
    def match_not(self, string, terms, position):
        ast, pos = self.dispatch(string, terms[1], position)
        if ast:
            for term in terms[2:]:
                self.debug_print(term)
                nast = self.dispatch(string, term, position)[0]
                if nast and ast.match == nast.match:
                    return self.fail(term, position)
            return ast, pos
        return self.fail(terms[1], position)
    def match_custom(self, string, term, position):
        expression = self.custom_defs[term]
        self.indent += 1
        ast = self.dispatch(string, expression, position)[0]
        self.indent -= 1
        if ast:
            if isinstance(expression, tuple) and expression[0] == "OR":
                ast = ASTNode(term, [ast], ast.match)
            else:
                ast.term = term
            return self.cache_and_return(term, position, ast)
        else:
            return self.fail(term, position)
    def match_core(self, string, term, position):
        match = re.match(PEGParser.CORE_DEFS[term], string[position:])
        if match:
            ast = ASTNode(term, [], match.group(0))
            return self.cache_and_return(term, position, ast)
        return self.fail(term, position)
    def match_literal(self, string, term, position):
        if string[position:].find(term[1:-1]) == 0:
            ast = ASTNode(term, [], term[1:-1])
            return self.cache_and_return(term, position, ast)
        return self.fail(term, position)
    def fail(self, term, position):
        self.debug_print("failed to match " + str(term) + " at position " + str(position))
        return ASTNode(), position
    def cache_and_return(self, term, position, ast):
        self.cache.setdefault(term, {})
        self.cache[term][position] = ast
        return self.get_cached(term, position)
    def get_cached(self, term, position):
        if (term in self.cache) and (position in self.cache[term]):
            self.debug_print("matched " + term + " at position " + str(position))
            ast = self.cache[term][position]
            return ast, position + len(ast.match)
        return ASTNode(), position
    def debug_print(self, obj):
        if self.debug:
            print("    " * self.indent + str(obj))

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
            result, parsed = self._postorder_traversal(child, depth=depth+1)
            if not isinstance(result, ASTWalker.EmptySentinel):
                if parsed:
                    results.append(result)
                else:
                    results.extend(result)
        function = 'parse_' + ast.term
        if hasattr(self, function):
            return getattr(self, function)(ast, results), True
        elif results:
            return results, False
        else:
            return ASTWalker.EmptySentinel(), False
    def parse(self, text):
        ast = self.parser.parse(text, self.term)
        return self._postorder_traversal(ast)[0]
    @staticmethod
    def term_in_definition(term, definition):
        return any((term == element or (isinstance(element, tuple) and ASTWalker.term_in_definition(term, element))) for element in definition)

class EBNFWalker(ASTWalker):
    def __init__(self):
        super().__init__(PEGParser(EBNF_DEFS), 'Syntax')
    def flatten(self, ast, results):
        return tuple((DESIGNATOR_MAPPING[ast.term], *results))
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
    def parse_ZeroOrMore(self, ast, results):
        return self.flatten(ast, results)
    def parse_ZeroOrOne(self, ast, results):
        return self.flatten(ast, results)
    def parse_OneOrMore(self, ast, results):
        return self.flatten(ast, results)
    def parse_Reserved(self, ast, results):
        return ast.match
    def parse_Identifier(self, ast, results):
        return ast.match
    def parse_Literal(self, ast, results):
        return ast.match

def test():
    import unittest
    from os.path import dirname, join as join_path
    with open(join_path(dirname(__file__), 'ebnf.ebnf')) as fd:
        text = fd.read()
    assert EBNFWalker().parse(text) == EBNF_DEFS

def main():
    from argparse import ArgumentParser
    from fileinput import input as fileinput
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-e", dest="expression", help="starting expression; if omitted, first defined term is used")
    arg_parser.add_argument("-g", dest="grammar", help="EBNF grammar file")
    arg_parser.add_argument("-v", dest="verbose", default=False, action="store_true", help="show what the parser is doing")
    arg_parser.add_argument("file", default="-", nargs="?", help="text file to be parsed")
    args = arg_parser.parse_args()
    if args.grammar:
        grammar = ""
        with open(args.grammar, "r") as fd:
            grammar = fd.read()
        parser = create_parser(grammar)
        if parser is None:
            print("error: grammar file cannot be parsed")
            exit(1)
        if args.expression:
            if args.expression not in parser.custom_defs:
                print("error: specified expression not defined")
                exit(1)
            term = args.expression
        else:
            term = PEGParser(EBNF_DEFS).parse(grammar, "Syntax").first_descendant("Definition/Identifier").match
    else:
        parser = PEGParser(EBNF_DEFS)
        term = "Syntax"
    parser.debug = args.verbose
    contents = "".join(fileinput(files=args.file))
    ast, chars_parsed = parser.partial_parse(contents, term)
    length = len(contents)
    if not ast or chars_parsed != length:
        print("failed: only parsed {} of {} characters\n".format(chars_parsed, length))
        exit(1)
    else:
        ast.pretty_print()

if __name__ == "__main__":
    main()
