#!/usr/bin/env python3

import sys
from textwrap import dedent
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pegparse import PEG_GRAMMAR, PEG_DEFS, ASTNode, PEGWalker
from pegparse import create_parser


def assert_equal(expected, actual, message_stem):
    message = '\n'.join([
        message_stem,
        'expected: ' + str(expected),
        'actual: ' + str(actual),
    ])
    assert actual == expected, message


def test_ast_structure():

    def _test_descendants(string, term, path, expected):
        ast = parser.parse(string, term)
        actual = [node.match for node in ast.descendants(path)]
        assert_equal(
            expected,
            actual,
            f'expected {path} descendants of "{string}"to be {expected} but got {actual}',
        )

    parser = create_parser(dedent('''
        expression = operand ( operator operand )*;
        operand    = paren
                   | number;
        paren      = "(" expression ")";
        number     = ( DIGIT )+;
        operator   = "+"
                   | "-"
                   | "*"
                   | "/";
    '''))
    _test_descendants('1', 'operand', 'number', ['1'])
    _test_descendants('(1)', 'operand', 'paren/expression', ['1'])
    _test_descendants('(1+(2*3))-5', 'expression', 'operand', ['(1+(2*3))', '5'])
    _test_descendants('(1+(2*3))-5', 'expression', 'operand/*', ['(1+(2*3))', '5'])
    _test_descendants('(1+(2*3))-5', 'expression', 'operand/paren/expression/operand', ['1', '(2*3)'])

    parser = create_parser(dedent('''
        a = b;
        b = ( c )*;
        c = "!";
    '''))
    _test_descendants('!!!', 'a', 'b/c', ['!', '!', '!'])

    parser = create_parser(dedent('''
        a = ( b c | ( d | e ) ) f ;
        b = "b";
        c = "c";
        d = "d";
        e = "e";
        f = "f";
    '''))
    _test_descendants('df', 'a', '*', ['d', 'f'])
    _test_descendants('bcf', 'a', '*', ['b', 'c', 'f'])
    _test_descendants('ef', 'a', '*', ['e', 'f'])



def test_peg_walker():
    with open(PEG_GRAMMAR) as fd:
        text = fd.read()
    ast = PEGWalker().parse(text)
    assert isinstance(ast, dict), 'PEGWalker failed to return a dict'


def test_peg_representation():
    with open(PEG_GRAMMAR) as fd:
        text = fd.read()
    defs = PEGWalker().parse(text)
    if defs != PEG_DEFS:
        lines = []
        # check keys
        if set(defs.keys()) != set(PEG_DEFS.keys()):
            diff1 = set(defs.keys()) - set(PEG_DEFS.keys())
            lines.append(f'  new parse has extra keys: {sorted(diff1)}')
            diff2 = set(PEG_DEFS.keys()) - set(defs.keys())
            lines.append(f'  old parse has extra keys: {sorted(diff2)}')
        # check values
        for key in sorted(defs.keys()):
            if defs[key] != PEG_DEFS[key]:
                lines.append(f'  new parse for key "{key}":')
                lines.append(f'    {defs[key]}')
                lines.append(f'  old parse for key "{key}":')
                lines.append(f'    {PEG_DEFS[key]}')
        assert defs == PEG_DEFS, 'Parsed PEG defs do not match internal defs\n' + '\n'.join(lines)


def test_trace():
    try:
        walker = PEGWalker()
        walker.debug = True
        walker.parse('syntax = ;')
        assert False
    except SyntaxError as e:
        pass
    except Exception as e:
        print(e)
        assert False


def test_anbncn():
    parser = create_parser(dedent('''
        s = &(ab "c") "a"+ bc;
        ab = "a" ab? "b";
        bc = "b" bc? "c";
    '''.lstrip('\n').rstrip(' ')))
    for n in range(1, 10):
        s = ''.join(n * letter for letter in 'abc')
        parser.parse(s, 's')
        for i in range(2):
            for d in (-1, 1):
                s = ''.join(
                    (n + (d if i == j else 0)) * letter
                    for j, letter in enumerate('abc')
                )
                try:
                    parser.parse(s, 's')
                    assert False
                except SyntaxError as e:
                    pass


def main():
    test_ast_structure()
    test_peg_walker()
    test_peg_representation()
    test_trace()
    test_anbncn()


if __name__ == '__main__':
    main()
