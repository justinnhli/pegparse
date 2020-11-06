#!/usr/bin/env python3

import sys
from textwrap import dedent
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pegparse import PEG_GRAMMAR, PEG_DEFS, ASTNode, PEGWalker
from pegparse import create_parser


def assert_equal(actual, expected, message_stem):
    message = '\n'.join([
        message_stem,
        'expected: ' + str(expected),
        'actual: ' + str(actual),
    ])
    assert actual == expected, message


def test_descendants():
    # setup
    parser = create_parser(dedent('''
        expression = operand ( operator operand )*;
        operand = paren_expression
                | number;
        paren_expression = "(" expression ")";
        operator = "+"
                 | "-"
                 | "*"
                 | "/";
        number = ( DIGIT )+;
    '''.lstrip('\n').rstrip(' ')))
    ast = parser.parse('(1+(2*3))-5', 'expression')
    message_stem = 'descendants() docstring is incorrect'
    # tests
    path = 'operand/paren_expression/expression/operand'
    matches = [node.match for node in ast.descendants(path)]
    assert_equal(['1', '(2*3)'], matches, message_stem)
    descendants = ast.descendants('operand')
    assert_equal(
        ['(1+(2*3))', '5'],
        [node.match for node in descendants],
        message_stem
    )
    assert_equal(
        ['operand', 'operand'],
        [node.term for node in descendants],
        message_stem
    )
    descendants = ast.descendants('operand/*')
    assert_equal(
        ['(1+(2*3))', '5'],
        [node.match for node in descendants],
        message_stem
    )
    assert_equal(
        ['paren_expression', 'number'],
        [node.term for node in descendants],
        message_stem
    )


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
    test_descendants()
    test_peg_walker()
    test_peg_representation()
    test_trace()
    test_anbncn()


if __name__ == '__main__':
    main()
