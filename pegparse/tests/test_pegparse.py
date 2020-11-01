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
        Expression = Operand ( Operator Operand )*;
        Operand = ParenExpression
                | Number;
        ParenExpression = "(" Expression ")";
        Operator = "+"
                 | "-"
                 | "*"
                 | "/";
        Number = ( digit )+;
    '''.lstrip('\n').rstrip(' ')))
    ast = parser.parse('(1+(2*3))-5', 'Expression')
    message_stem = 'descendants() docstring is incorrect'
    # tests
    path = 'Operand/ParenExpression/Expression/Operand'
    matches = [node.match for node in ast.descendants(path)]
    assert_equal(['1', '(2*3)'], matches, message_stem)
    descendants = ast.descendants('Operand')
    assert_equal(
        ['(1+(2*3))', '5'],
        [node.match for node in descendants],
        message_stem
    )
    assert_equal(
        ['Operand', 'Operand'],
        [node.term for node in descendants],
        message_stem
    )
    descendants = ast.descendants('Operand/*')
    assert_equal(
        ['(1+(2*3))', '5'],
        [node.match for node in descendants],
        message_stem
    )
    assert_equal(
        ['ParenExpression', 'Number'],
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
    assert defs == PEG_DEFS, 'Parsed PEG defs do not match internal defs'


def test_trace():
    with open(PEG_GRAMMAR) as fd:
        line = fd.readline().strip()
    line = line[:-1] + ' ' + line[-1]
    try:
        walker = PEGWalker()
        walker.debug = True
        walker.parse(line)
        assert False
    except SyntaxError as e:
        pass
    except Exception as e:
        print(e)
        assert False


def main():
    test_descendants()
    test_peg_walker()
    test_peg_representation()
    test_trace()


if __name__ == '__main__':
    main()
