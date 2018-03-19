#!/usr/bin/env python3

from pegparse import EBNF_GRAMMAR, EBNF_DEFS, ASTNode, EBNFWalker


def test_ebnf_walker():
    with open(EBNF_GRAMMAR) as fd:
        text = fd.read()
    ast = EBNFWalker().parse(text)
    assert isinstance(ast, dict), 'EBNFWalker failed to return a dict'


def test_ebnf_representation():
    with open(EBNF_GRAMMAR) as fd:
        text = fd.read()
    defs = EBNFWalker().parse(text)
    assert defs == EBNF_DEFS, 'Parsed EBNF defs do not match internal defs'


def main():
    test_ebnf_walker()
    test_ebnf_representation()


if __name__ == '__main__':
    main()
