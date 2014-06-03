#!/usr/bin/env python3

import re

# TODO  have ASTNodes dynamically generate the match; saves memory from storing a string multiple times
#           this can be done by storing the full string via variable capture in a function which takes substring start and end positions

EBNF_DEFS = {
	"Syntax"     : ("AND", ("ONE-OR-MORE", ("ZERO-OR-MORE", "EmptyLine"), "Definition", "newline"), ("ZERO-OR-MORE", "EmptyLine")),
	"Definition" : ("AND", "Identifier", "Whitespace", "\"= \"", "Expression", "\";\""),
	"Expression" : ("OR", "Disjunct", "Except", "Conjunct"),
	"Disjunct"   : ("AND", "Atom", ("ONE-OR-MORE", "newline", "Whitespace", "\"| \"", "Atom")),
	"Except"     : ("AND", "Atom", ("ONE-OR-MORE", "newline", "Whitespace", "\"- \"", "Atom")),
	"Conjunct"   : ("AND", "Item", ("ZERO-OR-MORE", "\" \"", "Item")),
	"Item"       : ("OR", "ZeroOrOne", "ZeroOrMore", "OneOrMore", "Identifier", "Reserved", "Literal"),
	"Atom"       : ("OR", "Identifier", "Reserved", "Literal"),
	"ZeroOrOne"  : ("AND", "\"( \"", "Conjunct", "\" )?\""),
	"ZeroOrMore" : ("AND", "\"( \"", "Conjunct", "\" )*\""),
	"OneOrMore"  : ("AND", "\"( \"", "Conjunct", "\" )+\""),
	"Identifier" : ("AND", ("ONE-OR-MORE", "upper", ("ZERO-OR-MORE", "lower"))),
	"Reserved"   : ("AND", ("ONE-OR-MORE", "lower")),
	"Literal"    : ("OR", "DString", "SString"),
	"DString"    : ("AND", "'\"'", ("ZERO-OR-MORE", "NoDQuote"), "'\"'"),
	"SString"    : ("AND", "\"'\"", ("ZERO-OR-MORE", "NoSQuote"), "\"'\""),
	"NoDQuote"   : ("NOT", "print", "'\"'"),
	"NoSQuote"   : ("NOT", "print", "\"'\""),
	"Whitespace" : ("AND", ("ONE-OR-MORE", "blank")),
	"EmptyLine"  : ("AND", ("ZERO-OR-ONE", "\"#\"", ("ZERO-OR-MORE", "print")), "newline"),
}

REPETITION_MAPPING = {
	"ZeroOrMore" : "ZERO-OR-MORE",
	"ZeroOrOne"  : "ZERO-OR-ONE",
	"OneOrMore"  : "ONE-OR-MORE",
}

def create_parser(bnf):
	ast_root, chars_parsed = PEGParser(EBNF_DEFS).parse(bnf, "Syntax")
	if ast_root and chars_parsed == len(bnf):
		return PEGParser(_ast2defs(ast_root)), ast_root
	else:
		return None

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
				result = [child for child in result.children if child.term == term][0]
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
	def parse(self, string, term, complete=False):
		self.cache = {}
		self.indent = 0
		ast, parsed = self.dispatch(string, term, 0)
		if complete:
			if ast and parsed == len(string):
				return ast
			else:
				return ASTNode()
		else:
			return ast, parsed
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
		new_terms = tuple(["AND"] + list(terms[1:]))
		children = []
		last_pos = position
		ast, pos = self.dispatch(string, new_terms, position)
		while ast:
			children.extend(ast.children)
			last_pos = pos
			ast, pos = self.dispatch(string, new_terms, pos)
		return ASTNode("ZERO-OR-MORE", children, string[position:last_pos]), last_pos
	def match_zero_or_one(self, string, terms, position):
		new_terms = tuple(["AND"] + list(terms[1:]))
		ast, pos = self.dispatch(string, new_terms, position)
		if ast:
			return ast, pos
		return self.dispatch(string, "empty", position)
	def match_one_or_more(self, string, terms, position):
		new_terms = tuple(["AND"] + list(terms[1:]))
		ast, pos = self.dispatch(string, new_terms, position)
		if not ast:
			return ast, pos
		else:
			children = ast.children
			last_pos = pos
			ast, pos = self.dispatch(string, new_terms, pos)
			while ast:
				children.extend(ast.children)
				last_pos = pos
				ast, pos = self.dispatch(string, new_terms, pos)
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
				if nast:
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
		return ASTNode(""), position
	def debug_print(self, obj):
		if self.debug:
			print("    " * self.indent + str(obj))

def _ast2defs(ast):
	definitions = {}
	used_defs = set()
	for definition in ast.descendants("Definition"):
		identifier = definition.first_descendant("Identifier").match
		assert identifier not in definitions, "identifier {} is defined multiple times\n".format(identifier)
		expression = definition.first_descendant("Expression/*")
		if expression.term == "Disjunct":
			flattened, used = _ast2list(expression, "Atom", "OR")
			used_defs |= used
			definitions[identifier] = flattened
		elif expression.term == "Except":
			flattened, used = _ast2list(expression, "Atom", "NOT")
			used_defs |= used
			definitions[identifier] = flattened
		elif expression.term == "Conjunct":
			flattened, used = _ast2list(expression, "Item", "AND")
			used_defs |= used
			definitions[identifier] = flattened
		else:
			assert False, "Unknown expression type '{}'".format(expression.term)
	undefined = used_defs - (PEGParser.CORE_DEFS.keys()) - set(definitions.keys())
	assert len(undefined) == 0, "undefined identifiers: {}".format(", ".join(undefined))
	return definitions

def _ast2list(ast, descentry, operator):
	flattened = []
	used = set()
	for descendant in ast.descendants(descentry):
		item, used_ids = _ast2item(descendant.first_descendant("*"))
		flattened.append(item)
		used |= used_ids
	return tuple([operator] + flattened), used

def _ast2item(ast):
	if ast.term in REPETITION_MAPPING:
		return _ast2list(ast, "Conjunct/Item", REPETITION_MAPPING[ast.term])
	elif ast.term in ("Identifier", "Reserved"):
		return ast.match, set([ast.match])
	elif ast.term == "Literal":
		return ast.match, set()
	assert False, "Unknown expression type '{}'".format(ast.term)

if __name__ == "__main__":
	from argparse import ArgumentParser, FileType
	from sys import stdin
	arg_parser = ArgumentParser()
	arg_parser.set_defaults(verbose=False)
	arg_parser.add_argument("-e", dest="expression", action="store",      help="starting expression; if omitted, first defined symbol is used")
	arg_parser.add_argument("-g", dest="grammar",    action="store",      help="EBNF grammar file")
	arg_parser.add_argument("-v", dest="verbose",    action="store_true", help="show what the parser is doing")
	arg_parser.add_argument("text", metavar="TEXT_FILE", action="store", nargs="?", help="text file to be parsed", type=FileType("r"), default=stdin)
	args = arg_parser.parse_args()
	if args.grammar:
		with open(args.grammar, "r") as fd:
			parser, ast = create_parser(fd.read())
		if not parser:
			print("error: grammar file cannot be parsed")
			exit(1)
		if args.expression:
			if args.expression not in parser.custom_defs:
				print("error: specified expression not defined")
				exit(1)
			term = args.expression
		else:
			term = ast.first_descendant("Definition/Identifier").match
	else:
		parser = PEGParser(EBNF_DEFS)
		term = "Syntax"
	parser.debug = args.verbose
	contents = args.text.read()
	ast, chars_parsed = parser.parse(contents, term)
	length = len(contents)
	if not ast or chars_parsed != length:
		print("failed: only parsed {} of {} characters\n".format(chars_parsed, length))
		exit(1)
	else:
		ast.pretty_print()
