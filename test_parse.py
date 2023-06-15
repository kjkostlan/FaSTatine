# testing the parsers.
from . import python_parse, bashparse
import numpy as np

def ints_eq(a,b):
    # Equality of all integer elements, handles lists and/or np arrays.
    if len(a) != len(b):
        return False
    return np.sum(np.abs(np.asarray(a)-b))==0

def test_assigned_var_parse(simple_assigned_vars):
    out = True
    f = simple_assigned_vars
    out = out and f('x = 1')==['x']
    txt = '''
foo = [1,2,3]
bar,baz = 4,5
'''
    out = out and f(txt)==['foo','bar','baz']
    return out

def test_vdif_report(vdif_report):
    f = vdif_report
    out = True
    #vdif_report(vars0, vars1, the_input, err)
    x = f({'x':0},{'x':0}, 'x = 1', '')
    out = out and '(x = 0)' in x and 'no vars changed' in x
    x = f({'x':0, 'y':0},{'x':1, 'y':0}, 'x = 1', '')
    out = out and '(' not in x and 'x = 1' in x and 'no vars' not in x and 'y' not in x
    x = f({'x':0, 'y':0},{'x':1, 'y':1}, 'x = 1', '')
    out = out and 'y' in x
    x = f({'_x':0},{'_x':1}, '_x = 1', '')
    out = out and len(x)<5
    x = f({'_x':0},{'_x':1}, '_x = 1\n_x', '')
    out = out and '_x' in x
    return out

def test_bash_parse():
    # Bash parsing (tokens and AST).
    # This is NOT a complete test! We are not writing a bash parser.
    out = True

    x = bash_parse.ParsedStr('x=-ls')
    out = out and [1,2,3,3,3]

    x = bash_parse.ParsedStr('foo bar baz')
    out = out and ints_eq(x.token,[1,1,1,0,3,3,3,0,3,3,3])

    x = bash_parse.ParsedStr('A B=C D')
    out = out and ints_eq(x.token,[1,0,3,3,3,0,3])

    x = bash_parse.ParsedStr('A=B')
    out = out and ints_eq(x.token,[1,2,3])

    x = bash_parse.ParsedStr('echo $bar')
    out = out and ints_eq(x.token, [1,1,1,1,0,2,1,1,1])

    x = bash_parse.ParsedStr('echo foo${bar}baz')
    out = out and ints_eq(x.token, [1,1,1,1,0,3,3,3,2,4,1,1,1,5,3,3,3])

    x = bash_parse.ParsedStr('echo 123$(echo abc)def')
    out = out and ints_eq(x.token, [1,1,1,1,0,3,3,3,2,4,1,1,1,1,0,3,3,3,5,3,3,3])
    out = out and ints_eq(x.paren, [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0])

    x = bash_parse.ParsedStr('echo AA{1\,2,3}B, C')
    out = out and ints_eq(x.token, [1,1,1,1,0,3,3,4,3,3,3,3,0,3,5,3,3,0,3])

    t = bash_parse.ast_bash('foo bar')
    out = out and str(t)=="[foo, 'bar']"

    t = bash_parse.ast_bash('echo foo{1,2}bar{3,4}{5,6}') # freezes here.
    out = out and str(t)=="[echo, [BEX, 'foo', [ARR, '1', '2'], 'bar', [ARR, '3', '4'], '', [ARR, '5', '6']]]"

    x = bash_parse.ParsedStr('x=$(ls -a -r)')
    out = out and ints_eq(x.token,[1,2,2,4,1,1,0,3,3,0,3,3,5])

    return out

def test_bash2py():
    # Tests A: is it Python and B: if it is Bash what is the Python code?
    # Is it Python? Does our simplistic AST work for enough cases?

    def is_bash(txt, _assert=True):
        return bash_parse.is_line_bash(bash_parse.ParsedStr(txt, python=True), _assert)

    out = True

    x = 'foo.bar()'; y = None
    out = out and not is_bash(x)

    x = 'x = 1'; y = None
    out = out and not is_bash(x)

    x = 'ls'; y = bash_parse.bash2py(x).replace('"',"'")
    out = out and (is_bash(x) is None) and y.strip()=='ans = ls()' # Even though is_bash is None, it likely is a Bash line.

    x = 'echo $bar'; y = bash_parse.bash2py(x).replace('"',"'")
    out = out and is_bash(x) and y.strip()=='ans = echo(bar)'

    x = 'echo bar'; y = bash_parse.bash2py(x).replace('"',"'")
    out = out and is_bash(x) and y.strip()=="ans = echo('bar')"

    x = 'x=$(ls -a -r)'; y = bash_parse.bash2py(x).replace('"',"'") # This example is wrong!
    out = out and is_bash(x) and y.strip()=="x = ls('-a', '-r')"

    x = '_ans, _err = run("blender", [])'; y = None
    out = out and (is_bash(x) is False)

    x = 'import foo'; y = None
    out = out and (is_bash(x) is False)

    x = 'from foo import bar'; y = None
    out = out and (is_bash(x) is False)

    x = 'def some_test(*args):'; y = None
    out = out and (is_bash(x) is False)

    x = 'return foo'; y = None
    out = out and (is_bash(x) is False)

    x = 'assert foo'; y = None
    out = out and (is_bash(x) is False)

    x = 'x=grep if ls>1 else touch'; y = None # Trick question.
    out = out and (is_bash(x) is False)

    x = 'foo: Foo'; y = None # Trick question.
    out = out and (is_bash(x, False) is None)

    x = 'x= a[b]'; y = None # Trick question.
    out = out and (is_bash(x) is False)

    x = 'x=a[b]'; y = None # Trick question.
    out = out and (is_bash(x, False) is None)

    x = 'a = 1 #!/bin/bash'; y = None
    out = out and is_bash(x)

    x = 'grep foo #Grab the grep'; y = None
    out = out and is_bash(x)

    x = 'a = 1 #a is a scalar and this is a comment.'; y = None
    out = out and (is_bash(x) is False)

    x = 'grep foo #!/bin/python'; y = None
    out = out and (is_bash(x) is False)

    x = 'ls -a'; y = None # Trick question, since 10 -3 is valid Python.
    out = out and (is_bash(x, False) is None)

    x = 'ls -n "foo" -b bar'; y = bash_parse.bash2py(x).replace('"',"'")
    out = out and y == "ans = ls('-n', 'foo', '-b', 'bar')"

    x = 'mv foo $bar'; y = bash_parse.bash2py(x).replace('"',"'")
    out = out and y == "ans = mv('foo', bar)"

    return out

def test_py_fsm():
    out = True

    iEq = ints_eq
    tok, paren, sps = python_parse.fsm_parse('foo')
    out = out and iEq(tok,[1,1,1]) and iEq(paren,[0,0,0]) and iEq(sps,[0,0,0])

    txt = '''
def foo(bar):
    z = 1
w = 3
'''
    tok, paren, sps = python_parse.fsm_parse(txt)
    out = out and (np.sum(tok==3)==2) and (np.sum(tok==6)==0) and (np.sum(tok==4)==1) and (np.sum(tok==5)==1) and (np.sum(tok==1)==11) and (np.sum(paren)==5) and (np.max(sps)==4) and (np.sum(sps)==10*4)

    txt = '''
a = "123 #4(56)[78]{90}"
b = """foo #aaa
       bar "#bbb
       baz"""
c = 1+2 # which is 3.
d = [1234]
e = {(5)}
'''
    tok, paren, sps = python_parse.fsm_parse(txt)
    out = out and (np.sum(tok==3)>24) and (np.sum(tok==6)==13) and (np.max(paren)==2) and (np.sum(paren)==14) and (np.sum(sps)==0)

    txt = """
a = '''foo'''
def b(bar, baz=1):
    b = quantum_mech_is_awesome
    def c(c1):
        c2 = c3
    BQP = win
"""
    tok, paren, sps = python_parse.fsm_parse(txt)
    out = out and (np.sum(tok==3)==9+1) and (set(sps)==set([0,4,8])) and (np.sum(sps==8)==16)

    txt = '''
def python_spreadsheet(x):
    abc

    def
 #comment

    ghi
jkl
'''

    tok, paren, sps = python_parse.fsm_parse(txt)
    out = out and (np.sum(sps==4)==8+1+8+10+1+8)

    txt = r'''
"foo\"bar"
Not in a quote
"baz\\"
Not in a quote
"bats\\\""
Not in a quote
'''
    tok, paren, sps = python_parse.fsm_parse(txt)
    out = out and (np.sum(tok==3)==10+7+10)

    txt = r'''
"foo"
no_quote
r"bar"
no_quote
'''
    tok, paren, sps = python_parse.fsm_parse(txt)
    out = out and (np.sum(tok==3)==5+6)

    txt = r'''
r"foo\\"
not in a quote
r"bar\""
'''
    tok, paren, sps = python_parse.fsm_parse(txt)
    out = out and (np.sum(tok==3)==8+8) # the r is quoted, but otherwise the rules for string termination are unchanged.

    txt = """
'''foo''''''bar'''
not in a quote
'''123''''456'''
Also not in a quote. Make sense?
"""
    tok, paren, sps = python_parse.fsm_parse(txt)
    out = out and (np.sum(tok==3)==9+9+9+5+2)

    txt = r'''
# Foo code:
def foo(x):
    y = 1
#comment1
    yy=1

def bar(z): # Bar code
    w = 10
'''.strip()

    tok, paren, sps = python_parse.fsm_parse(txt)
    out = out and np.sum(sps>0)==10+10+9+11

    return out

def test_bash_fsm():
    out = True
    iEq = ints_eq

    txt = 'foo'
    tok, paren, quote = bash_parse.fsm_parse_bash(txt)
    out = out and iEq(tok,[1,1,1]) and iEq(paren,[0,0,0])

    txt = r'''
echo foo bar
'''
    tok, paren, quote = bash_parse.fsm_parse_bash(txt)
    out = out and iEq(tok,[0,1,1,1,1,0,3,3,3,0,3,3,3,0])

    # No more exmples for now, there is no real need to write a comprehensive FSM parser for bash!
    return out

def test_py_defs():
    # Can we use defs.
    iEq = ints_eq; out = True

    code = r'''
def foo():
    # Bar
    return True

def bar():
    # Foo
    return False
'''.strip()

    levels, _, _ = python_parse.line_indent_levels(code)
    defs = python_parse.sourcecode_defs(code, nest=True)
    out = out and iEq(levels,[0,0,1,1,0,0,1])
    out = out and '# Bar' in defs['foo'] and '# Foo' in defs['bar']

    code = r'''
# Foo code:
def foo(x):
    y = 1

def bar(z): # Bar code
    w = 10
'''.strip()

    levels, _, _ = python_parse.line_indent_levels(code)
    out = out and iEq(levels,[0,0,1,1,0,1]) # the second last 1 could also be a 0.
    defs = python_parse.sourcecode_defs(code, nest=True)

    out = out and 'y = 1' in defs['foo'] and 'w = 10' in defs['bar'] and '# Bar code' in defs['bar']

    code = r'''
def foo(x):
    y = 1

"""def commented(x):
    y = 1"""

def bar(z): # Bar code
    w = 10
'''.strip()

    defs = python_parse.sourcecode_defs(code, nest=True)
    out = out and 'commented' not in defs and len(defs)==2

    code = r'''
def foo(x):
    y = 1
    """indented...
without bieng indented.
          or bieng too indented.
Freedom in the triple quotes!"""
    yy = y+2
zz = 1

def bar(z): # Bar code
    w = 10
'''.strip()

    defs = python_parse.sourcecode_defs(code, nest=True)
    out = out and 'y = 1' in defs['foo'] and 'without bieng' in defs['foo'] and 'or bieng too' in defs['foo']
    out = out and 'triple quotes!"""' in defs['foo'] and 'yy = y+2' in defs['foo']
    out = out and 'zz = 1' not in defs['foo'] and 'bar' in defs

    code = r'''
def foo(x):
    y = 2*x

class Vehicle:
    def __init__(self):
        pass

    def motor(self, rpm):
        if self.electric:
            """Why not download a
   DOT-approved engine sound mod?"""
            def synth(Hz):
                fourier.waves.softsquare()
            synth(rpm/60)
        else:
            CO2_ppm = 420
            ping(self.pistons*rpm/60)

    def keys(self, key):
        return self.bitting==key

def bar(z): # Bar code
    cycling = 20 #km/hr speed.

'''.strip()

    defs = python_parse.sourcecode_defs(code, nest=True, unindent_nested=True)
    defsA = python_parse.sourcecode_defs(code, nest=True, unindent_nested=False)
    defs0 = python_parse.sourcecode_defs(code, nest=False)
    out = out and len(defs0)==3 and 'foo' in defs0 and 'bar' in defs0 and 'Vehicle' in defs0
    out = out and len(defs)==7 and 'Vehicle.__init__' in defs and 'Vehicle.motor.synth' in defs
    out = out and 'fourier.waves' in defs['Vehicle.motor.synth'] and 'fourier.waves' in defs['Vehicle.motor'] and 'fourier.waves' in defs['Vehicle']
    out = out and 'bitting' not in defs['Vehicle.motor'] and 'bitting' in defs['Vehicle']
    out = out and '\n        ping(self' in defs['Vehicle.motor'] and '\n   DOT-approved' in defs['Vehicle.motor']
    out = out and '\n            ping(self' in defsA['Vehicle.motor'] and '\n   DOT-approved' in defsA['Vehicle.motor']
    out = out and 'cycling = 20' in defs['bar']

    return out
