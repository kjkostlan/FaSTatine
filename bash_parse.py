# Simple bash-like parsing tools and Python vs bash dection.
# Used in the command line to convert slick bash one-liners to Python and save typing.
# DEFINITELY NOT intended to be a comprehensive bash parser or intrepreter.
# But 90% of what a user types into a bash shell is <10% of the full syntax and is one-line cmds.
    # (And I don't know enough bash!)
    # (A much more comprehesive C implementation: https://github.com/clarity20/bash2py)

#https://www.cyberciti.biz/tips/bash-shell-parameter-substitution-2.html
#https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html
#https://opensource.com/article/18/5/you-dont-know-bash-intro-bash-arrays
#https://linuxhint.com/simulate-bash-array-of-arrays/     Bash has no nested arrays!?
import sys, re, numba
import numpy as np
from . import python_parse

# False: Bash behaves more consistently and intuitivly.
# True: Bash behaves more like Bash.
strict_mode = False

################################### The BASH-runtime world ########################

def BRG(start, end, step=1):
    # Bash range. Can be letter or number range.
    if type(start) is str:
        return [chr(i) for i in range(ord(start),ord(end),step)]
    else:
        return list(range(start, end, step))

def BVC(*items):
    return items

def BEX(*items):
    # Brace expansion. Odd indexes are expanded.
    # See: https://unix.stackexchange.com/questions/402315/nested-brace-expansion-mystery-in-bash
    dims = len(items)
    items1 = []
    if dims==0:
        return ''
    kvals = []
    for i in range(dims):
        if i%2==0:
            kvals.append(1)
            items1.append([str(items[i])])
        elif type(items[i]) is list or type(items[i]) is tuple:
            kvals.append(len(items[i]))
            items1.append([str(itm) for itm in items[i]])
        else:
            kvals.append(1)
            items1.append([str(items[i])])

    ixss = [x.ravel(order='F') for x in np.meshgrid(*[np.arange(k) for k in kvals])]
    out = []
    for i in range(len(ixss)):
        ixs = [ixss[o][i] for o in range(dims)]
        out.append(''.join([items1[j] for j in range(dims)]))

    return out

def BCT(*items):
    # Concatenation with string output.
    return ' '.join(items)

def BIF(cond, if_true, if_false):
    # Requires wrapping the true and false in lambdas.
    return if_true() if bool(cond) else if_false()

def add_bash_syntax_fns(module_name):
    # setattr for functions with specific bash syntatical constructs, such as A{1,2}B
    m = sys.modules[module_name]
    fns = {'BRG':BRG,'BVC':BVC,'BEX':BEX,'BCT':BCT,'BIF':BIF}
    for k,v in fns.items():
        setattr(m, k, v)

################################### The BASH-compiletime world ########################

class Symbol:
    # Symbols are string literals except that they don't get quoted.
    def __init__(self, val):
        self.val = str(val)
    def __str__(self):
        return self.val
    def __repr__(self):
        return self.val

#@numba.njit # The compile time is slower than the time saved in this case (but not in the pyparse case).
def _fsm_core_bash(x):
    #https://devhints.io/bash
    #https://riptutorial.com/bash/example/2465/quoting-literal-text
    #https://tldp.org/HOWTO/Bash-Prog-Intro-HOWTO-5.html
    # Notes:
    #  a=b must not have space around the = else command not found instead of var set.
    #In bash some strings are not quoted (i.e. taken as literals).
    #Note: The bash syntax is very complex. This function is not intended to handle all cases as our use of bash syntax is minimal.

    singleline_comment = False # # = 35; \n = 10,
    colon_comment = False # : = 58, ' = 39. Only recognizes :' or : ' one space (space = 32). : is a do-nothing used in other places.
    double_gt_comment_ix_start = -1 #<<comment\n\ncomment. Note that comment can be replaced with any matching pair of normal txt.
    double_gt_comment_ix_end = -1 # Range check. Reset to -1 when it is no longer in comment mode. Exclusive ix.
    escape = False # Much like Python!
    plus_minus_as_alphanum = True # Most of the time they count.
    quote = 0 # Single and double quotes are also Pythonesque. But we set it to 3 to indicate a naked quote.
    freshness_deluxe = 1 # The first command in a line or $() or after-; is not quote=3 but instead a symbol.
    paren_lev = 0
    in_multiplex_brace = 0
    aln = python_parse.alphanum_status(x)

    N = len(x)

    token_types = np.zeros(N, dtype=np.int32)
    inclusive_paren_nests = np.zeros(N, dtype=np.int32)
    quote_types = np.zeros(N, dtype=np.int32)

    ci1 = -1; ci2 = -1; ci3=-1
    for i in range(N):
        ci = x[i]
        if i<N-1:
            ci1 = x[i+1]
        if i<N-2:
            ci2 = x[i+2]
        if i<N-3:
            ci3 = x[i+3]
        inclusive_paren_nests[i] = paren_lev # Default.

        escape1 = escape; quote1 = quote; double_gt_comment_ix_start1 = double_gt_comment_ix_start
        singleline_comment1 = singleline_comment; colon_comment1 = colon_comment; freshness_deluxe1 = freshness_deluxe
        in_multiplex_brace1 = in_multiplex_brace
        in_comment = singleline_comment or colon_comment or (double_gt_comment_ix_start>-1)
        open_paren = ci==0x28 or ci==0x5B or ci==0x7B
        close_paren = ci==0x29 or ci==0x5D or ci==0x7D
        paren_valid = not escape and (quote==0 or quote==3 or (i>0 and x[i-1]==0x24 and (i==1 or x[i-2] != 0x5C)))
        alni_plus = aln[i] or (plus_minus_as_alphanum and (ci==0x2d or ci==0x2b)) # - and + can act like alpahum sometimes.

        if escape:
            token_types[i] = token_types[i-1]
            escape1 = False
        if quote==1 and not in_comment and not escape:
            if ci==0x27:
                quote1 = 0
            token_types[i] = 3; quote_types[i]=1
        if quote==2 and not in_comment and not escape:
            if ci==0x22:
                quote1 = 0
            elif ci==0x24: # $
                token_types[i] = 2; quote_types[i]=0; freshness_deluxe1=1; quote1 = 0
            token_types[i] = 3; quote_types[i]=2
        if quote==3 and not in_comment and not escape:
            if ci==0xA or ci==0x20 or ci==0x9: # Space newline.
                quote1 = 0; token_types[i] = 0; quote_types[i]=0
            elif ci==0x24: # $
                token_types[i] = 2; quote_types[i]=0; freshness_deluxe1=1; quote1 = 0
            else:
                token_types[i] = 3; quote_types[i]=3
        if singleline_comment:
            if ci==0xA:
                token_types[i] = 0
                singleline_comment1 = False
            else:
                token_types[i] = 6
        if colon_comment:
            if ci==0x27:
                colon_comment1 = False
            token_types[i] = 6
        if double_gt_comment_ix_start>-1 and not escape:
            if aln[i]>0: # Ignores plus_minus_as_alphanum I think.
                all_match = True
                K = double_gt_comment_ix_end - double_gt_comment_ix_start
                for j in range(K): # Equal token check.
                    j0 = double_gt_comment_ix_start+j; j1 = i-K+j+1 # When j = K-1 we get j1 = i is the last letter.
                    if j1>=N or x[j1] != x[j0]:
                        all_match = False; break
                if all_match:
                    double_gt_comment_ix_start1 = -1
                    double_gt_comment_ix_end = -1
        if paren_valid and close_paren:
            freshness_deluxe1 = 0 # It forces a quote next.
            in_multiplex_brace1 = 0 # Clear.
        if ci==0x5C and not in_comment and not escape:
            escape1 = True
        if not escape and not in_comment and quote==0:
            if alni_plus>0 and freshness_deluxe==0:
                token_types[i] = 3; quote1 = 3; quote_types[i]=3
            if alni_plus>0 and freshness_deluxe>0:
                token_types[i] = 1
            if ci==0x20 or ci==0x29 and freshness_deluxe>0:
                freshness_deluxe1 = 0
            if ci==0xA: # Newline or ; makes things very fresh.
                freshness_deluxe1 = 1
            if ci==0x3D: # = sign quotes the next thing.
                freshness_deluxe1 = 0
            if (ci==0x20 or ci==0xA or ci==0x9) and ci1==0x23: # Space/newline then # enters in a comment.
                singleline_comment1 = True
            if i==0 and ci==0x23:
                singleline_comment1 = True; token_types[i] = 6
            if ci==0x24: # dollar sign breaks the quote.
                freshness_deluxe1 = 1
                token_types[i] = 2
            if ci==0x2a or ci==0x2f or ci==0x25 or ci==0x3d or ci==0x26 or ci==0x7c or ci==0x5e or ci==0x3e or ci==0x3c or ci==0x7e or ci==0x40 or ci==0x2e or (not plus_minus_as_alphanum and (ci==0x2d or ci==0x2b)):
                token_types[i] = 2 #Maybe more symbolic chars belong here?
            if (ci==0x20 or ci==0xA) and ci1==0x3A and (ci2==0x27 or (ci2==0x20 and ci3==0x27)): # The :'  comment.
                colon_comment1 = True; token_types[i] = 6
            if ci==0x27:
                quote1 = 1; token_types[i] = 3; quote_types[i]=1
            if ci==0x22:
                quote1 = 2; token_types[i] = 3; quote_types[i]=2
        if open_paren and paren_valid:
            paren_lev = paren_lev+1; token_types[i] = 4
            inclusive_paren_nests[i] = paren_lev
            quote_types[i]=0 # just for the paren itself.
        if close_paren and paren_valid:
            paren_lev = paren_lev-1; token_types[i] = 5
            quote_types[i]=0
        if (escape or ci != 0x24) and ci1==0x7B and (quote==3 or quote1==3):
            in_multiplex_brace1 = 1 # Special multiplex braces make the commas into spaced tokens.

        if in_multiplex_brace and not escape and ci==0x2C:
            token_types[i] = 0 # Commas in the special brace expansion.
        escape = escape1; quote = quote1; double_gt_comment_ix_start = double_gt_comment_ix_start1
        singleline_comment = singleline_comment1; colon_comment = colon_comment1; freshness_deluxe = freshness_deluxe1
        in_multiplex_brace = in_multiplex_brace1

    return token_types, inclusive_paren_nests, quote_types

def fsm_parse(txt):
    #token, paren, quote_type
    x = np.frombuffer(txt.encode('UTF-32-LE'), dtype=np.uint32)
    return _fsm_core_bash(x)
