import re, numba
import numpy as np

py_kwds = {'import','from','def','lambda','class', 'try','except','raise','assert','finally',\
           'await','async','yield', 'if','or','not','elif','else','and','is', \
           'while','for','in','return','pass','break','continue', \
           'global','with','as','nonlocal',  'del',\
           'False','None','True'}

@numba.njit
def alphanum_status(x): # 1 = alpha or _. 2 = numeric. 3 = unicode.
    out = np.zeros(x.shape, dtype=np.int32)
    out[(x>=0x30)*(x<=0x39)] = 2 #0-9
    out[(x>=0x41)*(x<=0x5A)] = 1 #A-Z
    out[x==0x5F] = 1 # Underscore.
    out[(x>=0x61)*(x<=0x7A)] = 1 #a-z
    out[x>=0xC0] = 3 # Unicode
    return out

@numba.njit
def _fsm_core(x):
    N = x.size
    token_types = np.zeros(N, dtype=np.int32)
    inclusive_paren_nests = np.zeros(N, dtype=np.int32)
    indent_spaces = np.zeros(N, dtype=np.int32) # Can include multible lines.

    comment = False
    quote = 0; quote3 = 0; # 1 = single, 2 = double.
    paren_lev = 0
    leading_spaces = 0 # The number of spaces in "this" line, where "this" carries through non-indent-affecting stuff.
    next_leading_spaces = 0 # May or may not become leading_spaces.
    acc_leading_spaces = True
    escaped = False # the \ char.
    tok_ty = 0 # Running token type.
    quote1_count = 0; quote2_count = 0 # Detects triple quotes
    in_symbol = False; in_number = False
    newlines_are_in_comments = False # False makes sense: If the newline was removed problems would happen.
    aln = alphanum_status(x)

    ci0 =-1; ci00 = -1; ci000 = -1
    for i in range(N):
        ci = x[i]
        if i>0:
            ci0 = x[i-1]
        if i>1:
            ci00 = x[i-2]
        if i>2:
            ci000 = x[i-3]

        # Defaults that may change:
        indent_spaces[i] = leading_spaces
        token_types[i] = tok_ty
        inclusive_paren_nests[i] = paren_lev
        quote2_count = quote2_count+1 if (ci==0x22 and not escaped and not comment) else 0
        quote1_count = quote1_count+1 if (ci==0x27 and not escaped and not comment) else 0

        if escaped:
            escaped = False
        elif comment:
            if ci==0xA:
                if not newlines_are_in_comments:
                    token_types[i] = 0
                comment = False; tok_ty = 0
                next_leading_spaces = 0
                acc_leading_spaces = True
        elif ci==0x5C: # raw strings parse just like strings, so r"foo\\" and r"foo\"" are both valid.
            escaped=True
        elif quote3==1:
            if quote1_count==3:
                quote3 = 0; quote=0; quote1_count = 0; tok_ty = 0
        elif quote3>0:
            if quote2_count==3:
                quote3 = 0; quote=0; quote2_count = 0; tok_ty = 0
        elif quote==1:
            if ci==0x27:
                quote = 0; tok_ty = 0
        elif quote>0:
            if ci==0x22:
                quote = 0; tok_ty = 0
        else: # In symbol and/or in number or not so are "active" and lumped together.
            # Update in_number and in_symbol:
            ai = aln[i] # next alphanum character.
            if in_symbol and ai==0:
                in_symbol = False; tok_ty = 0; token_types[i] = 0
            elif in_number and ai==0 and ci !=0X2E and ci !=0x2D: # Alpha allowed in numbers, 0xABC, 1.5e-123.
                in_number = False; tok_ty = 0; token_types[i] = 0
            elif not in_number and not in_symbol and (ai==1 or ai==3):
                in_symbol = True; tok_ty = 1; token_types[i] = 1; in_number = False
            elif not in_symbol and not in_number and ai==2:
                in_number = True; tok_ty = 3; token_types[i] = 3; in_symbol = False
                if ci0==0x2D: # Include the leading - sign.
                    token_types[i-1] = 3

            if ci==0x27:
                quote = 1; tok_ty = 3; token_types[i] = 3
                if ci0==0x72: # Raw strings prepend with r and escape backslashes.
                    token_types[i-1] = 3
                if quote1_count==3:
                    quote3 = 1; quote = 0; quote1_count = 0
                    tok_ty = 3; token_types[i] = 3
                    if ci0==0x72:
                        token_types[i-1] = 3
            elif ci==0x22:
                quote = 2; tok_ty = 3; token_types[i] = 3
                if ci0==0x72:
                    token_types[i-1] = 3
                if quote2_count==3:
                    quote3 = 2; quote = 0; quote2_count = 0
                    tok_ty = 3; token_types[i] = 3
                    if ci0==0x72:
                        token_types[i-1] = 3
            elif ci==0x23:
                comment = True; token_types[i] = 6; tok_ty=6
            elif ci==0x28 or ci==0x5B or ci==0x7B:
                paren_lev = paren_lev+1; token_types[i] = 4
                inclusive_paren_nests[i] = paren_lev
            elif ci==0x29 or ci==0x5D or ci==0x7D:
                paren_lev = paren_lev-1; token_types[i] = 5
            elif tok_ty==0 and (ci==0x2b or ci==0x2d or ci==0x2a or ci==0x2f or ci==0x25 or ci==0x3d or ci==0x26 or ci==0x7c or ci==0x5e or ci==0x3e or ci==0x3c or ci==0x7e or ci==0x40 or ci==0x2e):
                token_types[i] = 2 # Operators and decorators. Does not change tok_type.

            # Complicated rules to compute leading spaces:
            if paren_lev==0:
                if ci==0xA: # newline.
                    next_leading_spaces = 0
                    acc_leading_spaces = True
                if ci==0x20 and acc_leading_spaces: # space.
                    next_leading_spaces = next_leading_spaces+1
                elif ci==0x9 and acc_leading_spaces: # tab (please do not use!)
                    next_leading_spaces = next_leading_spaces+4
                if acc_leading_spaces and ci !=0x9 and ci !=0xA and ci !=0x20 and ci !=0x23: # There is actually "stuff" in this line, so indent matters.
                    acc_leading_spaces = False
                    leading_spaces = next_leading_spaces
                    for j in range(i-leading_spaces,i+1): # The entire line gets it's leading space count.
                        indent_spaces[j] = leading_spaces
                if acc_leading_spaces and i==N-1:
                    for j in range(i-leading_spaces,i+1): # The entire line gets it's leading space count.
                        indent_spaces[j] = leading_spaces
    return token_types, inclusive_paren_nests, indent_spaces

def fsm_parse(txt):
    # Finite state machine-based parser.
    #https://stackoverflow.com/questions/75540444/python-efficently-convert-string-to-numpy-integer-array-character-by-character
    # Returns token, paren, indent as np arrays one per character.
    x = np.frombuffer(txt.encode('UTF-32-LE'), dtype=np.uint32)
    return _fsm_core(x)

def simple_tokens(code):
    # Simple tokenize that tries to isolate vars. Not smart about strings.
    pattern = '[ \n\[\]\{\}\(\)\#:\/\-=\+\%\*<>]+'
    return re.split(pattern, code)

def canon_str(code):
    return code.replace('\r\n','\n').replace('\t','    ')

def line_start_ixs(code_or_lines):
    # The character index of the first element in the line.
    if type(code_or_lines) is str:
        code_or_lines = code_or_lines.split('\n')
    ix = 0; out = []

    for l in code_or_lines:
        out.append(ix)
        ix = ix+len(l)+1
    if len(code_or_lines)>0 and len(code_or_lines[-1])==0:
        out[-1] = out[-1]-1 # tweak to prevent out-of-array errors.
    return out

def line_indent_levels(code):
    # Increases on indent, decreased on dedent.
    # Levl is relative to first line.
    # (Do not do too much of this "roll your own" text analysis, there are better libraries for that!)
    code = canon_str(code)
    toks, paren, sps = fsm_parse(code)
    lines = code.split('\n')
    sps_ea_line = []; ix = 0
    in_quote_line_start = []
    st_ixs = line_start_ixs(lines)
    for lx in range(len(st_ixs)):
        sps_ea_line.append(sps[st_ixs[lx]])
        in_quote_line_start.append(toks[st_ixs[lx]]==3)

    spaces2depth = {}
    last_depth = 0
    indent_nest = [0]
    for i in range(1, len(lines)):
        nsp = sps_ea_line[i]; nsp0 = sps_ea_line[i-1]
        if nsp==nsp0:
            depth = last_depth
        elif nsp>nsp0:
            depth = last_depth+1
            for i in range(nsp0+1, nsp+1): # Only i=nsp is syntactically allowed.
                spaces2depth[i] = depth
        elif nsp<nsp0:
            depth = spaces2depth.get(nsp,0)
        indent_nest.append(depth)
        last_depth = depth

    return indent_nest, in_quote_line_start, sps_ea_line

def sourcecode_defs(code, nest=True, unindent_nested=True):
    # Map from var name to def source. Option to nest inside defs and classes.
    # (nested defs will be reported here but are not seen as a module or class attribute).
    code = canon_str(code); lines = code.split('\n')
    levels, in_quote_line_starts, _ = line_indent_levels(code)
    nestings = []
    out = {}
    N = len(lines)
    for i in range(N):
        l0 = lines[i].strip()
        if l0.startswith('def') or l0.startswith('class'):
            vname = (simple_tokens(l0)+['__PARSEERROR__'])[1].strip()
            nestings.append(vname)
            vnamefull = '.'.join(filter(lambda x: len(x)>0, nestings))

            if nest or len(nestings)<=1:
                src_lines = []; hit_the_body_lines = False
                for j in range(i, len(lines)):
                    if levels[j]>levels[i]:
                        hit_the_body_lines = True
                    if levels[j]<=levels[i] and hit_the_body_lines:
                        break # hit_the_body_lines prevents comment lines just below the def (which inherit the def level) from stopping us.
                    src_lines.append(lines[j])
                if unindent_nested: # remove leading space shared among all lines.
                    nsp0 = len(lines[i])-len(lines[i].lstrip())
                    if nsp0>0:
                        for u in range(len(src_lines)):
                            if not in_quote_line_starts[u+i]:
                                nspj = len(src_lines[u])-len(src_lines[u].lstrip())
                                src_lines[u] = src_lines[u][min(nspj,nsp0):]
                src = '\n'.join(src_lines)
                out[vnamefull] = src
        elif i<N-1 and levels[i+1]>levels[i]:
            nestings.append('') # Non-nesting indent.
        elif i<N-1 and levels[i+1]<levels[i]: # dedent.
            nestings = nestings[0:levels[i+1]]
    return out

def statement_contents(code, line_f, dedent=True):
    # Extracts blocks of code for which line_f(the_line) is True.
    code = canon_str(code); lines = code.split('\n')
    levels, in_quote_line_starts, sps_ea_line = line_indent_levels(code)
    N = len(lines)
    out = []; i=0
    while i<N-1:
        outi = []; sp = 0; ixs = []
        if line_f(lines[i]):
            lev = levels[i]; sp = in_quote_line_starts[i]
            for j in range(i+1, N):
                if levels[j]<=lev:
                    break; i = j
                outi.append(lines[j]); ixs.append(j)
        if dedent:
            for ixi in range(len(ixs)):
                if not in_quote_line_starts[ixs[ixi]]:
                    for j in range(sps_ea_line[ixs[ixi]]): # inefficient but a bit more robust.
                      if outi[ixi].startswith(' '):
                          outi[ixi] = outi[ixi][1:]
        if len(outi)>0:
            out.append(outi)
        i = i+1
    return ['\n'.join(outi) for outi in out]

def get_main_blocks(txt):
    # The module must be loaded.
    # Returns a vector.
    def _line_f(line):
        no_comment = re.sub(r'#.*', '', line)
        gold = 'if __name__ == "__main__":'
        _rep = lambda txt: txt.replace('==','is').replace(' ','').replace('"',"'").strip()
        return _rep(no_comment)==_rep(gold)
    main_contents = statement_contents(txt, _line_f, dedent=True)
    return main_contents
