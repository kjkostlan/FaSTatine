# Finite state parser.
# TODO: make sure it conforms to legend.txt.
# WARNING: Don't dig too much into C++ code, there are libraries for that!
import numpy as np
import numba

#def txt2ints(txt): # TODO refactor here.


@numba.njit
def max_depth(token_types):
    n = token_types.size
    d = 0; mx = 0
    for i in range(n):
        ty = token_types[i]
        if ty == 4:
            d = d+1
            if d>mx:
                mx = d
        elif ty==5:
            d = d-1
    return mx

@numba.njit
def _token_core(c_txt_ints, buffer_len):
    # buffer is to avoid getting off the end.
    #print('TODO: re-enable njit.')
    #np.frombuffer( 'foo'.encode('utf8'),dtype=np.uint8)
    n = c_txt_ints.size-buffer_len
    out = np.zeros((3, n+buffer_len), dtype=np.int64)

    k = 0
    in_str = -1
    in_num_literal = False
    escaping = False
    single_comment = False
    multi_comment = False
    last_char_is_op = False
    tok_start_ix = 0
    current_ty = -1 # At the beginning, this is the token type we are inside.
    print_state = False
    minus_prepend = False
    noun = False
    in_alphanum_symbol = False
    period_is_op = False # Optional.
    for i in range(n+1):
        char = c_txt_ints[i]; char1 = c_txt_ints[i+1]
        is_op = char==33 or char== 37 or char== 38 or (char >=42 and char <=47 and char != 44)\
                  or char>=60 and char<=63 or char == 94 or char == 124 or char == 126 # Does not include :
        if char==46:
            is_op = period_is_op
        if print_state:
            print('State:','txt?',in_str,'num?',in_num_literal,'esc?',escaping,\
                  '//?',single_comment,'/**/?',multi_comment,'op?',last_char_is_op,\
                  'current_ty:', current_ty,'ixs:',tok_start_ix,i, [char, char1])
        if i>0:
            char0 = c_txt_ints[i-1]
        else:
            char0 = -1
        next_ty = -1
        new_token = False; next_in_alphanum_symbol = False
        if escaping:
            escaping = False
        elif single_comment:
            if char== 10: #\n
                single_comment = False
        elif multi_comment:
            if char==47: #forward /
               if char0 == 42: #*
                   multi_comment = False
        elif char==92: #Backslash escapes everything that isn't a comment.
            in_num_literal = False
            escaping = True
        elif char==39 or char==34: # Quotes (single or double).
            in_num_literal = False
            if in_str <0:
                new_token = True
                in_str = char
            elif char==in_str:
                in_str = -1
            next_ty = 3; noun = True
        elif in_str>0: # Not much to do in strings.
            continue
        elif char==32 or char==9 or char==10 or char==13: #whitespace
            next_ty = 0; in_num_literal = False
            if current_ty != 0:
                new_token = True
        elif not in_num_literal and \
            ((not in_alphanum_symbol and char>=48 and char<=57) or (char==45 and char1>=48 and char1<=57 and not noun)): # Number literals.
            next_ty = 3; noun = True
            new_token = True
            in_num_literal = True
            minus_prepend = char == 45
        elif char==44 or char==59 or (char == 58 and char1 !=58 and (i==0 or c_txt_ints[i-1] !=58)): # , ;, or : not next to another :
            next_ty = 6; new_token = True; in_num_literal = False; noun = False
        elif char == 91 or char == 123 or char == 40: # Openings.
            next_ty = 4; new_token = True; in_num_literal = False; noun = False
        elif char == 93 or char == 125 or char == 41: # Closings.
            next_ty = 5; new_token = True; in_num_literal = False; noun = True
        elif char==47 and char1==47: # // comments.
            next_ty = 0; single_comment = True; in_num_literal = False
            if current_ty !=0:
                new_token = True
        elif char == 47 and char1 == 42: # /* */ comments.
            next_ty = 0; multi_comment = True; in_num_literal = False
            if current_ty !=0:
                new_token = True
        elif char == 35: # The # symbol.
            in_num_literal = False
            next_ty = 7
            new_token = True
        else: # Symbols.
            next_ty = 1
            if current_ty==7: # The hash prepend maintains this token as a macro.
                next_ty = 7
            elif in_num_literal:
                next_ty = 3
            noun = not is_op
            if in_num_literal and char == 45 and (char0>=48 and char0<=57): # corner case of 12-34 at the -
                new_token = True; next_ty = 1;# do not change current_ty = 1
                in_num_literal = False; noun = True
            elif in_num_literal and (char == 46 or char0 == 46 or (minus_prepend and char0==45)): #1.23 and -123 continue numbers.
                noun = True
            elif (is_op or last_char_is_op) or current_ty != next_ty:
                new_token = True
            elif char==46 and char1==46 and i<n and c_txt_ints[i+2]==46: # ... beginning.
                new_token = True
            elif i>3 and c_txt_ints[i-1] == 46 and c_txt_ints[i-2] == 46 and c_txt_ints[i-3] == 46: #... end
                new_token = True
            next_in_alphanum_symbol = not is_op
        if (new_token or i==n) and current_ty>-1:
            out[0,k] = tok_start_ix; out[1,k] = i; out[2,k] = current_ty
            if current_ty==1: # Boolean literals.
                x = c_txt_ints
                if i>=5 and x[i-5]==102 and x[i-4]==97 and x[i-3]== 108 and x[i-2]==115 and x[i-1]== 101:
                    out[2,k] = 3 # false
                elif i>=4 and x[i-4]==116 and x[i-3]== 114 and x[i-2]== 117 and x[i-1]== 101:
                    out[2,k] = 3 #true
            tok_start_ix = i; k = k+1
        new_token = False
        last_char_is_op = is_op
        in_alphanum_symbol = next_in_alphanum_symbol
        if next_ty != 3:
            minus_prepend = False
        if next_ty>-1:
            current_ty = next_ty
    return out[:,0:k]

@numba.njit
def _angle_brackets(c_txt_ints, tokens_3xn):
    # When are angle brackets nesting?
    # Modifies tokens_3xn in place.
    n = tokens_3xn.shape[1]
    angle_scores = np.zeros(n,dtype=np.int64)
    break_scores = np.zeros(n,dtype=np.int64)
    for i in range(n):
        ix0 = tokens_3xn[0,i]; ix1 = tokens_3xn[1,i]
        if ix1==ix0+1: # One char.
            char = c_txt_ints[tokens_3xn[0,i]]
            if char==60: # <
                angle_scores[i] = 1
            elif char==62: # >
                angle_scores[i] = -1
            elif char== 40 or char==41 or char==91 or char==93 or char==123 or char==125: # ()[]{}
                break_scores[i] = 1
            elif char==59 or char==61: # ;= which also break up statements.
                break_scores[i] = 1
    i = 0; next_break_ix = -1 # ; or {}[]().
    for i in range(n-1):
        if i>=next_break_ix:
            j = i+1
            while j<n:
                next_break_ix = j
                if break_scores[j]>0:
                    break
                j = j+1
        if angle_scores[i]==1: # Find the matching closing.
            closing_bracket = -1 # Can we balance the < with a >.
            lev = 0
            for j in range(i, next_break_ix+1):
                lev = lev+angle_scores[j]
                if lev==0:
                    closing_bracket = j
                    break
            if closing_bracket>0: # ALL <> inclusive between open and close.
                for j in range(i, closing_bracket+1):
                    if angle_scores[j] > 0:
                        tokens_3xn[2,j] = 4
                    elif angle_scores[j] < 0:
                        tokens_3xn[2,j] = 5

# Fun numba error in a fn that isn't really numbable:
# The use of a reflected set(unicode_type) type, assigned to variable 'control_flow_set'
#in globals, is not supported as globals are considered compile-time constants and
#there is no known way to compile a reflected set(unicode_type) type as a constant.
control_flow_set = set(['break', 'case', 'catch', 'continue','do','else','for','goto','if','return','switch','throw','try','while'])
#@numba.njit
def _control_flow_mark(c_txt, tokens_3xn):
    # In place modification.
    symbols = np.nonzero(tokens_3xn[2,:]==1)[0]
    for sym_ix in symbols:
        if c_txt[tokens_3xn[0,sym_ix]:tokens_3xn[1,sym_ix]] in control_flow_set:
            tokens_3xn[2,sym_ix] = 2

@numba.njit
def _op_combine(c_txt_ints, tokens_3xn):
    # The following two-char ops are actually one op:
    # ++ -- += -= *= /= %= == != >= <= && || << >> ->
    # Returns a maybe shorter tokens_3xn.
    # Run after the _angle_brackers
    n = tokens_3xn.shape[1]
    combine_with_last = np.zeros(n, dtype=np.int64)
    busy = False
    for i in range(1,n):
        if busy:
            busy = False; continue
        t0 = tokens_3xn[0,i]; t1 = tokens_3xn[1,i]; ty = tokens_3xn[2,i]
        if t1 - t0 == 1 and ty == 1: # length 1 symbol.
            c0 = c_txt_ints[t0-1]; c1 = c_txt_ints[t0]
            is_op = c1==33 or c1== 37 or c1== 38 or (c1 >=42 and c1 <=47 and c1 != 44)\
                      or c1>=60 and c1<=63 or c1 == 94 or c1 == 124 or c1 == 126
            if not is_op:
                busy = False
            elif c1==c0 and c1 !=42: # ++ -- << etc. but not **
                busy = True
            elif c1==61: # +-*/?<>% and then =.
                if c0==43 or c0==45 or c0==42 or c0==47 or c0==63 or c0==60 or c0==62 or c0 == 37:
                    busy = True
            elif c1==62 and c0==45: #->
                busy = True # Prevent the next token from combining with this one.
            if busy:
                combine_with_last[i] = 1
    tokens_3xn1 = np.zeros((3,n-np.sum(combine_with_last)), dtype=np.int64)
    ix = -1
    for i in range(n):
        if combine_with_last[i]:
            tokens_3xn1[0,ix] = tokens_3xn[0,i-1] # Overwrites previous uses of ix.
            tokens_3xn1[1,ix] = tokens_3xn[1,i]
            tokens_3xn1[2,ix] = tokens_3xn[2,i] # Counts as a symbol and should be 1.
        else:
            ix = ix+1
            tokens_3xn1[:,ix] = tokens_3xn[:,i] # simple replace

    return tokens_3xn1

@numba.njit
def utf8_shift(c_txt_ints, tokens_3xn):
    # Unicode is variable width.
    out = np.zeros_like(tokens_3xn)
    shift = 0; k = tokens_3xn.shape[1]
    for i in range(k):
        ix0 = tokens_3xn[0,i]; ix1 = tokens_3xn[1,i]
        out[0,i] = tokens_3xn[0,i]-shift
        for j in range(ix0,ix1):
            #https://blog.birost.com/a?ID=01700-d0f09357-6df5-45a1-b7fd-644ec2136641
            x = c_txt_ints[j]
            if x<192: # All non-first chars in a multichar sequence have
                extra_bytes = 0
            elif x<224:
                extra_bytes = 1
            elif x<240:
                extra_bytes = 2
            elif x<248:
                extra_bytes = 3
            elif x<252:
                extra_bytes = 4
            else:
                extra_bytes = 5
            shift = shift+extra_bytes
        out[1,i] = tokens_3xn[1,i]-shift
        out[2,i] = tokens_3xn[2,i]
    return out

def token_mark(c_txt, mark_control_flow=True):
    #Returns an 3xn numpy int array of [token_start_ix, token_end_ix (slice end), token_type]
    # Token types are:
    # 0 = space/comments.
    # 1 = symbols. Includes *+= etc. foo::bar counts as one symbol. Includes & and * ref and deref.
    # 2 (OPTIONAL, can be a symbol instead) Control-flow like if (as a convention we do not have this in lisp languages since the control flow is more like a function call)
    # 3 Literals (numbers, strings, but not arrays/lists/etc).
    # 4 = opening brackets. Not quotes in strings.
    # 5 = closing brackets. SOMETIMS <> count.
    # 6 = Punctuation, which is , or ;, or : ending public: in C++. Not the :: though in C++ that is one symbol.
    # 7 = #define #include #undef #if, #elif, #else, #endif, #ifdef and #ifndef (any others?) Note: Space is allowed between the # and stuff.
    # 8 = Decorator symbols (C++ does not have explicit symbols so unused, but python's @ would he here.)
    buffer_len = 4
    #np.frombuffer( 'foo'.encode('utf8'),dtype=np.uint8)
    c_txt_ints = np.frombuffer((c_txt+(' '*buffer_len)).encode('utf8'),dtype=np.uint8)
    tokens_3xn = _token_core(c_txt_ints, buffer_len)
    _angle_brackets(c_txt_ints, tokens_3xn)
    if mark_control_flow:
        _control_flow_mark(c_txt, tokens_3xn)
    tokens_3xn = _op_combine(c_txt_ints, tokens_3xn)
    tokens_3xn = utf8_shift(c_txt_ints, tokens_3xn) # Must put at end.

    return tokens_3xn

def tokenize(c_txt, remove_space=False, mark_control_flow=True):
    tokens_3xn = token_mark(c_txt, mark_control_flow=mark_control_flow)
    tokens = []; token_tys = []
    for i in range(tokens_3xn.shape[1]):
        ty = tokens_3xn[2,i]
        if ty != 0 or not mark_control_flow:
            tokens.append(c_txt[tokens_3xn[0,i]:tokens_3xn[1,i]])
            token_tys.append(ty)
    return tokens, token_tys

# @numba.njit
def paren_match_np(c_txt, tokenmark_3xn):
    # Match the opening with the closing.
    TODO

def decorators(tokens, token_types):
    # Returns what_we_decorate, what_decorates_us
    # Decoration cases to handle:
      # int x = 5, y = 6, z = 50;
      # vector<vector<int>> vect = foo;
      # typedef int int_t, *intp_t, (&fp)(int, ulong), arr_t[10];
      # typedef struct {int a; int b;} S, *pS;
      #for (int i = 0, j = 10; i < 10 && j > 0; i++, j--)
      # using namespace ContosoData;
      # template <class myType> myType GetMax(...){}
      # #define WARN_IF(EXP) ...
    n = len(tokens)
    linesplit_set = {'=','+=','-=','*=','/=','--','++','<','>','<=','>=','||','<<','>>','=='} # Not 100% sure here. No && since that's a double ref.

    # Semicolons deluxe (except for macros for some reason but those are easy):
    # Semicolons end statements and thus decorations
    # =, *=, ==, etc stops decoration untill we get , or ;. Only LHS can get decorated or decorate.
    # Each statement is broken up into comma-seperated pieces.
      # All but the last piece of the first , decorate everything else.
      # Unless inside ()[] in which commas are hard-stops.
    # [](){} insides do not decorate and we do those recursivly.
    # <> is recursive but we do decorate.
    token_types = list(token_types)+[6]
    tokens = list(tokens)+[';']
    token_types = np.asarray(token_types, dtype=np.int64)
    n = token_types.size; deep = max_depth(token_types)
    decors = [[] for _ in range(deep+1)] # level to decorators.
    onLHS = [True for _ in range(deep+1)] # nested eq are possible.
    LHSsymbols = [[[]] for _ in range(deep+1)] # [level] [commas_sep] = ix. LHS only.
    angle_open_ixs = [-1 for _ in range(deep+1)]
    angle_close_ixs = [-1 for _ in range(deep+1)]
    round_open = [False for _ in range(deep+1)]
    namespace = [False for _ in range(deep+1)]

    targets = [[] for _ in range(n)]
    shooters = [[] for _ in range(n)]
    isLHS = [];
    for_loop = -1
    macro_shadow = -1 # 1:1 decoration.
    just_closed = False
    curley_lev = 0; simple_lev = 0; angle_lev = 0;
    add_nested_angles = True
    # See: https://stackoverflow.com/questions/6352723/when-is-a-semicolon-after-mandated-in-c-c
    semicolon_after_all_brace = False
    n = len(tokens)
    for i in range(n):
        #print('Macro shadow:', macro_shadow, 'at:',tokens[i])
        tok = tokens[i]; ty = token_types[i]
        lev = curley_lev + simple_lev + angle_lev # Does not include angle!
        open = (ty==4)# and tok != '<')
        close = (ty==5)# and tok != '>')
        # For loop with multible vars: for (int i = 0, j = 10; i < 10 && j > 0; i++, j--)
        comma_ok = (simple_lev + angle_lev)==0 or for_loop>-1
        if ty==4 and tok=='<':
            angle_open_ixs[lev] = i
        if ty==5 and tok=='>':
            angle_close_ixs[lev-1] = i
        if ty==4 and tok=='(':
            round_open[lev] = True # Indicates that } ends the line, as in int foo(){}
        if tok=='namespace': # Namespaces don't need a ; after } so we have to mark them twice.
            namespace[lev] = True

        if tok==';' or tok==':' or (not comma_ok and tok==',') or close: # End of line which includes closings.
            one_two = 1
            if semicolon_after_all_brace and close and tok=='}': # Have a virtual ; after each }
                one_two = 2
            elif lev>0 and round_open[lev-1] and tok == '}': # int foo(){} means double-close the final }
                one_two = 2
            elif lev>0 and namespace[lev-1]: # Namespaces.
                one_two = 2
            round_open[lev] = False # Note: for ( () ) the second ) only stops round_open in the first ).
            #print('Onetwo:',one_two, 'at:',i, tok)
            for j in range(one_two):
                symss = LHSsymbols[lev] # Comma-seperated.
                #print('Syms are:', symss, 'at:', i, tokens[i])
                if len(symss)>0 and len(symss[0])>1:
                    #print('**filling**', i, tokens[i])
                    origins = symss[0][0:-1]
                    destss = [[symss[0][-1]]]+symss[1:]
                    for dests in destss:
                        for _d in dests:
                            for o in origins:
                                targets[o].append(_d)
                                shooters[_d].append(o)
                    if add_nested_angles: #and angle_lev==0:
                        lowest_origin = np.min(origins)
                        if lowest_origin < angle_open_ixs[lev]:
                            for j in range(angle_open_ixs[lev], angle_close_ixs[lev]+1):
                                for dests in destss:
                                    for _d in dests:
                                        if j<_d and token_types[j] != 0:
                                            targets[j].append(_d)
                                            shooters[_d].append(j)
                onLHS[lev] = True; for_loop = -1
                LHSsymbols[lev] = [[]]
                if close:
                    just_closed = True
                else:
                    just_closed = False
                lev = lev-1
            if ty== 5: #Closings.
                namespace[lev-1] = False
                if tok==')' or tok==']':
                    simple_lev = simple_lev-1
                elif tok =='}':
                    curley_lev = curley_lev-1
                elif tok =='>':
                    macro_shadow = -1 # Special #include <foo>
                    angle_lev = angle_lev-1
        elif tok=='for':
            for_loop = 1
        elif open: # Openings.
            if tok=='(' or tok == '[':
                simple_lev = simple_lev+1
            elif tok=='{':
                curley_lev = curley_lev+1
            elif tok=='<':
                angle_lev = angle_lev+1
        elif tok==',': #and onLHS[lev]: # Commasep (note: comma_ok will be True which means don't include the ,)
            LHSsymbols[lev].append([])
            onLHS[lev] = True
        #elif tok==',': # Commasep without being on LHS.
        #    onLHS[lev] = True
        elif tok in linesplit_set and ty==1: # Stop the LHS.
            onLHS[lev] = False
        elif (ty==1 or ty==3) and macro_shadow>-1: # Macros decrorate the next token.
            targets[macro_shadow].append(i); shooters[i].append(macro_shadow)
            if i>0 and tokens[i-1]=='<': # Special #include <>.
                for u in [i-1,i+1]:
                    if u<n-1 and i<n-1:
                        targets[u].append(i); shooters[i].append(u)
                if tokens[macro_shadow] != '#include':
                    raise Exception('Non-include uses <> like an #include would. Need to address this syntax.')
            else:
                macro_shadow = -1
        elif ty==7: # Macros.
            macro_shadow = i
        elif macro_shadow==-1 and (ty==1 or (tok=='<' and ty==4) or (tok=='>' and ty==5)) and onLHS[lev]: # Add in the symbol.
            if tok[0] != '-' and tok[0] != '+' and tok[0] != '/' and tok[0] != '|':
                LHSsymbols[lev][-1].append(i)
            for_loop = -1
        isLHS.append(onLHS[lev]) # isLHS is not that important.
    return targets[0:n-1], shooters[0:n-1], isLHS[0:n-1]
