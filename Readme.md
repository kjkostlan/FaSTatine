# Fast FSM-based Tokenizer for common languages.

(sorry, this is not a AST tool, it does not generate trees, the name may be confusing. That being said tokenization via FSM is the first step in AST generation).

*This is very much a work in progress!*

Most of the time code manipulation is performed on an Abstract Syntax Tree. This involves converting a language into a tree, modifying said tree, and using the resulting code (may or may not convert back).

However this loses "look and feel" information about the text: such as whats in the comments, where spaces are added, etc. For simple text-based manipulations to a code it is helpful to have a tokenizer to determine what is in a string/comment/etc.

FaSTatine uses numba for performance and is based on finite-state-machine (FSM) parsers. FSM parsers run through the code character by character and keep track of whether we currently in a quote, comment, escaping a string, what indent level we are, etc. The output is an np-array of indent levels and token types (see legend.txt).
