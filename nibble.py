from pygments.lexer import RegexLexer, bygroups
from pygments.token import *

class CustomLexer(RegexLexer):
    name = 'Nibble'
    aliases = 'nibble'
    filenames = ['*.nb']
    tokens = {
        'root' : [
            (r'(/\*)([^*]|\*[^/])*(\*/)', Comment.Multiline),
            (r'\s+', Text),
            (r'//.*$', Comment.Single),
            (r'(let)(\s+)(\w+)', bygroups(Keyword, Text, Name.Function)),
            (r'match', Keyword),
            (r'\w+', Name),
            (r'"([^"\\]|\\.)*"', String),
            (r'[|.!]', Operator),
            (r'[();/=]', Punctuation)
        ]
    }
