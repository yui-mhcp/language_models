# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import re
import ast
import sys
import inspect
import builtins
import traceback
import importlib

from functools import partial
from contextlib import redirect_stdout, redirect_stderr

_orphan_re  = r'\n?[\s\S] → None\n'

_dangerous_builtins = {
    'exec', 'eval', 'compile', 'open', 'input', 
    'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr',
    'breakpoint', 'memoryview', 'dir'
}
_dangerous_modules  = {
    'os', 'sys', 'subprocess', 'shutil', 'pathlib', 'io', 'socket',
    'pickle', 'shelve', 'dbm', 'sqlite3', 'pdb', 'pty', 'platform',
    'ctypes', 'multiprocessing', 'threading', 'asyncio', 'concurrent',
    '__main__', '__builtin__', 'builtins', 'posix'
}

_safe_modules   = {
    'math', 'numpy', 'pandas', 'random', 'datetime', 'collections', 're', 'json',
    'itertools', 'functools', 'operator', 'string', 'copy',
    'bisect', 'heapq', 'array', 'enum', 'numbers', 'fractions',
    'statistics', 'decimal', 'typing'
}

def extract_code(text):
    """
        Extract text and python code parts from `text`
        
        Arguments :
            - text  : (str), the text to extract code from
        Return :
            - raw_text  : text that is not a code block
            - codes     : a list of all code blocks detected
        
        A code block is delimited by :
        ```python
        [... code ...]
        ```
    """
    parts = re.split(r'(```[a-z]*)', text)
    
    i, texts, codes = 0, [], []
    while i < len(parts):
        part = parts[i]
        if part in ('```', '```python') and i + 2 < len(parts) and parts[i + 2] == '```':
            codes.append(parts[i + 1])
            i += 3
        else:
            texts.append(part)
            i += 1
    
    return ''.join(texts), codes

def remove_simulated_output(text):
    lines = []
    for line in text.split('\n'):
        if '#' in line and line.strip().startswith('print('):
            line, _, _ = line.rpartition('#')
        
        lines.append(line)
    
    return '\n'.join(lines)
    
def execute_code(code,
                 tools  = None,
                 *,
                 
                 globals_dict   = None,
                 add_traceback  = False,
                 allowed_modules = _safe_modules,
                 
                 ** kwargs
                ):
    try:
        parsed_ast = ast.parse(code)
        
        security_visitor = SecurityVisitor(allowed_modules)
        security_visitor.visit(parsed_ast)
        
        if security_visitor.issues:
            return {
                'variables' : [],
                'stdout'    : '',
                'stderr'    : "⚠️ Unsafe code detected ⚠️\n{}".format(security_visitor)
            }
        
        code_lines = code.splitlines()
        orphan_visitor = FunctionCallVisitor(code_lines)
        orphan_visitor.visit(parsed_ast)
    except SyntaxError as e:
        return {'variables' : {}, 'stdout' : '', 'stderr' : str(e)}

    locals_dict     = {}
    safe_globals    = create_safe_globals(globals_dict)
    
    if tools: safe_globals.update({tool.name : partial(tool, ** kwargs) for tool in tools})
    
    stdout_buffer   = io.StringIO()
    stderr_buffer   = io.StringIO()

    tool_names  = [tool.name for tool in tools]
    # Exécution de chaque bloc
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        for block in parsed_ast.body:
            try:
                code = '\n'.join([
                    line.rstrip() for line in code_lines[block.lineno - 1 : block.end_lineno]
                ]).strip()
                if isinstance(block, ast.FunctionDef) and block.name in tool_names:
                    continue
                elif block in orphan_visitor:
                    compiled_block = """
                    _tmp = [code]\nif _tmp is not None: print(f'[code] → {_tmp}')
                    """.replace('[code]', orphan_visitor[block].replace("'", r"\'")).strip()
                else:
                    compiled_block = compile(
                        ast.Module(body = [block], type_ignores = []),
                        filename    = "<string>", 
                        mode    = "exec"
                    )
                
                if code.startswith('print(') and ('format' not in code or code[7] != 'f'):
                    print(code.strip() + ' # ', end = '')
                exec(compiled_block, safe_globals, locals_dict)
                
                safe_globals.update({
                    k : locals_dict.pop(k) for k, v in list(locals_dict.items()) if inspect.ismodule(v)
                })
                
            except Exception as e:
                print('{} : {}'.format(e.__class__.__name__, e), file = stderr_buffer)
                if add_traceback: traceback.print_exc(file = stderr_buffer)
                #raise e
    
    cleaned_vars = {
        k: v for k, v in locals_dict.items() 
        if not k.startswith('_') and not callable(v) and not isinstance(v, type) and v is not None
    }
    
    return {
        "variables" : cleaned_vars,
        "stdout"    : stdout_buffer.getvalue().strip(),
        "stderr"    : stderr_buffer.getvalue()
    }

def format_code_result(result):
    if not any(v for v in result.values()): return ''
    
    output = ''
    if result['variables'] and not result['stdout']:
        output += 'Variables :\n' + '\n'.join([
            '- {} : {}'.format(k, v) for k, v in result['variables'].items()
        ]) + '\n\n'
    
    if result['stdout']:
        output += 'Stdout :\n```bash\n{}\n```'.format(result['stdout'])
        if result['stderr']: output += '\n\n'
    
    if result['stderr']:
        output += 'Stderr :\n```bash\n{}\n```'.format(result['stderr'])
    
    return output



class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self, code_lines):
        self.code_lines = code_lines
        
        self.orphan_nodes = {}
    
    def __contains__(self, node):
        return node in self.orphan_nodes
    
    def __getitem__(self, node):
        return self.orphan_nodes[node]
    
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.col_offset == 0:
                func_name = node.value.func.id
            elif isinstance(node.value.func, ast.Attribute):
                func_name = f"{self.get_attribute_source(node.value.func)}.{node.value.func.attr}"
            else:
                func_name = "[fonction anonyme]"
            
            if func_name != 'print':
                self.orphan_nodes[node] = '\n'.join([
                    line.rstrip() for line in self.code_lines[node.lineno - 1 : node.end_lineno]
                ]).strip()
        
        self.generic_visit(node)
    
    def get_attribute_source(self, node):
        """Obtient la source complète d'un attribut (ex: module.submodule)"""
        if isinstance(node.value, ast.Name):
            return node.value.id
        elif isinstance(node.value, ast.Attribute):
            return f"{self.get_attribute_source(node.value)}.{node.value.attr}"
        return "???"

class SecurityVisitor(ast.NodeVisitor):
    def __init__(self, allowed_modules = None):
        self.allowed_modules = allowed_modules or []
        
        self.issues = []
    
    def __str__(self):
        if not self.issues: return "The code is safe !"
        return "\n".join(['- ' + issue for issue in self.issues])
    
    def _check_import(self, name):
        if self.allowed_modules:
            if name in self.allowed_modules: return ''
            return 'The module {} is not authorized'.format(name)
        elif name in _dangerous_modules:
            return 'The module {} is dangerous'.format(name)
        else:
            return ''
    
    def visit_Import(self, node):
        for name in node.names:
            module = name.name.split('.')[0]
            error  = self._check_import(module)
            if error: self.issues.append(error)
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        module = node.module.split('.')[0] if node.module else ''
        error  = self._check_import(module)
        if error: self.issues.append(error)
        
        self.generic_visit(node)


def create_safe_globals(user_globals = None, import_safe_modules = True):
    safe_globals    = {
        '__builtins__'  : {
            k : getattr(builtins, k) for k in dir(builtins) if k not in _dangerous_builtins
        }
    }
    if import_safe_modules:
        safe_globals.update({
            name : importlib.import_module(name) for name in _safe_modules
        })
    
    if user_globals:
        safe_globals.update({
            k : v for k, v in user_globals.items()
            if k not in _dangerous_builtins
        })
    
    return safe_globals
