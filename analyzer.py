# analyzer.py
import ast
import sys
import inspect
import logging
import os
import time
from functools import wraps
from typing import Dict, List, Any, Optional, Callable
from rich.console import Console
from rich.traceback import install
from rich.logging import RichHandler
from rich.table import Table
from datetime import datetime

# Install rich traceback handler
install(show_locals=True)

# Configure rich logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

log = logging.getLogger("code_analyzer")
console = Console()

class RuntimeStats:
    """Class to track runtime statistics"""
    def __init__(self):
        self.function_calls = {}
        self.execution_times = {}
        self.print_calls = []
        self.errors = []

    def record_function_call(self, func_name: str, args: tuple, kwargs: dict) -> None:
        if func_name not in self.function_calls:
            self.function_calls[func_name] = []
        self.function_calls[func_name].append({
            'timestamp': datetime.now(),
            'args': args,
            'kwargs': kwargs
        })

    def record_execution_time(self, func_name: str, execution_time: float) -> None:
        if func_name not in self.execution_times:
            self.execution_times[func_name] = []
        self.execution_times[func_name].append(execution_time)

    def record_print(self, content: str, line_no: int) -> None:
        self.print_calls.append({
            'timestamp': datetime.now(),
            'content': content,
            'line_no': line_no
        })

    def record_error(self, error: Exception, func_name: str) -> None:
        self.errors.append({
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'message': str(error),
            'function': func_name
        })

runtime_stats = RuntimeStats()

def runtime_logger(func: Callable) -> Callable:
    """Decorator to log runtime behavior of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        
        log.info(f"[bold blue]Entering function:[/] {func_name}")
        runtime_stats.record_function_call(func_name, args, kwargs)
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            runtime_stats.record_execution_time(func_name, execution_time)
            
            log.info(f"[bold green]Exiting function:[/] {func_name} "
                    f"(took {execution_time:.4f}s)")
            return result
            
        except Exception as e:
            runtime_stats.record_error(e, func_name)
            log.error(f"[bold red]Error in function {func_name}:[/] {str(e)}")
            raise
            
    return wrapper

class CodeAnalyzer(ast.NodeVisitor):
    """Analyzes Python source code for various metrics and patterns"""
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.print_calls: List[Dict[str, Any]] = []
        self.variables: Dict[str, List[str]] = {}
        self.complexity_scores: Dict[str, int] = {}
        self.current_function: Optional[str] = None

    def _has_return(self, node: ast.AST) -> bool:
        """Check if a node contains any return statements"""
        return any(isinstance(n, ast.Return) for n in ast.walk(node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyze function definitions and inject runtime logging"""
        self.current_function = node.name
        self.functions[node.name] = {
            'line_number': node.lineno,
            'args': [arg.arg for arg in node.args.args],
            'returns': self._has_return(node),
            'docstring': ast.get_docstring(node)
        }
        
        # Calculate complexity
        self.complexity_scores[node.name] = self._calculate_complexity(node)
        
        log.info(f"[bold green]Found function:[/] {node.name}")
        self.generic_visit(node)
        self.current_function = None

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Assert,
                                ast.Try, ast.ExceptHandler)):
                complexity += 1
        return complexity

    def visit_Call(self, node: ast.Call) -> None:
        """Analyze function calls, including print statements"""
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            self._record_print(node)
        self.generic_visit(node)

    def _record_print(self, node: ast.AST) -> None:
        """Record information about print statements"""
        self.print_calls.append({
            'line': node.lineno,
            'function': self.current_function,
            'type': 'print'
        })
        log.debug(f"[yellow]Print statement found[/] at line {node.lineno}")

    def visit_Name(self, node: ast.Name) -> None:
        """Analyze variable usage"""
        if isinstance(node.ctx, ast.Store):
            if self.current_function not in self.variables:
                self.variables[self.current_function] = []
            self.variables[self.current_function].append(node.id)
        self.generic_visit(node)

    def generate_runtime_injection_code(self, source_code: str) -> str:
        """Generate modified source code with runtime logging"""
        tree = ast.parse(source_code)
        
        # Add imports
        runtime_imports = (
            "from analyzer import runtime_logger, runtime_stats\n"
            "import inspect\n\n"
        )
        
        # Add runtime logging decorator to functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.decorator_list.append(
                    ast.Name(id='runtime_logger', ctx=ast.Load())
                )
        
        # Modify print statements for tracking
        class PrintTransformer(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    return ast.Call(
                        func=node.func,
                        args=[
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='runtime_stats', ctx=ast.Load()),
                                    attr='record_print',
                                    ctx=ast.Load()
                                ),
                                args=[
                                    ast.Constant(value=str([ast.unparse(arg) for arg in node.args])),
                                    ast.Constant(value=node.lineno)
                                ],
                                keywords=[]
                            ),
                            *node.args
                        ],
                        keywords=node.keywords
                    )
                return node
        
        # Transform the AST
        transformed = PrintTransformer().visit(tree)
        ast.fix_missing_locations(transformed)
        
        return runtime_imports + ast.unparse(transformed)

    def generate_report(self) -> None:
        """Generate comprehensive analysis report"""
        console.print("\n[bold blue]Code Analysis Report[/]", style="bold")
        
        # Function Analysis Table
        function_table = Table(title="Function Analysis")
        function_table.add_column("Function Name", style="cyan")
        function_table.add_column("Line Number")
        function_table.add_column("Arguments")
        function_table.add_column("Returns", justify="center")
        function_table.add_column("Complexity Score")
        function_table.add_column("Has Docstring", justify="center")
        
        for fname, finfo in self.functions.items():
            function_table.add_row(
                fname,
                str(finfo['line_number']),
                ", ".join(finfo['args']) or "None",
                "✓" if finfo['returns'] else "✗",
                str(self.complexity_scores.get(fname, 'N/A')),
                "✓" if finfo['docstring'] else "✗"
            )
        
        console.print(function_table)
        
        # Print Statements Analysis
        if self.print_calls:
            print_table = Table(title="Print Statements Analysis")
            print_table.add_column("Line")
            print_table.add_column("Function")
            
            for print_call in sorted(self.print_calls, key=lambda x: x['line']):
                print_table.add_row(
                    str(print_call['line']),
                    print_call['function'] or 'global'
                )
            
            console.print("\n", print_table)
        
        # Variable Analysis
        if self.variables:
            var_table = Table(title="Variable Analysis")
            var_table.add_column("Scope")
            var_table.add_column("Variables")
            
            for scope, vars_list in self.variables.items():
                var_table.add_row(
                    scope or 'global',
                    ", ".join(sorted(set(vars_list)))
                )
            
            console.print("\n", var_table)

def analyze_file(file_path: str, inject_runtime_logging: bool = True) -> None:
    """Analyze a Python source file and optionally inject runtime logging"""
    try:
        with open(file_path, 'r') as file:
            source = file.read()
        
        log.info(f"[bold blue]Analyzing file:[/] {file_path}")
        
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            log.error(f"[bold red]Syntax error in file:[/] {e}")
            return

        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        
        if inject_runtime_logging:
            # Generate modified source with runtime logging
            modified_source = analyzer.generate_runtime_injection_code(source)
            
            # Save modified source to a new file
            output_path = file_path.replace('.py', '_analyzed.py')
            with open(output_path, 'w') as f:
                f.write(modified_source)
            
            log.info(f"[bold green]Generated analyzed version:[/] {output_path}")
        
        analyzer.generate_report()

    except Exception as e:
        log.error(f"[bold red]Error analyzing file:[/] {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        console.print("[bold red]Usage: python analyzer.py <python_file>[/]")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        console.print(f"[bold red]File not found:[/] {file_path}")
        sys.exit(1)

    analyze_file(file_path)