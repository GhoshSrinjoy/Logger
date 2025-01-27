# rich_logger.py
import logging
import time
import inspect
import sys
import importlib.util
from typing import Any, Optional, Dict
from functools import wraps
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from rich.tree import Tree
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from datetime import datetime
from contextlib import contextmanager
from types import ModuleType

# Install rich traceback handler
install(show_locals=True)

class RichLogger:
    """Advanced logger using Rich for beautiful console output"""
    def __init__(self, name: str = "RichLogger"):
        self.console = Console()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(
                rich_tracebacks=True,
                console=self.console,
                show_time=True,
                show_path=True
            )]
        )
        
        self.log = logging.getLogger(name)
        self.execution_tree = Tree("📊 Execution Flow")
        self.start_time = time.time()
        self.current_progress = None
        self.function_stats: Dict[str, Dict[str, Any]] = {}
    
    def start_logging(self) -> None:
        """Start the logging session"""
        self.console.print("\n[bold blue]====== Starting Execution Logging ======[/]")
        self.console.print(Panel(
            f"[yellow]Started at:[/] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"[yellow]Python version:[/] {sys.version.split()[0]}\n"
            f"[yellow]Platform:[/] {sys.platform}",
            title="Session Info",
            border_style="blue"
        ))
    
    def end_logging(self) -> None:
        """End the logging session and display summary"""
        execution_time = time.time() - self.start_time
        
        # Create execution summary table
        summary_table = Table(title="Execution Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Execution Time", f"{execution_time:.2f} seconds")
        summary_table.add_row("Functions Executed", str(len(self.function_stats)))
        
        if self.function_stats:
            longest_func = max(self.function_stats.items(), 
                             key=lambda x: x[1]['total_time'])[0]
            summary_table.add_row("Longest Running Function", 
                                f"{longest_func} ({self.function_stats[longest_func]['total_time']:.2f}s)")
        
        self.console.print("\n")
        self.console.print(summary_table)
        self.console.print("\n[bold blue]====== Logging Session Ended ======[/]")
    
    @contextmanager
    def log_step(self, description: str) -> None:
        """Context manager for logging execution steps"""
        start_time = time.time()
        self.log.info(f"[bold cyan]►[/] Starting: {description}")
        
        try:
            yield
            duration = time.time() - start_time
            self.log.info(f"[bold green]✓[/] Completed: {description} [dim]({duration:.2f}s)[/]")
        except Exception as e:
            self.log.error(f"[bold red]✗[/] Failed: {description}\n  Error: {str(e)}")
            raise

    def wrap_function(self, func):
        """Wrap a function with logging"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Initialize function stats if not exists
            if func_name not in self.function_stats:
                self.function_stats[func_name] = {
                    'calls': 0,
                    'total_time': 0,
                    'last_args': None,
                    'errors': 0
                }
            
            start_time = time.time()
            
            # Create progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Running {func_name}", total=None)
                
                try:
                    # Log function entry
                    self.log.info(f"[bold cyan]→[/] Entering function: {func_name}")
                    if args or kwargs:
                        self.log.debug(f"  Args: {args}\n  Kwargs: {kwargs}")
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Update stats
                    duration = time.time() - start_time
                    self.function_stats[func_name]['calls'] += 1
                    self.function_stats[func_name]['total_time'] += duration
                    self.function_stats[func_name]['last_args'] = (args, kwargs)
                    
                    # Log success
                    self.log.info(
                        f"[bold green]←[/] Exited function: {func_name} "
                        f"[dim]({duration:.2f}s)[/]"
                    )
                    
                    return result
                    
                except Exception as e:
                    # Update error stats
                    self.function_stats[func_name]['errors'] += 1
                    
                    # Log error
                    self.log.error(
                        f"[bold red]⚠[/] Error in {func_name}: {str(e)}"
                    )
                    raise
                finally:
                    progress.update(task, completed=100)
        
        return wrapper

    def wrap_module_functions(self, module: ModuleType):
        """Wrap all functions in a module with logging"""
        for name, obj in inspect.getmembers(module):
            # Only wrap functions/methods defined in the module (not imported ones)
            if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                setattr(module, name, self.wrap_function(obj))
            elif inspect.isclass(obj) and obj.__module__ == module.__name__:
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    if not method_name.startswith('__'):
                        setattr(obj, method_name, self.wrap_function(method))

    def run_script(self, script_path: str):
        """Run a Python script with logging"""
        try:
            with self.log_step(f"Loading script: {script_path}"):
                # Load the script as a module
                spec = importlib.util.spec_from_file_location("monitored_script", script_path)
                module = importlib.util.module_from_spec(spec)
                
                # Wrap the module's functions with logging
                self.wrap_module_functions(module)
                
                # Execute the script
                with self.log_step("Executing script"):
                    spec.loader.exec_module(module)
                    
                    # If script has main(), call it
                    if hasattr(module, 'main'):
                        module.main()
            
            # Show function statistics
            self.get_function_stats()
            
        except Exception as e:
            self.log_error(f"Script execution failed: {str(e)}")
            raise

    def log_info(self, message: str) -> None:
        """Log an info message"""
        self.log.info(f"[bold white]ℹ[/] {message}")
    
    def log_warning(self, message: str) -> None:
        """Log a warning message"""
        self.log.warning(f"[bold yellow]⚠[/] {message}")
    
    def log_error(self, message: str) -> None:
        """Log an error message"""
        self.log.error(f"[bold red]✗[/] {message}")
    
    def log_success(self, message: str) -> None:
        """Log a success message"""
        self.log.info(f"[bold green]✓[/] {message}")
    
    def log_debug(self, message: str) -> None:
        """Log a debug message"""
        self.log.debug(f"[bold blue]🔍[/] {message}")

    def get_function_stats(self) -> None:
        """Display function execution statistics"""
        if not self.function_stats:
            self.log_warning("No function statistics available")
            return
        
        stats_table = Table(title="Function Statistics")
        stats_table.add_column("Function", style="cyan")
        stats_table.add_column("Calls", style="green")
        stats_table.add_column("Total Time", style="yellow")
        stats_table.add_column("Avg Time", style="yellow")
        stats_table.add_column("Errors", style="red")
        
        for func_name, stats in self.function_stats.items():
            avg_time = stats['total_time'] / stats['calls'] if stats['calls'] > 0 else 0
            stats_table.add_row(
                func_name,
                str(stats['calls']),
                f"{stats['total_time']:.2f}s",
                f"{avg_time:.2f}s",
                str(stats['errors'])
            )
        
        self.console.print("\n")
        self.console.print(stats_table)

def main():
    if len(sys.argv) != 2:
        print("Usage: python rich_logger.py <script_to_run.py>")
        sys.exit(1)
    
    script_path = sys.argv[1]
    logger = RichLogger("ScriptMonitor")
    logger.start_logging()
    
    try:
        logger.run_script(script_path)
    finally:
        logger.end_logging()

if __name__ == "__main__":
    main()