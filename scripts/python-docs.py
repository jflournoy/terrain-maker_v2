#!/usr/bin/env python3
"""
Python API Documentation Generator

Extracts docstrings from Python modules and generates markdown documentation
for the API reference.

Usage:
    python scripts/python-docs.py          # Generate all API docs
    python scripts/python-docs.py validate # Validate existing docs
    python scripts/python-docs.py check    # Check for undocumented code
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DocstringExtractor:
    """Extract docstrings and signatures from Python files."""

    def __init__(self, module_path: str):
        """Initialize extractor for a module."""
        self.module_path = Path(module_path)
        self.functions: Dict[str, Dict] = {}
        self.classes: Dict[str, Dict] = {}
        self.module_docstring = ""

    def extract(self) -> bool:
        """Extract all docstrings from module. Returns True if successful."""
        if not self.module_path.exists():
            print(f"  ⚠️  File not found: {self.module_path}")
            return False

        try:
            with open(self.module_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            # Extract module docstring
            self.module_docstring = ast.get_docstring(tree) or ""

            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._extract_function(node)
                elif isinstance(node, ast.ClassDef):
                    self._extract_class(node)

            return True
        except Exception as e:
            print(f"  ✗ Error parsing {self.module_path}: {e}")
            return False

    def _extract_function(self, node: ast.FunctionDef) -> None:
        """Extract function signature and docstring."""
        # Skip private functions
        if node.name.startswith('_'):
            return

        sig = self._get_signature(node)
        docstring = ast.get_docstring(node) or "No documentation"

        self.functions[node.name] = {
            'signature': sig,
            'docstring': docstring,
            'lineno': node.lineno
        }

    def _extract_class(self, node: ast.ClassDef) -> None:
        """Extract class info and methods."""
        # Skip private classes
        if node.name.startswith('_'):
            return

        docstring = ast.get_docstring(node) or "No documentation"
        methods = {}

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                methods[item.name] = {
                    'signature': self._get_signature(item),
                    'docstring': ast.get_docstring(item) or "No documentation"
                }

        self.classes[node.name] = {
            'docstring': docstring,
            'methods': methods,
            'lineno': node.lineno
        }

    @staticmethod
    def _get_signature(node: ast.FunctionDef) -> str:
        """Generate function signature from AST node."""
        args = []

        # Regular arguments
        for arg in node.args.args:
            args.append(arg.arg)

        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # Keyword-only arguments
        for arg in node.args.kwonlyargs:
            args.append(arg.arg)

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        return f"{node.name}({', '.join(args)})"


def generate_markdown_docs(extractor: DocstringExtractor, module_name: str) -> str:
    """Generate markdown documentation from extracted docstrings."""
    lines = []

    # Module header
    lines.append(f"## {module_name}\n")

    # Module docstring
    if extractor.module_docstring:
        lines.append(f"{extractor.module_docstring}\n")

    # Classes
    if extractor.classes:
        lines.append("### Classes\n")
        for class_name in sorted(extractor.classes.keys()):
            class_info = extractor.classes[class_name]
            lines.append(f"#### `{class_name}`\n")
            lines.append(f"{class_info['docstring']}\n")

            # Class methods
            if class_info['methods']:
                lines.append("**Methods:**\n")
                for method_name in sorted(class_info['methods'].keys()):
                    method_info = class_info['methods'][method_name]
                    lines.append(
                        f"- `{method_info['signature']}` - "
                        f"{method_info['docstring'].split(chr(10))[0]}\n"
                    )
            lines.append("")

    # Functions
    if extractor.functions:
        lines.append("### Functions\n")
        for func_name in sorted(extractor.functions.keys()):
            func_info = extractor.functions[func_name]
            lines.append(f"#### `{func_info['signature']}`\n")
            lines.append(f"{func_info['docstring']}\n")

    return '\n'.join(lines)


def scan_python_modules() -> List[Tuple[str, Path]]:
    """Scan for Python modules to document."""
    src_path = Path('src/terrain')
    modules = []

    if src_path.exists():
        for py_file in sorted(src_path.glob('*.py')):
            if py_file.name != '__init__.py':
                module_name = f"terrain.{py_file.stem}"
                modules.append((module_name, py_file))

    return modules


def generate_all_docs() -> None:
    """Generate documentation for all modules."""
    print('📚 Generating Python API Documentation...\n')

    modules = scan_python_modules()

    if not modules:
        print('  ⚠️  No Python modules found in src/terrain/')
        return

    doc_sections = []
    total_functions = 0
    total_classes = 0

    for module_name, module_path in modules:
        print(f'  Extracting {module_name}...')
        extractor = DocstringExtractor(str(module_path))

        if extractor.extract():
            total_functions += len(extractor.functions)
            total_classes += len(extractor.classes)
            markdown = generate_markdown_docs(extractor, module_name)
            doc_sections.append(markdown)
            print(f'    ✓ Found {len(extractor.functions)} functions, {len(extractor.classes)} classes')

    if doc_sections:
        # Generate full API reference
        output_path = Path('docs/API_REFERENCE.md')
        full_content = (
            "# terrain-maker API Reference\n\n"
            "Comprehensive documentation of all classes and functions in the terrain-maker library.\n\n"
            "**Auto-generated from source code docstrings.**\n\n"
            "## Table of Contents\n\n"
        )

        # Generate TOC
        for module_name, _ in modules:
            full_content += f"- [{module_name}](#{module_name.replace('.', '-').lower()})\n"

        full_content += "\n---\n\n"
        full_content += '\n---\n\n'.join(doc_sections)

        # Write file
        output_path.write_text(full_content)
        print(f'\n✅ Generated API documentation:')
        print(f'  • File: {output_path}')
        print(f'  • Classes: {total_classes}')
        print(f'  • Functions: {total_functions}')
    else:
        print('\n❌ No documentation generated')


def validate_docstrings() -> None:
    """Check for undocumented functions and classes."""
    print('🔍 Checking for undocumented code...\n')

    modules = scan_python_modules()
    issues = []

    for module_name, module_path in modules:
        print(f'  Checking {module_name}...')
        extractor = DocstringExtractor(str(module_path))

        if extractor.extract():
            # Check for missing docstrings
            for func_name, func_info in extractor.functions.items():
                if func_info['docstring'] == "No documentation":
                    issues.append(f"{module_name}.{func_name}: Missing docstring")
                    print(f'    ⚠️  {func_name}: Missing docstring')

            for class_name, class_info in extractor.classes.items():
                if class_info['docstring'] == "No documentation":
                    issues.append(f"{module_name}.{class_name}: Missing docstring")
                    print(f'    ⚠️  {class_name}: Missing class docstring')

    print()
    if issues:
        print(f'❌ Found {len(issues)} undocumented items:')
        for issue in issues:
            print(f'   - {issue}')
    else:
        print('✅ All public code is documented')


def show_stats() -> None:
    """Show documentation statistics."""
    print('📊 Documentation Statistics\n')

    modules = scan_python_modules()
    total_functions = 0
    total_classes = 0

    for module_name, module_path in modules:
        extractor = DocstringExtractor(str(module_path))
        if extractor.extract():
            total_functions += len(extractor.functions)
            total_classes += len(extractor.classes)
            print(f'  {module_name}:')
            print(f'    • Classes: {len(extractor.classes)}')
            print(f'    • Functions: {len(extractor.functions)}')

    print(f'\n  Total Classes: {total_classes}')
    print(f'  Total Functions: {total_functions}')


def show_help() -> None:
    """Show usage help."""
    print('📚 Python Documentation Generator\n')
    print('Usage: python scripts/python-docs.py [command]\n')
    print('Commands:')
    print('  (default)  Generate API_REFERENCE.md from docstrings')
    print('  validate   Check for undocumented code')
    print('  stats      Show documentation statistics')
    print('  help       Show this help message\n')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'generate'

    if command == 'help':
        show_help()
    elif command == 'validate':
        validate_docstrings()
    elif command == 'stats':
        show_stats()
    else:
        generate_all_docs()
