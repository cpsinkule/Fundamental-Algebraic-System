"""
Output display utilities for FAS Calculator.

Provides interactive Jupyter widgets for viewing large symbolic expressions
that would otherwise be too long to display comfortably in notebooks.
"""

import sympy as sp
from typing import Optional, Union
import re
import warnings

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, Latex, Markdown
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    warnings.warn(
        "ipywidgets or IPython not available. Install with: pip install ipywidgets",
        ImportWarning
    )


class SymbolicExpressionWidget:
    """
    Interactive widget for viewing large SymPy expressions in Jupyter notebooks.

    Displays a compact summary by default with expandable sections for:
    - LaTeX preview (truncated)
    - Full LaTeX rendering
    - Top N terms
    - Simplified form
    - Full string representation

    The full SymPy expression is always accessible via the `.expr` attribute.

    Parameters
    ----------
    expr : sp.Basic
        The SymPy expression to display
    name : str, optional
        A name/label for this expression (e.g., "Minor(0,0,1)")
    max_preview_length : int, optional
        Maximum character length for LaTeX preview (default: 500)
    max_terms_display : int, optional
        Number of terms to show in "Top N Terms" section (default: 10)

    Attributes
    ----------
    expr : sp.Basic
        The full SymPy expression (always accessible for computation)
    name : str
        Name/label of the expression

    Examples
    --------
    >>> from sympy import symbols, expand
    >>> x, y = symbols('x y')
    >>> big_expr = expand((x + y)**10)
    >>> widget = SymbolicExpressionWidget(big_expr, name="(x+y)^10")
    >>> widget.display()  # Shows interactive widget in notebook
    >>>
    >>> # Access full expression for computation
    >>> result = widget.expr.subs(x, 1)
    """

    def __init__(
        self,
        expr: sp.Basic,
        name: str = "Expression",
        max_preview_length: int = 500,
        max_terms_display: int = 10
    ):
        if not WIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets and IPython required for SymbolicExpressionWidget. "
                "Install with: pip install ipywidgets"
            )

        self.expr = expr
        self.name = name
        self.max_preview_length = max_preview_length
        self.max_terms_display = max_terms_display

        # Internal caches
        self._latex_cache: Optional[str] = None

        # Compute metadata
        self._compute_metadata()

        # Build widget components
        self._build_widget()

    def _compute_metadata(self):
        """Compute summary metadata about the expression."""
        # Convert to string for length estimation
        expr_str = str(self.expr)
        self.expr_length = len(expr_str)

        # Count terms (for Add expressions)
        if isinstance(self.expr, sp.Add):
            self.term_count = len(self.expr.args)
        else:
            self.term_count = 1

        # Get free symbols
        self.free_symbols = sorted(self.expr.free_symbols, key=str)
        self.symbol_count = len(self.free_symbols)

        # Polynomial degree (if applicable)
        try:
            if self.symbol_count > 0:
                self.degree = sp.degree(self.expr)
            else:
                self.degree = 0
        except:
            self.degree = "N/A"

        # Expression type
        self.expr_type = type(self.expr).__name__

    def _build_widget(self):
        """Build the interactive widget components."""
        # Summary section (always visible)
        summary_html = f"""
        <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px; background-color: #f9f9f9; margin-bottom: 10px;">
            <h3 style="margin-top: 0; color: #4CAF50;">📊 {self.name}</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 5px;"><b>Expression Type:</b></td>
                    <td style="padding: 5px;">{self.expr_type}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Total Size:</b></td>
                    <td style="padding: 5px;">{self.expr_length:,} characters</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Term Count:</b></td>
                    <td style="padding: 5px;">{self.term_count:,} terms</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Polynomial Degree:</b></td>
                    <td style="padding: 5px;">{self.degree}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Number of Variables:</b></td>
                    <td style="padding: 5px;">{self.symbol_count}</td>
                </tr>
            </table>
        </div>
        """
        self.summary_widget = widgets.HTML(value=summary_html)

        # Create accordion for expandable sections
        self.accordion = widgets.Accordion()

        # Section 1: Variables list
        if self.symbol_count > 0:
            vars_str = ", ".join([sp.latex(sym) for sym in self.free_symbols[:20]])
            if self.symbol_count > 20:
                vars_str += f", ... ({self.symbol_count - 20} more)"
            vars_widget = widgets.HTML(value=f"<div style='padding: 10px;'>$${vars_str}$$</div>")
        else:
            vars_widget = widgets.HTML(value="<div style='padding: 10px;'>No variables (constant expression)</div>")

        # Section 2: LaTeX preview (truncated) — lazy for very large expressions
        # Avoid generating LaTeX eagerly for huge expressions to prevent freezes
        PREVIEW_LAZY_THRESHOLD = 5000  # based on string length of expression

        def _get_latex_cached() -> str:
            if self._latex_cache is None:
                self._latex_cache = sp.latex(self.expr)
            return self._latex_cache

        if self.expr_length > PREVIEW_LAZY_THRESHOLD:
            preview_output = widgets.Output()
            with preview_output:
                print(
                    "Preview is lazy due to expression size. Click to render a truncated LaTeX preview."
                )
            preview_button = widgets.Button(
                description="Render LaTeX Preview",
                button_style='info'
            )

            def on_preview_click(b):
                with preview_output:
                    preview_output.clear_output()
                    print("Rendering truncated LaTeX preview...")
                    try:
                        latex_str = _get_latex_cached()
                        if len(latex_str) > self.max_preview_length:
                            latex_preview = latex_str[: self.max_preview_length] + " \\ldots \\text{(truncated)}"
                        else:
                            latex_preview = latex_str
                        display(Latex(f"$${latex_preview}$$"))
                    except Exception as e:
                        print(f"Preview rendering failed: {e}")

            preview_button.on_click(on_preview_click)
            latex_preview_widget = widgets.VBox([preview_output, preview_button])
        else:
            try:
                latex_str = _get_latex_cached()
                if len(latex_str) > self.max_preview_length:
                    latex_preview = latex_str[: self.max_preview_length] + " \\ldots \\text{(truncated)}"
                else:
                    latex_preview = latex_str
                latex_preview_widget = widgets.HTML(
                    value=f"<div style='padding: 10px; overflow-x: auto;'>$${latex_preview}$$</div>"
                )
            except Exception as e:
                latex_preview_widget = widgets.HTML(
                    value=f'<div style="padding: 10px; color: #b00;">LaTeX preview unavailable: {e}</div>'
                )

        # Section 3: Top N terms
        if isinstance(self.expr, sp.Add) and self.term_count > 1:
            terms = list(self.expr.args)[:self.max_terms_display]
            terms_latex = " + ".join([sp.latex(term) for term in terms])
            if self.term_count > self.max_terms_display:
                terms_latex += " + \\ldots \\text{(+" + str(self.term_count - self.max_terms_display) + " more terms)}"
            terms_widget = widgets.HTML(
                value=f"<div style='padding: 10px; overflow-x: auto;'>$${terms_latex}$$</div>"
            )
        else:
            terms_widget = widgets.HTML(
                value="<div style='padding: 10px;'>Expression is not a sum or has only one term</div>"
            )

        # Section 4: Simplified form
        simplified_widget = widgets.Output()
        with simplified_widget:
            print("Click 'Compute Simplified Form' to simplify...")
        simplify_button = widgets.Button(description="Compute Simplified Form", button_style='info')

        def on_simplify_click(b):
            with simplified_widget:
                simplified_widget.clear_output()
                print("Simplifying... (this may take a while for large expressions)")
                try:
                    simplified = sp.simplify(self.expr)
                    simplified_latex = sp.latex(simplified)
                    simplified_len = len(str(simplified))
                    print(f"Simplified to {simplified_len:,} characters")
                    display(Latex(f"$${simplified_latex}$$"))
                except Exception as e:
                    print(f"Simplification failed: {e}")

        simplify_button.on_click(on_simplify_click)
        simplified_section = widgets.VBox([simplify_button, simplified_widget])

        # Section 5: Full LaTeX (warning for very large expressions)
        if self.expr_length > 10000:
            full_latex_widget = widgets.Output()
            with full_latex_widget:
                print(f"⚠️  Warning: Expression is very large ({self.expr_length:,} characters)")
                print("Rendering full LaTeX may freeze your browser.")
            render_button = widgets.Button(description="Render Anyway", button_style='warning')

            def on_render_click(b):
                with full_latex_widget:
                    full_latex_widget.clear_output()
                    print("Rendering...")
                    try:
                        # Use cached LaTeX if already generated
                        display(Latex(f"$${_get_latex_cached()}$$"))
                    except Exception as e:
                        print(f"Rendering failed: {e}")

            render_button.on_click(on_render_click)
            full_latex_section = widgets.VBox([full_latex_widget, render_button])
        else:
            full_latex_widget = widgets.Output()
            with full_latex_widget:
                try:
                    display(Latex(f"$${_get_latex_cached()}$$"))
                except Exception as e:
                    print(f"Full LaTeX rendering failed: {e}")
            full_latex_section = full_latex_widget

        # Section 6: String representation — lazy for very large expressions
        STRING_LAZY_THRESHOLD = 10000
        if self.expr_length > STRING_LAZY_THRESHOLD:
            str_output = widgets.Output()
            with str_output:
                print(
                    f"String representation is large ({self.expr_length:,} characters). "
                    "Click to render; may be slow."
                )
            str_button = widgets.Button(description="Render String", button_style='warning')

            def on_str_click(b):
                with str_output:
                    str_output.clear_output()
                    print("Rendering string representation...")
                    try:
                        str_widget = widgets.Textarea(
                            value=str(self.expr),
                            layout=widgets.Layout(width='100%', height='300px'),
                            disabled=True,
                        )
                        display(str_widget)
                    except Exception as e:
                        print(f"String rendering failed: {e}")

            str_button.on_click(on_str_click)
            str_section = widgets.VBox([str_output, str_button])
        else:
            str_section = widgets.Textarea(
                value=str(self.expr),
                layout=widgets.Layout(width='100%', height='300px'),
                disabled=True,
            )

        # Assemble accordion
        self.accordion.children = [
            vars_widget,
            latex_preview_widget,
            terms_widget,
            simplified_section,
            full_latex_section,
            str_section
        ]
        self.accordion.set_title(0, f'📋 Variables ({self.symbol_count})')
        self.accordion.set_title(1, '🔍 LaTeX Preview (Truncated)')
        terms_title = (
            f"📝 Top {min(self.max_terms_display, self.term_count)} Terms"
            if isinstance(self.expr, sp.Add) and self.term_count > 1
            else "📝 Terms"
        )
        self.accordion.set_title(2, terms_title)
        self.accordion.set_title(3, '⚡ Simplified Form')
        self.accordion.set_title(4, '📄 Full LaTeX')
        self.accordion.set_title(5, '💻 String Representation')

        # Close all sections by default
        self.accordion.selected_index = None

        # Export button
        self.export_button = widgets.Button(
            description='💾 Export to File',
            button_style='success',
            tooltip='Export expression to file'
        )
        self.export_output = widgets.Output()

        def _sanitize_filename(text: str) -> str:
            # Keep only alphanumerics, underscore, hyphen, dot; replace others with underscore
            sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", text)
            # Collapse multiple underscores
            sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
            return sanitized or "expression"

        def on_export_click(b):
            with self.export_output:
                self.export_output.clear_output()
                filename = f"{_sanitize_filename(self.name)}.txt"
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Expression: {self.name}\n")
                        f.write(f"Type: {self.expr_type}\n")
                        f.write(f"Terms: {self.term_count}\n")
                        f.write(f"Size: {self.expr_length} characters\n")
                        f.write(f"Degree: {self.degree}\n")
                        f.write(f"\n{'='*80}\n")
                        # Use cached LaTeX to avoid regenerating
                        f.write(f"LaTeX:\n{_get_latex_cached()}\n")
                        f.write(f"\n{'='*80}\n")
                        f.write(f"String:\n{str(self.expr)}\n")
                    print(f"✅ Exported to: {filename}")
                except Exception as e:
                    print(f"❌ Export failed: {e}")

        self.export_button.on_click(on_export_click)

        # Main container
        self.container = widgets.VBox([
            self.summary_widget,
            self.accordion,
            widgets.HBox([self.export_button]),
            self.export_output
        ])

    def display(self):
        """Display the widget in the notebook."""
        display(self.container)

    def _repr_html_(self):
        """Return HTML representation for automatic display in notebooks."""
        # This allows the widget to display automatically when returned from a cell
        self.display()
        return ""

    def show_terms(self, n: int = 10) -> None:
        """
        Display the first n terms of the expression.

        Parameters
        ----------
        n : int
            Number of terms to display
        """
        if isinstance(self.expr, sp.Add):
            terms = list(self.expr.args)[:n]
            total = len(self.expr.args)
            print(f"Showing {len(terms)} of {total} terms:")
            for i, term in enumerate(terms, 1):
                display(Latex(f"$$\\text{{Term {i}:}} \\quad {sp.latex(term)}$$"))
            if total > n:
                print(f"... ({total - n} more terms)")
        else:
            print("Expression is not a sum (single term):")
            display(Latex(f"$${sp.latex(self.expr)}$$"))

    def export_to_file(self, filename: Optional[str] = None) -> str:
        """
        Export the full expression to a text file.

        Parameters
        ----------
        filename : str, optional
            Output filename. If not provided, generates one based on expression name.

        Returns
        -------
        str
            The filename that was written
        """
        def _sanitize_filename(text: str) -> str:
            sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", text)
            sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
            return sanitized or "expression"

        if filename is None:
            base = _sanitize_filename(self.name)
            filename = f"{base}.txt"
        else:
            base = _sanitize_filename(filename)
            filename = base if base.endswith('.txt') else f"{base}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Expression: {self.name}\n")
            f.write(f"Type: {self.expr_type}\n")
            f.write(f"Terms: {self.term_count}\n")
            f.write(f"Size: {self.expr_length} characters\n")
            f.write(f"Degree: {self.degree}\n")
            f.write(f"\n{'='*80}\n")
            # Use cached LaTeX to avoid recomputation cost
            latex_full = self._latex_cache if self._latex_cache is not None else sp.latex(self.expr)
            self._latex_cache = latex_full
            f.write(f"LaTeX:\n{latex_full}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"String:\n{str(self.expr)}\n")

        return filename

    def get_latex(self, truncate: Optional[int] = None) -> str:
        """
        Get LaTeX representation of the expression.

        Parameters
        ----------
        truncate : int, optional
            If provided, truncate to this many characters

        Returns
        -------
        str
            LaTeX string
        """
        if self._latex_cache is None:
            self._latex_cache = sp.latex(self.expr)
        latex = self._latex_cache
        if truncate and len(latex) > truncate:
            return latex[:truncate] + r" \ldots"
        return latex


def create_expression_widget(
    expr: sp.Basic,
    name: str = "Expression",
    **kwargs
) -> SymbolicExpressionWidget:
    """
    Convenience function to create a SymbolicExpressionWidget.

    Parameters
    ----------
    expr : sp.Basic
        SymPy expression to display
    name : str
        Name/label for the expression
    **kwargs
        Additional arguments passed to SymbolicExpressionWidget

    Returns
    -------
    SymbolicExpressionWidget
        The widget instance
    """
    return SymbolicExpressionWidget(expr, name=name, **kwargs)
