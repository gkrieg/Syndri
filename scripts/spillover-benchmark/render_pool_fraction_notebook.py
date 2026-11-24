#!/usr/bin/env python3
"""Execute the figure notebook and save its displayed tables/plots without Jupyter."""
from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import traceback

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import IPython.display as ipdisplay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=Path(__file__).resolve().parents[2]/
                        'figures/Figure3_SyndriSIM_PoolFraction.ipynb')
    parser.add_argument('--cell-ids', nargs='+',
                        help='Execute only these code-cell IDs, preserving all other outputs')
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text())
    namespace = {'__name__': '__main__'}
    execution_count = 0
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        if args.cell_ids is not None and cell.get('id') not in args.cell_ids:
            continue
        execution_count += 1
        print(f'Cell {index}: {cell.get("id", "")}', flush=True)
        outputs, stdout = [], io.StringIO()
        original_display, original_show = ipdisplay.display, plt.show

        def display(value, *args, **kwargs):
            data = {'text/plain': repr(value).splitlines(keepends=True)}
            if hasattr(value, '_repr_html_'):
                html = value._repr_html_()
                if html:
                    data['text/html'] = html.splitlines(keepends=True)
            outputs.append(dict(output_type='display_data', data=data, metadata={}))

        def show(*args, **kwargs):
            for number in plt.get_fignums():
                fig = plt.figure(number)
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=180, bbox_inches='tight')
                outputs.append(dict(output_type='display_data', data={
                    'image/png': base64.b64encode(buffer.getvalue()).decode(),
                    'text/plain': [f'<Figure: {cell.get("id", index)}>']}, metadata={}))
                plt.close(fig)

        ipdisplay.display, plt.show = display, show
        namespace['display'] = display
        failure = None
        try:
            with redirect_stdout(stdout):
                exec(compile(''.join(cell['source']), f'{args.notebook.name}:cell-{index}', 'exec'), namespace)
        except Exception as error:
            failure = error
            outputs.append(dict(output_type='error', ename=type(error).__name__, evalue=str(error),
                                traceback=traceback.format_exc().splitlines()))
        finally:
            ipdisplay.display, plt.show = original_display, original_show
            plt.close('all')
        if stdout.getvalue():
            outputs.insert(0, dict(output_type='stream', name='stdout', text=stdout.getvalue().splitlines(keepends=True)))
            print(stdout.getvalue(), end='', flush=True)
        cell['outputs'], cell['execution_count'] = outputs, execution_count
        args.notebook.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n')
        if failure is not None:
            raise failure
    print(f'Rendered {execution_count} code cells: {args.notebook}', flush=True)


if __name__ == '__main__':
    main()
