import runpy

g = runpy.run_path('auto-job-apply/autojobapply.py')
if 'process_once' in g:
    g['process_once']()
else:
    print('process_once not found in autojobapply.py')
