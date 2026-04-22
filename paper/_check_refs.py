import re
with open('cop_rfs_tsp2026.tex', 'r', encoding='utf-8') as f:
    text = f.read()

cites = set()
# Find all \cite{...} by searching for the literal text
idx = 0
while True:
    pos = text.find('\\cite{', idx)
    if pos == -1:
        break
    brace_start = pos + 6  # len('\\cite{')
    brace_end = text.index('}', brace_start)
    keys = text[brace_start:brace_end]
    for key in keys.split(','):
        cites.add(key.strip())
    idx = brace_end + 1

with open('references.bib', 'r', encoding='utf-8') as f:
    bib = f.read()

bib_keys = set()
for m in re.finditer(r'@\w+\{(\w+),', bib):
    bib_keys.add(m.group(1))

print('=== Cited keys ===')
for k in sorted(cites):
    print(f'  {k}')
print(f'\nTotal cited: {len(cites)}')
print(f'\n=== Bib keys ===')
for k in sorted(bib_keys):
    print(f'  {k}')
print(f'\nTotal bib entries: {len(bib_keys)}')
print(f'\n=== Missing from bib (cited but no entry) ===')
missing = sorted(cites - bib_keys)
for k in missing:
    print(f'  {k}')
if not missing:
    print('  (none)')
print(f'\n=== Unused in bib (entry but not cited) ===')
unused = sorted(bib_keys - cites)
for k in unused:
    print(f'  {k}')
if not unused:
    print('  (none)')
