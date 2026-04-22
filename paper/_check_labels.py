import re
with open('cop_rfs_tsp2026.tex', 'r', encoding='utf-8') as f:
    text = f.read()

labels = set()
refs = set()

# Find labels
idx = 0
while True:
    pos = text.find('\\label{', idx)
    if pos == -1:
        break
    brace_start = pos + 7
    brace_end = text.index('}', brace_start)
    labels.add(text[brace_start:brace_end])
    idx = brace_end + 1

# Find refs
for prefix in ['\\ref{', '\\eqref{']:
    idx = 0
    while True:
        pos = text.find(prefix, idx)
        if pos == -1:
            break
        brace_start = pos + len(prefix)
        brace_end = text.index('}', brace_start)
        refs.add(text[brace_start:brace_end])
        idx = brace_end + 1

print(f"Labels defined: {len(labels)}")
print(f"Refs used: {len(refs)}")
print(f"\nRefs without labels (BROKEN REFS):")
for r in sorted(refs - labels):
    print(f"  {r}")
if not (refs - labels):
    print("  (none)")
print(f"\nLabels without refs (UNUSED LABELS):")
for l in sorted(labels - refs):
    print(f"  {l}")
if not (labels - refs):
    print("  (none)")

# Count specific items
print(f"\nFigure labels: {len([l for l in labels if l.startswith('fig:')])}")
print(f"Table labels: {len([l for l in labels if l.startswith('tab:')])}")
print(f"Algorithm labels: {len([l for l in labels if l.startswith('alg:')])}")
print(f"Equation labels: {len([l for l in labels if l.startswith('eq:')])}")
print(f"Section labels: {len([l for l in labels if l.startswith('sec:')])}")
