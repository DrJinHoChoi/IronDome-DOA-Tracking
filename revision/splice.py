# -*- coding: utf-8 -*-
"""본문의 옛 Experimental Results 절을 새 focused 절로 치환."""
import io, os
base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "ieee-tsp-2026")
main_path = os.path.join(base, "cop_rfs_tsp2026_R1.tex")
sec_path = os.path.join(base, "sec_experiments_R1.tex")

with io.open(main_path, encoding="utf-8") as f:
    main = f.read()
with io.open(sec_path, encoding="utf-8") as f:
    sec = f.read()

start_tag = r"\section{Experimental Results}"
end_tag = r"\section{Conclusion}"
i = main.index(start_tag)
j = main.index(end_tag)
new = main[:i] + sec.strip() + "\n\n" + main[j:]

with io.open(main_path, "w", encoding="utf-8") as f:
    f.write(new)

print(f"spliced: removed {j-i} chars, inserted {len(sec.strip())} chars")
print(f"main length: {len(main)} -> {len(new)}")
