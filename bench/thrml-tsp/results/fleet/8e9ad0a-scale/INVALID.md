# INVALID for energy

This directory is the `--scale` run of revision `8e9ad0a` on `framework-128gb-0000`, evidence grade contended host. The tables, summary, and reports are the measurement as it came off the machine.

**Do not use the joules.** 143 of 150 trials have negative incremental energy, and 99 of those were still marked attributable. The idle baseline was taken once, at 144.8 W, while load1 moved from about 11 to about 24 under other jobs. A single baseline on that host is not an idle. No energy comparison is supported.

The raw counter traces were too large to put in the repository. They stay on the machine that ran the benchmark. What is here is the tables, the summary, and the reports.

Accuracy rows are a separate question from those joules. The report's oracle field says "exact enumeration" for every workload, including mean-only scaled references. That label is wrong for those rows. The 32×32 and 64×64 Ising numbers were compared with Onsager's infinite lattice, not with a finite-grid mean.
