# 5c54005 quiet-gate evidence

Three hosts, three meters, three grades. Nothing here is pooled.

Harness `5c54005fd0ff8dc62adedda478faa52388911902`, clean tree. The quiet
threshold is 2. It was not loosened. No joules are filled in where a receipt
left them empty.

| Directory | Host | Meter | Grade |
| --- | --- | --- | --- |
| `m5/` | `Davids-MacBook-Pro-M5-Max-48GB-4.local` | macmon 0.11.0, scope `soc-telemetry` | preflight pass, energy refused |
| `framework-128gb-0000/` | AMD Ryzen AI MAX+ 395, cpus `4-7,20-23` | RAPL package and core, separate | `load_gated` |
| `jetson-hub/` | hostname `jetson-hub`, Intel Core i9-13900H | RAPL package and core, separate | `load_gated` |

`soc-telemetry` is not wall power. RAPL package and RAPL core are not added.
The M5 refusal is not a RAPL result.
