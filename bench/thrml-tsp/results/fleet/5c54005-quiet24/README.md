# 5c54005 quiet-gate evidence

Three hosts. Grades stay on the host and the meter that produced them. Nothing here is pooled.

Harness `5c54005fd0ff8dc62adedda478faa52388911902`, clean tree. The quiet
threshold is 2. It was not loosened. No joules are filled in where a receipt
left them empty.

| Directory | Host | Meter | Grade |
| --- | --- | --- | --- |
| `m5/` | `Davids-MacBook-Pro-M5-Max-48GB-4.local` | macmon 0.11.0, scope `soc-telemetry` | preflight pass, energy refused |
| `framework-128gb-0000/` | AMD Ryzen AI MAX+ 395, cpus `4-7,20-23` | RAPL package and core, separate | `load_gated` |
| `jetson-hub/` attempt 2 | hostname `jetson-hub`, Intel Core i9-13900H | RAPL, refused before a trial | `load_gated` |
| `jetson-hub/attempt-003/` | same host, cpus `8-11` | RAPL package, RAPL core, and NVML, separate | quiet preflight on 5c54005 (max load1 0.55); trial windows unguarded; NOT quiet-host joules under the c82c1e96 rule |

`soc-telemetry` is not wall power. RAPL package and RAPL core are not added.
The M5 refusal is not a RAPL result.
