//! FPGA deployment-target database for the ferrotherm VM — from edge parts a classroom owns to
//! the cloud instances a lab rents. Every number is labelled: [DS] datasheet fact, [EST]
//! engineering estimate, [SWEEP] verified market/status research (Aug 2026). The capacity model
//! (p-bits per LUT budget, flips/s at a colored-update clock) is an ESTIMATE until the
//! calibration-anchor pass lands; the published-machine anchors below bound it from above.
//!
//! Toolchain ground truth (Aug 2026): full open flows exist for iCE40/ECP5 (yosys+nextpnr) and
//! Gowin (Apicula); 7-series is experimental-open (openXC7); UltraScale+/Versal require Vivado —
//! the compiler path there is emit-RTL+XDC and drive Vivado in batch (the aws-fpga model).

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    EdgeMicro,
    Edge,
    CloudInstance,
    PcieCard,
    AcademicCluster,
    Salvage,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpenFlow {
    /// Complete yosys+nextpnr flow, production quality.
    Full,
    /// Real but young or partial coverage.
    Partial,
    /// Vendor tools required (emit RTL + drive them in batch).
    VendorOnly,
}

pub struct FpgaTarget {
    pub name: &'static str,
    pub class: Class,
    /// LUT count (native size per provenance; LUT6 fabrics weighted 1.6x in the capacity model).
    pub luts: u32,
    pub lut6: bool,
    pub bram_kb: u32,
    pub dsp: u32,
    /// Realistic dense-design clock range, MHz.
    pub clock_mhz: (u16, u16),
    /// Power envelope, W.
    pub power_w: (f32, f32),
    pub price: &'static str,
    pub availability: &'static str,
    pub open_flow: OpenFlow,
    pub provenance: &'static str,
}

impl FpgaTarget {
    /// [EST, anchored] p-bit capacity at ~150 LUT4-equivalents per p-bit — the density DSIM-2
    /// realized on Versal silicon (~55.5k p-bits per VP1902; arXiv:2606.25313) — with LUT6
    /// fabrics weighted 1.6x. The first version of this model assumed 60 LUT/p-bit and was
    /// caught ~10x optimistic by the anchor test below; this is the corrected, measured density.
    pub fn est_pbits(&self) -> u32 {
        let lut4_eq = if self.lut6 { self.luts as f64 * 1.6 } else { self.luts as f64 };
        (lut4_eq / 150.0) as u32
    }
    /// [EST, anchored] flips/s with 4-color scheduling at the dense-SAMPLING-fabric clock: the
    /// part's midpoint clock capped at 33 MHz, the highest a full-die colored-Gibbs fabric has
    /// actually closed timing at (DSIM-2). Small parts run below the cap on their own clocks.
    pub fn est_flips_per_s(&self) -> f64 {
        let mid = 0.5 * (self.clock_mhz.0 as f64 + self.clock_mhz.1 as f64) * 1e6;
        let clk = mid.min(33e6);
        self.est_pbits() as f64 / 4.0 * clk
    }
}

/// The database, smallest to largest by capability class.
pub const TARGETS: &[FpgaTarget] = &[
    FpgaTarget { name: "Lattice iCE40 UP5K", class: Class::EdgeMicro, luts: 5280, lut6: false,
        bram_kb: 1144, dsp: 8, clock_mhz: (12, 24), power_w: (0.001, 0.01),
        price: "$11 part; UPduino $25-36; iCEBreaker $80 [SWEEP]",
        availability: "in stock (DigiKey)", open_flow: OpenFlow::Full,
        provenance: "[DS] 5,280 LUT4, 120Kb EBR + 1,024Kb SPRAM, 8 MAC16. [EST] clock, power. Milliwatt always-on class." },
    FpgaTarget { name: "Gowin GW1NR-9 (Tang Nano 9K)", class: Class::EdgeMicro, luts: 8640, lut6: false,
        bram_kb: 468, dsp: 20, clock_mhz: (27, 60), power_w: (0.02, 0.1),
        price: "board $15-25 [SWEEP]", availability: "in stock (Sipeed/Seeed)", open_flow: OpenFlow::Partial,
        provenance: "[DS] 8,640 LUT4, 468Kb BSRAM, +64Mb PSRAM in package. Apicula open flow first-class. Cheapest real FPGA." },
    FpgaTarget { name: "Gowin GW2AR-18 (Tang Nano 20K)", class: Class::EdgeMicro, luts: 20736, lut6: false,
        bram_kb: 828, dsp: 48, clock_mhz: (50, 100), power_w: (0.05, 0.3),
        price: "board $25-40 [SWEEP]", availability: "in stock", open_flow: OpenFlow::Partial,
        provenance: "[DS] 20,736 LUT4, 828Kb BSRAM, +64Mb SDRAM. [EST] best flips-per-dollar of the cheap fleet." },
    FpgaTarget { name: "Lattice ECP5 LFE5U-85F", class: Class::Edge, luts: 83640, lut6: false,
        bram_kb: 3700, dsp: 156, clock_mhz: (60, 100), power_w: (0.2, 1.5),
        price: "$20-45 part; ULX3S $155; OrangeCrab ~$99 [SWEEP]", availability: "in stock", open_flow: OpenFlow::Full,
        provenance: "[DS] 84K LUT4, 3.7Mb. Largest FPGA with a COMPLETE open flow (yosys+nextpnr+trellis) - the primary open compile target." },
    FpgaTarget { name: "Lattice CertusPro-NX 100", class: Class::Edge, luts: 100000, lut6: false,
        bram_kb: 7300, dsp: 156, clock_mhz: (100, 150), power_w: (0.2, 1.0),
        price: "[EST] $100-250 part", availability: "in stock (DigiKey/Arrow)", open_flow: OpenFlow::VendorOnly,
        provenance: "[DS] ~100K LC, 7.3Mb w/ LRAM (best weight-store per dollar in Lattice line), 28nm FD-SOI. Open flow covers smaller Nexus only." },
    FpgaTarget { name: "AMD Artix-7 XC7A100T", class: Class::Edge, luts: 63400, lut6: true,
        bram_kb: 4860, dsp: 240, clock_mhz: (100, 150), power_w: (1.0, 3.0),
        price: "[EST] $100-200 part; Arty A7-100T ~$299", availability: "in stock", open_flow: OpenFlow::Partial,
        provenance: "[DS] 63,400 LUT6, 4,860Kb BRAM, 240 DSP48E1. Vivado free tier; openXC7 experimental. The university-lab default." },
    FpgaTarget { name: "AMD Artix-7 XC7A200T", class: Class::Edge, luts: 134600, lut6: true,
        bram_kb: 13140, dsp: 740, clock_mhz: (100, 150), power_w: (2.0, 6.0),
        price: "[EST] $300-500 part; boards $150-600", availability: "in stock", open_flow: OpenFlow::Partial,
        provenance: "[DS] 134,600 LUT6, 13.1Mb, 740 DSP. [EST] cheapest plausible 0.1 Tflips/s-class part." },
    FpgaTarget { name: "Alchitry Cu V2 (iCE40-HX8K)", class: Class::EdgeMicro, luts: 7680, lut6: false,
        bram_kb: 128, dsp: 0, clock_mhz: (50, 100), power_w: (0.05, 0.3),
        price: "$59.99 (SparkFun DEV-27875) [DS]", availability: "in stock [verified Aug 2026]", open_flow: OpenFlow::Full,
        provenance: "[DS] Lattice iCE40-HX8K: 7,680 LUT4, 32x4Kb EBR = 128Kb, no DSP; 100 MHz osc, FT2232 USB. FULL open flow (IceStorm/yosys/nextpnr, stated on the listing). Dean's ladder, rung 1." },
    FpgaTarget { name: "Alchitry Au V2 (XC7A35T-2)", class: Class::Edge, luts: 20800, lut6: true,
        bram_kb: 1800, dsp: 90, clock_mhz: (100, 150), power_w: (1.0, 5.0),
        price: "$149.99 (SparkFun DEV-27874) [DS]", availability: "in stock [verified Aug 2026]", open_flow: OpenFlow::Partial,
        provenance: "[DS] XC7A35T-2FTG256I (-2 speed vs V1's -1): 20,800 LUT6, 1,800Kb BRAM, 90 DSP48E1; 256MB DDR3L-800. Vivado ML Standard (free). Dean's ladder, rung 2." },
    FpgaTarget { name: "Alchitry Pt V2 (XC7A100T-2)", class: Class::Edge, luts: 63400, lut6: true,
        bram_kb: 4860, dsp: 240, clock_mhz: (100, 200), power_w: (2.0, 8.0),
        price: "$349.99 (SparkFun DEV-27873) [DS]", availability: "in stock [verified Aug 2026]", open_flow: OpenFlow::Partial,
        provenance: "[DS] XC7A100T-2FGG484I (the 'FGG84I' on the vendor listings is a typo - confirmed from the Rev A schematic): 63,400 LUT6, 4,860Kb, 240 DSP, 4x GTP 6.25Gb/s broken out = PCIe Gen2 x4 capable; 256MB DDR3L; 206 I/O. Vivado free. Dean's ladder, rung 3 (top Alchitry)." },
    FpgaTarget { name: "AMD Kria KV260 (XCK26 SOM)", class: Class::Edge, luts: 117120, lut6: true,
        bram_kb: 23616, dsp: 1248, clock_mhz: (200, 300), power_w: (5.0, 15.0),
        price: "$249 AMD list; $274-292 distributors, in stock [verified Aug 2026]", availability: "in stock (DigiKey 218, Mouser 261)", open_flow: OpenFlow::VendorOnly,
        provenance: "[DS, DS987] XCK26-SFVC784-2LV: 117,120 LUT6, 5.1Mb BRAM + 18Mb URAM, 1,248 DSP48E2, 4x GTH; quad-A53 PS + 4GB DDR4 on-SOM (self-contained sampling appliance: fabric samples, PS serves). Vivado ML Standard covers Kria FREE - the widely repeated Enterprise-required claim is FALSE (AMD licensing FAQ). Dean's ladder, rung 4." },
    FpgaTarget { name: "Numato Aller (M.2 XC7A200T-2)", class: Class::Edge, luts: 134600, lut6: true,
        bram_kb: 13140, dsp: 740, clock_mhz: (100, 150), power_w: (2.0, 10.0),
        price: "~$500 [EST: quote-only since 2025; last published $499.99]", availability: "made to order (Numato); the ONLY first-party 2280 M.2 FPGA still manufactured (LiteFury/NiteFury/Acorn all dead)",
        open_flow: OpenFlow::Partial,
        provenance: "[DS] XC7A200T-2FBG484I in M.2 2280 M-key, PCIe Gen2 x4 (~2GB/s), 256MB DDR3-800, TPM, mandatory heatsink (~8-10W M.2 budget). The open compute-stick, buyable now vs the vendor's 2027 stick. Vivado free. Dean's ladder, rung 5." },
    FpgaTarget { name: "Salvage Kintex US+ KU3P (Alibaba AS02MC04)", class: Class::Salvage, luts: 163000, lut6: true,
        bram_kb: 26200, dsp: 1368, clock_mhz: (200, 350), power_w: (10.0, 25.0),
        price: "~$200 on eBay vs $900+ commercial boards [SWEEP]", availability: "salvage market, circulating Feb 2026",
        open_flow: OpenFlow::VendorOnly,
        provenance: "[DS-approx] 163K LUT6, 12.7Mb BRAM + 13.5Mb URAM, 1,368 DSP. UltraScale+ primitives = same-fabric CI target for VU47P/VU9P netlists; Vivado Standard (free) covers KU3P; RapidWright path exists." },
    FpgaTarget { name: "AMD Alveo U55C (PCIe card)", class: Class::PcieCard, luts: 1304000, lut6: true,
        bram_kb: 349000, dsp: 9024, clock_mhz: (250, 450), power_w: (75.0, 150.0),
        price: "$4.5k refurb - $6.1k new [SWEEP]", availability: "ACTIVE product; in stock (SHI/DigiKey)",
        open_flow: OpenFlow::VendorOnly,
        provenance: "[DS] VU47P-class: 1,304K LUT6, 70.9Mb BRAM + 270Mb URAM, 9,024 DSP, 16GiB HBM2 @460GB/s. Same silicon as AWS F2: develop local, burst to cloud." },
    FpgaTarget { name: "AMD Alveo V80 (PCIe card)", class: Class::PcieCard, luts: 2500000, lut6: true,
        bram_kb: 0, dsp: 10848, clock_mhz: (300, 500), power_w: (150.0, 300.0),
        price: "$9,495 MSRP; orderable, ~10wk lead [SWEEP]", availability: "ACTIVE flagship (AMD: 'all new designs')",
        open_flow: OpenFlow::VendorOnly,
        provenance: "[DS-approx] Versal HBM: ~2.5M LUTs, 10,848 DSP58, 32GiB HBM2e @820GB/s, hard NoC. No published Ising/sampling work yet [SWEEP: searched, empty] - first-mover slot; free validation units at ETHZ-HACC." },
    FpgaTarget { name: "AWS EC2 F2 (f2.6xlarge, 1x VU47P)", class: Class::CloudInstance, luts: 1304000, lut6: true,
        bram_kb: 349000, dsp: 9024, clock_mhz: (250, 450), power_w: (75.0, 150.0),
        price: "$1.98/hr on-demand, ~$0.66 spot [SWEEP, us-east-1 Aug 2026]",
        availability: "ACTIVE, 11+ regions; Vivado license included", open_flow: OpenFlow::VendorOnly,
        provenance: "[DS] VU47P + 16GiB HBM2 + 64GiB DDR4; aws-fpga f2 branch, Vivado/Vitis 2024.1-2025.1. THE cloud target. F1 retired Dec 2025." },
    FpgaTarget { name: "AWS EC2 F2 x8 (f2.48xlarge)", class: Class::CloudInstance, luts: 1303680, lut6: true,
        bram_kb: 349000, dsp: 9024, clock_mhz: (250, 350), power_w: (600.0, 1800.0),
        price: "$15.84/hr on-demand us-east-1; spot ~$3.65-4.96 [DS, verified Aug 2026]",
        availability: "ACTIVE; 8x VU47P + 192 vCPU EPYC Milan + 2TiB + 100Gbps ENA", open_flow: OpenFlow::VendorOnly,
        provenance: "[DS] 8x XCVU47P (1,303,680 LUT6, 70.9Mb BRAM + 270Mb URAM, 9,024 DSP, 16GiB HBM2 each; PCIe Gen4 x8/FPGA; shell clk 250 MHz). ARCHITECTURE-CRITICAL [AWS re:Post]: NO FPGA-to-FPGA links - no P2P, no ring (F1 had both). Inter-FPGA data bounces through host memory. Therefore the x8 tier runs REPLICA-EXCHANGE (parallel tempering: scalar energies per swap round fit host-mediated topology), not DSIM-2 lattice partitioning, which needs direct links. Dean's ladder, top rung." },
    FpgaTarget { name: "Azure NP10s (1x Alveo U250)", class: Class::CloudInstance, luts: 1728000, lut6: true,
        bram_kb: 54000, dsp: 12288, clock_mhz: (250, 400), power_w: (100.0, 225.0),
        price: "$1.65/hr on-demand, ~$0.30 spot [SWEEP]",
        availability: "SUNSETTING: retirement May 31 2027; new reservations closed Apr 2026", open_flow: OpenFlow::VendorOnly,
        provenance: "[DS] VU13P: 1,728K LUT6, 12,288 DSP (largest DSP pool of the family), 64GB DDR4. Attestation-gated bitstreams; older Vitis shells. 10-month experiment window only." },
    FpgaTarget { name: "AMD HACC (ETHZ + partners, academic)", class: Class::AcademicCluster, luts: 1304000, lut6: true,
        bram_kb: 349000, dsp: 9024, clock_mhz: (250, 450), power_w: (75.0, 300.0),
        price: "$0 for accepted academic teams [SWEEP]",
        availability: "10x U55C + U250/U280/VCK5000 + V80 at ETHZ; sister clusters UIUC/NUS/Paderborn; apply at amd-haccs.io",
        open_flow: OpenFlow::VendorOnly,
        provenance: "[SWEEP verified Aug 2026] maintained Vitis/Vivado, 100G inter-card. The zero-cost route to F2-class silicon and V80." },
    FpgaTarget { name: "NSF Open Cloud Testbed (24x Alveo U280)", class: Class::AcademicCluster, luts: 1304000, lut6: true,
        bram_kb: 349000, dsp: 9024, clock_mhz: (250, 450), power_w: (100.0, 225.0),
        price: "$0, NSF-funded, allocation by proposal [SWEEP]",
        availability: "bare-metal via CloudLab; dual 100GbE per card into shared switch",
        open_flow: OpenFlow::VendorOnly,
        provenance: "[DS] U280 = VU37P + 8GiB HBM2. The only two-dozen-card HBM FPGA mesh rentable for $0 - matches the DSIM-2 1-bit-boundary-exchange distributed-Gibbs pattern." },
];

/// Published machine anchors — the measured numbers the capacity model must stay below (from the
/// market sweep's evidence index; the dedicated calibration pass will refine them).
pub const ANCHORS: &[(&str, &str)] = &[
    ("Aadit et al., Nature Electronics 2022 (VCU118 / VU9P)",
     "multiplexed chromatic Gibbs sparse Ising machine; ~10^6x CPU Gibbs, 5-18x vs optimized TPU/GPU; flips/s linear in p-bits [measured]"),
    ("DSIM-2, arXiv:2606.25313 (Jun 2026, UCSB/CMU/Siemens/KFUPM)",
     "18x Versal Premium VP1902; ~10^6 p-bits; graph-colored Gibbs, N_color=22, 100^3 Edwards-Anderson glass; 1e12 flips/s @ 11 MHz, 3e12 @ 33 MHz, 1.4-1.6 kW; 1-bit boundary exchange between chips [measured]"),
    ("Toshiba SBM on AWS F1 (2019) -> SQBM+ (Azure Quantum 2022, AWS Marketplace 2023)",
     "FPGA dSB solver to 100,000-variable Ising [vendor claims]"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn database_is_coherent() {
        assert!(TARGETS.len() >= 20);
        for t in TARGETS {
            assert!(t.luts > 0 && t.clock_mhz.0 <= t.clock_mhz.1);
            assert!(!t.provenance.is_empty() && !t.price.is_empty());
            // capacity sanity against the measured machine: DSIM-2 realized ~9.7k-19.4k
            // flips/s per LUT (3e12 over 18 x ~8.6M-LUT VP1902s at N_color=22). Our model may
            // exceed that density only through its shallower coloring (4 vs 22), never by more
            // than that ratio: bound at 1.2e5 flips/s/LUT and 3e11 per chip.
            let density = t.est_flips_per_s() / t.luts as f64;
            assert!(density < 1.2e5, "{}: {:.1e} flips/s/LUT exceeds anchor-scaled density", t.name, density);
            assert!(t.est_flips_per_s() < 3e11,
                "{}: estimate {:.1e} exceeds the measured per-chip anchor", t.name, t.est_flips_per_s());
        }
        let up5k = &TARGETS[0];
        let v80 = TARGETS.iter().find(|t| t.name.contains("V80")).unwrap();
        assert!(up5k.est_pbits() < v80.est_pbits());
    }
}
