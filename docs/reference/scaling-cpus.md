# Scaling up the CPUs

Self-play is CPU-bound: the games and the actors' Python set the speed, and the RTX 3090 is far from full ([optimizations](../optimizations.md), [GPU unlock](gpu-unlock.md)). This page asks what the cheapest way is to get more games per second: what to buy, or whether to rent. Prices are UK prices of 24 October 2026, with UK electricity.

## What the workload needs

* **CPU.** On the training machine (Ryzen 9 5950X, 16 Zen 3 cores) self-play makes ~1,660 agent steps per second, ~104 a core. The machine's work is the games (~49% of the CPU), the actors' Python (~36%, less since the C features of 2026-10-24) and the kernel. A games-only machine carries the same per-step work without the learner.
* **Memory.** ~445 MB a game as it runs now (measured, PSS: the game ~346 MB, almost all its own heap, and ~95 MB of Wine's helper processes, each game having a prefix of its own), plus ~460 MB for each actor process. A machine with the throughput of ours runs ~56 games: ~30 GB, so 32 GB as things are. The kernel's same-page merging could make it ~16 GB ([below](#memory-same-page-merging)).
* **Inference.** Each step needs the policy's forward pass. Today the games of a run share the learner's machine and its GPU. A second machine works now for evaluation games, with inference on its own GPU (the laptop's 3050 Ti plays 16 games: [a remote evaluation worker](../architecture.md#a-remote-evaluation-worker)). Training games on another machine need more work first: the trajectories and weights sent over the network, and inference there (a small GPU, or the CPU) or sent to the 3090 over the LAN. None of this is built.
* **No GPU for the games.** They draw nothing (Direct3D calls return at once).

## The box we bought, measured (2026-10-30)

A Minisforum MS-A2 with the Ryzen 9 8945HX (`tora-ms-a2`): 16 cores, 32 threads, **one** 32 GB module (single channel, 5200 MT/s, 28 GB visible). Games alone, built-in AI against built-in AI on `duelfast`, with no wait for orders (`scripts/wine_sync_ab.py --wait-ms 0`), both machines otherwise idle:

| games | MS-A2: wall a step, game steps/s | 5950X: wall a step, game steps/s |
|---|---|---|
| 16 | 6.4 ms, ~2,500 | 6.4 ms, ~2,500 |
| 32 | 7.3 ms, ~4,400 | 7.1 ms, ~4,500 |
| 48 | 10.7 ms, ~4,500 | |

CPU a step 6.4-6.6 ms on both, so the one memory channel costs nothing visible up to 32 games. The ceiling is about 32 games' worth, the same as the main machine's. Those are early-game steps with no actors, so a run's rates are lower. Memory ~375 MB a game: 28 GB holds ~65 games without actors.

As an evaluation worker with inference on its CPU (32 games, 16 processes): 199 games in its first 11 minutes, ~1,085 an hour against the laptop's ~790 (the first minutes favour short games), 29% of the CPU idle, 20 GB used, Tctl ~90 °C at ~4.5 GHz (no throttling so far; the chip's limit is 100 °C).

The link to the main machine, which is on Wi-Fi: 36 MB/s to the box, 62 MB/s from it, 3.8 ms idle; **33 ms on average and 69 ms at worst while the link is busy**. A step has 35-40 ms for its orders, so remote training games need a cable into the main machine.

With a direct cable between the main machine's `eno1` and the box's `enp4s0` (2.5 GbE, addresses 10.77.0.1 and 10.77.0.2, NetworkManager's `games-link`, no internet route over it): **281 MB/s each way, 0.56 ms idle, 0.87 ms on average and 1.4 ms at worst while busy**. Tailscale's traffic takes the cable by itself.

As a training machine: 56 games with same-page merging and four gateways, 3,280 to **4,625 steps/s**; the limit is now the GPU ([log](optimization-log.md#training-games-on-a-second-machine-2026-10-30)).

## Electricity

| | |
|---|---|
| Unit rate (Ofgem cap, October-December 2026, direct debit, GB average) | **26.32p/kWh**, no VAT from 1 October 2026 to 31 March 2027 |
| January 2027 forecast | about +23-25% (Uswitch, "low confidence"; Octopus) |
| One watt drawn day and night for two years | **£4.61** (at 30p: £5.26) |

The training machine draws ~600 W at the wall under training (CPU ~140 W, GPU ~350 W at its old limit), so it costs ~£2,800 to run for two years. A games-only machine has no large GPU and draws far less: a 16-core desktop ~200-280 W, a mobile-chip board ~130 W.

## How the options are compared

* **Throughput** is estimated from the CPU's PassMark score: steps/s = PassMark × 0.0368, anchored on the 5950X (45,241 points, 1,664 steps/s). It is a proxy, good to perhaps ±20-30%: our load is many single-threaded games with Wine's threads, and SMT, Intel's efficiency cores and AMD's X3D cache may each count differently than in PassMark.
* **Power at the wall** is modelled as (the CPU's package power + 50 W) / 0.9, which fits our machine. The package power is the stock limit (AMD's PPT, Intel's PL2, a server's TDP), not a measurement, except where it says "measured".
* **Two years** of ownership: the purchase plus electricity day and night at 26.32p, with nothing back from a resale. A rental includes its electricity. Rental prices include 20% VAT.

## Two years, per option

| option | upfront | steps/s (× ours) | at the wall | electricity, 2 years | total, 2 years (at 30p) | steps/s per £1,000 |
|---|---|---|---|---|---|---|
| **Minisforum BD895i SE** board (Ryzen 9 8945HX, 16 cores) built up | £999 | ~1,890 (1.13×) | 133 W, measured | £613 | **£1,613** (£1,698) | **1,170** |
| Minisforum BD795i SE board (Ryzen 9 7945HX: the same chip as the 8945HX, renamed) built up, at its sale price* | ~£969 | ~1,890 | ~133 W (as the BD895i SE) | £613 | ~£1,582 | ~1,195 |
| **Minisforum MS-A2 (8945HX), 32 GB, 1 TB, complete**** | **£1,015** | ~1,890 | ~130 W | £599 | **£1,614** | **1,171** |
| Minisforum MS-A2 (Ryzen 7 7745HX: 8 cores, PassMark ~32,800), 32 GB, 1 TB, complete** | £895 | ~1,210 | ~100 W (estimate) | ~£461 | ~£1,356 | ~890 |
| Minisforum MS-A2 barebone with the newer Ryzen 9 9955HX (Zen 5), at its sale price*, + 32 GB | ~£1,326 | ~2,050 | ~140 W (estimate) | ~£645 | ~£1,971 | ~1,040 |
| **Core Ultra 7 270K Plus** desktop at 125 W (assumed ~85% of its stock speed) | £1,023 | ~2,140 (1.29×) | ~194 W | £897 | **£1,920** (£2,045) | **1,116** |
| used Ryzen 9 7950X in Eco mode (105 W), new AM5 parts | £1,017 | ~2,150 | ~213 W | £984 | £2,001 | 1,074 |
| Minisforum M1-1390 (i9-13900HX) + 32 GB | £896 | ~1,520 | ~120 W | £553 | £1,450 | 1,050 |
| Ryzen 9 9950X in Eco mode (assumed ~93% of its stock speed) | £1,222 | ~2,250 (1.35×) | ~213 W | £984 | £2,206 | 1,019 |
| Core Ultra 7 270K Plus, stock | £1,023 | ~2,520 (1.51×) | ~333 W | £1,537 | £2,560 | 985 |
| Ryzen 9 9950X, stock | £1,222 | ~2,420 (1.45×) | ~278 W | £1,281 | £2,503 | 966 |
| used EPYC 7702 (64 cores) build | ~£1,360 | ~2,530 | ~300 W | £1,383 | £2,743 | 923 |
| Ryzen 9 5900XT on AM4 (about a copy of our machine) | £822 | ~1,620 (0.97×) | ~213 W | £984 | £1,806 | 894 |
| **Hetzner EX44** rented (i5-13500, 64 GB) | £59 a month | ~1,130 (0.68×) | | included | **£1,424** | **793** |
| Hetzner EX63 rented (Core Ultra 7 265) | £100 a month | ~1,780 | | included | £2,464 | 723 |
| used Dell R640, 2× Xeon Gold 6138 | £732 | ~1,500 | ~370 W | £1,706 | £2,438 | 616 |
| Hetzner AX102 / AX162 rented | £160 / £321 a month | ~2,290 / ~3,490 | | included | £3,967 / £8,005 | 578 / 435 |
| AWS c8i.48xlarge / c7a.16xlarge, London, spot | £864 / £909 a month | ~5,900 / ~2,430 | | included, egress extra | £20.7k / £21.8k | 284 / 111 |

\*\* The prices the user saw on minisforum.uk on 24 October 2026 (the research had £519 for the 8945HX barebone and our own memory and drive: £1,046).

\* From search snapshots of minisforum.uk (its store refused more requests that day): the BD795i SE £359 (regular £559), the MS-A2 with the 9955HX £799 (regular £989), the BD790i X3D (7945HX3D) £479 but sold out. The research had priced seven Minisforum products and missed these three; there is no Minisforum ITX board with the 9955HX yet.

The first six are within ~15% of each other, inside the proxy's error: a benchmark of our own games should choose among them (Hetzner rents by the hour, €0.0945 for an EX44, a cheap way to measure an Intel hybrid CPU).

## What to buy, or rent

1. **A Minisforum MS-A2 with the Ryzen 9 8945HX, 32 GB and 1 TB, £1,015 complete.** The same 16 Zen 4 cores as the boards below, with its case, power supply and cooling, for about what building one up costs: ~£1,614 over two years for ~1,890 steps/s. (The £895 version has the Ryzen 7 7745HX: 8 cores, ~1,210 steps/s, worse value.)
1. **A Minisforum BD895i SE (£389) or BD795i SE (£359 on sale; the 7945HX is the same chip a generation earlier) built into a small box, ~£970-999.** The board has its CPU soldered on; the £999 adds 32 GB of DDR5 SODIMM (£438, 2×16 GB), a 550 W power supply (£45), a case (£30), a 500 GB NVMe drive (£89) and a fan (£8, the heatsink assumed in the box): the memory is ~44% of it. A Mini-ITX board with a 16-core Ryzen 9 8945HX laptop chip, measured at ~133 W at the wall under full load and ~90% of a stock 7950X's Cinebench score at 100 W. Two years: ~£1,613 for ~1,890 steps/s, about our machine's games again. Quiet; it would sit on the LAN. The MS-A2 barebone (£519) is the same chip ready-made, at about the same total.
2. **A Core Ultra 7 270K Plus (£310) desktop with a 125 W power limit, ~£1,023.** Two years: ~£1,920 for ~2,140 steps/s (the 85% at 125 W is an estimate). Without the limit ~2,520 steps/s for ~£2,560. AMD's alternative is a 9950X in Eco mode (~£2,206 for ~2,250).
3. **Rent a Hetzner EX44 (£59 a month, no setup fee, unlimited traffic).** Two years: ~£1,424 for ~1,130 steps/s: ~1.45 times the cost of owning, before resale. Capacity at once, monthly cancellation, no heat or noise at home. From the UK a round trip is ~20-40 ms, so a rented machine would run inference itself, on its CPU, not call our 3090.

A resale after two years (30-50%, £300-500 a machine) makes owning look better still.

Not worth it:

* **Laptops**: 1.5-2 times the price per point of a desktop, a GPU we do not need, and throttling under a sustained load (a Legion Pro 5 with a 9955HX: £2,250).
* **AWS and GCP**: 4-10 times the cost of owning, and ~7.8 TB a month of egress at 3 MB/s would add ~£500-700 a month.
* **Used dual-Xeon servers**: cheap to buy, but ~£1,700-1,950 of electricity over two years, and loud.
* **Threadripper**: registered DDR5 is ~£800 per 16 GB; a 64 GB 7960X build passes £5,000.
* **8-core mini PCs**: ~£900 for 29k PassMark once memory is added.

## Memory: same-page merging

A game's ~346 MB is almost all private heap, but the games load the same map, tables and code, so much of it is the same page for page. Linux merges identical pages of processes that opt in (KSM: `prctl(PR_SET_MEMORY_MERGE)`, inherited by the processes they start, and `/sys/kernel/mm/ksm/run`). A test on 2026-10-24: 4 games on Echo Isles, a minute into the game, opted in:

| | |
|---|---|
| memory saved | **1,014 MB** of the four games' ~1,380 MB |
| merged, per game | ~225 MB of ~345 MB |
| the merged pages' single copies | 93 MB |
| ksmd's CPU | ~8% of a core (1,000 pages each 20 ms) |

So a game would cost ~115 MB of its own plus a share of ~93 MB, and 55 games ~6.5 GB instead of ~19 GB. Not yet measured: how far pages drift apart over a 10-40 minute game and across reloads, Wine's helper processes and the actors opted in too, and ksmd's cost scanning ~20 GB. If it holds, a games box with 16 GB would run as many games as its CPU allows, ~£200 less on today's memory prices. The training machine does not need it (60 GB, ~42 GB used).

## Prices that move

* **Memory**: DDR5 is ~£14-17 per GB, ~3.7 times a year ago (and +15% in the last 90 days). 32 GB is ~£440, 35-45% of a new machine. A board with a laptop chip takes SODIMM at about the same price.
* **Electricity**: about +23-25% forecast for January 2027, and VAT back on 1 April 2027 (the 30p column). A fixed tariff at about the cap would hedge it.
* **The 3090**: each 100 W off its limit saves ~£461 over two years, but only if the learner does not need the speed; it is at 400 W by choice ([GPU unlock](gpu-unlock.md)).

## Before buying

* The games' transport to another machine (trajectories, weights, inference) has to be built for training games; evaluation games work today.
* A benchmark of the candidate CPU on our games (`scripts/bench_env.py`) is worth more than the PassMark proxy, above all on Intel's hybrid chips.
* Used parts: many cheap EPYCs are locked to one vendor's boards.

## Sources

Fetched on 24 October 2026 unless dated.

* Electricity: [Ofgem's price cap](https://ofgem.gov.uk/your-energy-supply/your-energy-bill/energy-price-cap-unit-rates-and-standing-charges); [Uswitch](https://milled.com/uswitch/dont-miss-the-nations-cheapest-fix-LQSP2bwk3guinNL4) (~19 October); [Octopus](https://octopus.energy/blog/october-2026-energy-price-cap-should-i-fix-my-tariff/) (26 August).
* Exchange rates (Bank of England, 22 October 2026: £1 = €1.1934 = $1.3363): [poundsterlinglive.com](https://www.poundsterlinglive.com/bank-of-england-spot/historical-spot-exchange-rates/gbp/GBP-to-EUR-2026).
* PassMark: [high-end CPUs](https://www.cpubenchmark.net/high_end_cpus.html), [mid-range CPUs](https://www.cpubenchmark.net/mid_range_cpus.html), [multi-CPU systems](https://www.cpubenchmark.net/multi_cpu.html).
* Parts: Scan's [AMD CPUs](https://www.scan.co.uk/shop/computer-hardware/cpu-amd-desktop/all), [Intel CPUs](https://www.scan.co.uk/shop/computer-hardware/cpu-intel-desktop/all), [AMD boards](https://www.scan.co.uk/shop/computer-hardware/motherboards-amd/all), [Intel boards](https://www.scan.co.uk/shop/computer-hardware/motherboards-intel/all), [memory](https://www.scan.co.uk/shop/computer-hardware/memory-ram/all), [power supplies](https://www.scan.co.uk/shop/computer-hardware/power-supplies/all), [laptops](https://www.scan.co.uk/search?q=9955HX); [Overclockers](https://www.overclockers.co.uk/search?sSearch=7950X); [ramprices.uk](https://www.ramprices.uk/ram-price-index).
* Power: [WePC on the 9950X](https://www.wepc.com/cpu/review/ryzen-9-9950x/); [Phoronix on the 270K Plus](https://www.phoronix.com/review/intel-core-ultra-7-270k-plus/15); [techtest.org on the BD895i SE](https://techtest.org/minisforum-bd895i-se-im-test-itx-mainboard-mit-ryzen-9-8945hx-und-desktop-power/).
* Mini PCs: Minisforum UK's [BD895i SE](https://www.minisforum.uk/products/bd895i-se-motherboard), [MS-A2](https://www.minisforum.uk/products/minisforum-ms-a2), [M1-1390](https://www.minisforum.uk/products/minisforum-m1-1295), [UM890 Pro](https://www.minisforum.uk/products/minisforum-um890pro).
* Used: eBay UK asking prices from search snippets (undated): [5950X](https://www.ebay.co.uk/itm/144568508325), [7950X](https://www.ebay.co.uk/itm/176824041711), [R640](https://www.ebay.co.uk/itm/126941176666); Bargain Hardware's [EPYCs](http://web.archive.org/web/20260202043409/https://www.bargainhardware.co.uk/components/processors-cpu/amd-epyc) (Wayback, 2 February 2026).
* Rental: Hetzner's [AX](https://www.hetzner.com/dedicated-rootserver/matrix-ax/) and [EX](https://www.hetzner.com/dedicated-rootserver/matrix-ex/) servers, [cloud](https://www.hetzner.com/cloud/general-purpose/) and [VAT](https://docs.hetzner.com/general/billing-and-account-management/billing-at-hetzner/value-added-tax/); [OVHcloud](https://eco.ovhcloud.com/en-gb/) (its RISE servers were "soon available", not orderable); [netcup](https://www.netcup.com/en/server/root-server); AWS London [on-demand](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/EU%20(London)/Linux/index.json) and [spot](https://instances.vantage.sh/instances.json) prices; [GCP europe-west2](https://gcloud-compute.com/europe-west2.html) (data of 18 October).
