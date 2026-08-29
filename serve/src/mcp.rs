//! Model Context Protocol server over stdio: JSON-RPC 2.0, one message per line.
//!
//! The tools are the same operations the HTTP API exposes, described with JSON Schema so a caller
//! can use them without being told how.

use crate::api;
use crate::json::{parse, write, Json};
use std::io::{BufRead, Write as _};

/// The protocol revision this server implements. A client asking for a different one is answered
/// with theirs when we can honour it, which is what the specification asks for.
pub const PROTOCOL: &str = "2025-06-18";
const SUPPORTED: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

fn schema_graph() -> Json {
    Json::obj(vec![
        ("type", Json::s("object")),
        (
            "description",
            Json::s(
                "Either a named family or an explicit edge list. Named: \
                 {\"builtin\":\"lattice2d\",\"l\":32,\"j\":1.0} or \
                 {\"builtin\":\"ring\",\"n\":16,\"j\":1.0,\"h\":0.0}. Explicit: \
                 {\"n\":4,\"couplings\":[[0,1,1.0]],\"biases\":[[0,0.5]]}. States are -1/+1 and \
                 energy is -sum_ij J_ij s_i s_j - sum_i h_i s_i.",
            ),
        ),
    ])
}

fn prop(t: &str, desc: &str) -> Json {
    Json::obj(vec![("type", Json::s(t)), ("description", Json::s(desc))])
}

fn tool(name: &str, desc: &str, props: Vec<(&str, Json)>, required: Vec<&str>) -> Json {
    Json::obj(vec![
        ("name", Json::s(name)),
        ("description", Json::s(desc)),
        (
            "inputSchema",
            Json::obj(vec![
                ("type", Json::s("object")),
                ("properties", Json::obj(props)),
                ("required", Json::Arr(required.into_iter().map(Json::s).collect())),
            ]),
        ),
    ])
}

pub fn tools() -> Json {
    Json::Arr(vec![
        tool(
            "ferrotherm_sample",
            "Draw a state from the Boltzmann distribution of an Ising graph by chromatic \
             block-Gibbs sampling. Returns the state, its energy, magnetization, an energy ledger \
             priced at Z1-class device figures, and a CERTIFICATE computed from the samples \
             themselves: the temperature actually sampled at with a confidence interval, the \
             autocorrelation time and effective sample size, and where the model is small enough, \
             the distance from the exact Boltzmann distribution beside the sampling-noise floor. \
             An empty findings list is the only thing that means the run is sound. Deterministic \
             given seed and thread count.",
            vec![
                ("graph", schema_graph()),
                ("beta", prop("number", "Inverse temperature, 1/T. Higher is colder and more ordered. Default 1.0.")),
                ("sweeps", prop("integer", "Burn-in sweeps before recording. Default 100.")),
                ("draws", prop("integer", "Samples recorded after burn-in, used to certify the run. Default 128.")),
                ("thin", prop("integer", "Sweeps between recorded draws. Raise this if the certificate reports correlated draws. Default 1.")),
                ("seed", prop("integer", "Random seed. Default 0.")),
                ("threads", prop("integer", "Parallel threads. Results stay reproducible per thread count. Default 1.")),
                ("clamp", prop("array", "Nodes held fixed, as [[index, -1 or +1], ...].")),
                ("return_state", prop("boolean", "Include the full state array. Defaults to true for graphs of 4096 nodes or fewer.")),
            ],
            vec!["graph"],
        ),
        tool(
            "ferrotherm_anneal",
            "Minimise the energy of an Ising graph by simulated annealing down a geometric beta \
             ladder, returning the lowest-energy state found. Use this for optimisation problems \
             encoded as couplings.",
            vec![
                ("graph", schema_graph()),
                ("beta_min", prop("number", "Starting (hot) inverse temperature. Default 0.1.")),
                ("beta_max", prop("number", "Final (cold) inverse temperature. Default 3.0.")),
                ("stages", prop("integer", "Rungs on the ladder. Default 24.")),
                ("sweeps_per_stage", prop("integer", "Sweeps at each rung. Default 20.")),
                ("seed", prop("integer", "Random seed. Default 0.")),
                ("return_state", prop("boolean", "Include the full state array.")),
            ],
            vec!["graph"],
        ),
        tool(
            "ferrotherm_energy",
            "Compute the energy and magnetization of a specific state under a specific graph. Use \
             it to score a candidate solution.",
            vec![
                ("graph", schema_graph()),
                ("state", prop("array", "One entry per node, each -1 or +1.")),
            ],
            vec!["graph", "state"],
        ),
        tool(
            "ferrotherm_verify",
            "Check the sampler against ground truth: enumerate the exact Boltzmann distribution \
             and report the total variation distance from the sampler's empirical one. Capped at \
             20 nodes because enumeration is 2^n. Compare the result against the reported sampling \
             noise floor rather than against zero.",
            vec![
                ("graph", schema_graph()),
                ("beta", prop("number", "Inverse temperature. Default 1.0.")),
                ("draws", prop("integer", "Samples to histogram. Default 20000.")),
                ("sweeps", prop("integer", "Burn-in sweeps before sampling. Default 200.")),
                ("thin", prop("integer", "Sweeps between recorded draws. Default 1. Raise this when the reported TV sits above the noise floor: consecutive sweeps are correlated, and thinning is what buys independent draws.")),
                ("seed", prop("integer", "Random seed. Default 0.")),
            ],
            vec!["graph"],
        ),
        tool(
            "ferrotherm_bound",
            "How far from optimal is this answer? Returns LOWER BOUNDS on the ground energy, and \
             if you pass a state, the gap between it and the best bound. Unlike ferrotherm_verify \
             this works at ANY size, and it answers a different question: verify checks the \
             sampler's DISTRIBUTION, this bounds the OPTIMUM. A gap of zero means the state is \
             proved optimal without trusting whatever produced it; anything above is an upper \
             limit on what a better search could still win. All bounds are sound, so \"best\" is \
             their maximum -- they disagree by a lot and in both directions.",
            vec![
                ("graph", schema_graph()),
                ("state", prop("array", "Optional. One entry per node, each -1 or +1. Supplying it adds \"energy\" and \"gap\" to the reply, which is the point of the tool.")),
                ("forest_rounds", prop("integer", "Subgradient rounds for the forest bound. Default 40. Worth nothing on a graph with no fields: a tree is never frustrated, so the bound degenerates to the trivial floor.")),
                ("max_cycle", prop("integer", "Longest frustrated cycle to charge for. Default 6. This is the bound that sees frustration, which is the only thing that makes max-cut hard.")),
                ("sdp_sweeps", prop("integer", "Mixing sweeps for the semidefinite bound. Default 200. Raising it buys almost nothing -- measured 12223 / 12224 / 12224 at 200 / 1000 / 4000 sweeps on G1.")),
                ("seed", prop("integer", "Random seed for the semidefinite search. Default 1. It only chooses WHICH dual point gets verified; a bad choice loosens the bound and cannot make it wrong.")),
            ],
            vec!["graph"],
        ),
        tool(
            "ferrotherm_exact_planar",
            "EXACT max-cut on a PLANAR graph, in polynomial time. The only tool here that returns \
             an optimum rather than an attempt: there is no budget, no seed, and no incumbent, and \
             the same request always returns the same answer because there is only one. Max-cut is \
             NP-hard in general and polynomial on a planar graph, and the difference is a theorem \
             rather than an engineering margin -- a cut in the graph is a cycle in the dual, so the \
             problem becomes a minimum-weight T-join and then a minimum-weight perfect matching. \
             Reach for this FIRST whenever the graph is planar: a grid with open boundaries, a road \
             network, a circuit layout. It refuses, with the reason, on a graph carrying fields, on \
             a non-planar graph (a PERIODIC lattice is a torus and is not planar), on a graph with \
             a cut vertex, and on weights that do not scale to integers -- four different things to \
             do next.",
            vec![
                ("graph", schema_graph()),
                ("scale", prop("number", "Multiply every coupling by this before rounding to an integer. Default 1.0, which is right for whole-number couplings. The matching underneath is exact only in exact arithmetic, so a weight that does not land on an integer is refused rather than rounded -- rounding here moves the optimum, not the last digit.")),
                ("return_state", prop("boolean", "Include the optimal partition in the reply. Defaults to true under 4096 nodes.")),
            ],
            vec!["graph"],
        ),
        tool(
            "ferrotherm_toroidal_bound",
            "An UPPER BOUND on the maximum cut of a TOROIDAL GRID -- the side of the benchmark \
             table nobody publishes. Every G-set figure is a best cut FOUND, which is a LOWER \
             bound; this is the other end of the bracket, so together they say where the optimum \
             actually is. A torus is not a plane, so ferrotherm_exact_planar refuses it -- but the \
             dual argument needs only faces, and an embedding on any surface has them. On a torus \
             the cycle space of the dual is four times the cut space, so the relaxation ranges over \
             sets that are not cuts and its optimum bounds the maximum from above. Measured, this \
             closes the bracket on G11 and proves its best-known cut of 564 optimal. Check \
             \"attained\": true means the bound is itself achieved by a cut and so IS the maximum, \
             proved; false leaves it a bound, which still stands. Refuses unless the graph really \
             is a toroidal grid -- the structure is recovered from the edge list, a match on all 2n \
             edges, rather than assumed.",
            vec![
                ("graph", schema_graph()),
                ("scale", prop("number", "Multiply every coupling by this before rounding to an integer. Default 1.0.")),
                ("return_state", prop("boolean", "Include the partition, when the bound is attained and there is one. Defaults to true under 4096 nodes.")),
            ],
            vec!["graph"],
        ),
        tool(
            "ferrotherm_optimize",
            "Minimise an Ising graph by a named method. Prefer ferrotherm_anneal for a quick answer; \
             reach for this when you need more than one. \"tabu\" remembers where it has been and \
             escapes local minima a descent gets stuck in. \"population\" runs many chains with \
             resampling and reports rho, a diagnostic that says whether to believe its own free \
             energy. \"branch\" returns a PROOF -- proved_optimal is true only when it exhausted \
             the search tree, and a run that hit its node budget says so rather than claiming one.",
            vec![
                ("graph", schema_graph()),
                ("method", prop("string", "\"tabu\", \"breakout\", \"cluster\", \"quantum\", \"goemans_williamson\", \"population\" or \"branch\". Default \"tabu\". \"cluster\" is parallel tempering with ISOENERGETIC CLUSTER MOVES, the baseline the Ising-machine literature measures against -- it flips a whole connected component of the disagreement between two replicas at once, at zero energy cost, and is refused on a graph with fields because the argument holds only at h = 0. \"quantum\" is simulated quantum annealing (path-integral Monte Carlo on the transverse-field model); it is NOT a quantum computer and returns a classical state, and trotter=1 is exactly classical annealing, which is the honest control. \"goemans_williamson\" rounds the semidefinite relaxation and is THE ONLY WORST-CASE GUARANTEE in max-cut -- but check \"guaranteed\" in the reply, because the 0.87856 ratio needs non-negative edge weights and is false on most real instances. \"breakout\" is breakout local search, the algorithm that holds the max-cut record on most of the G-set benchmark: steepest descent with an adaptive perturbation between local optima. One iteration is one spin flip for both \"tabu\" and \"breakout\", so giving them the same number is a matched-budget comparison. Check \"descents\" in the reply -- a run with a handful of them spent its budget inside one basin and is a descent, not a breakout search.")),
                ("seed", prop("integer", "Random seed. Default 0.")),
                ("iterations", prop("integer", "tabu and breakout: flips to run. Default 50000. Check \"iterations_run\" in the reply -- a shorter run was truncated.")),
                ("tenure", prop("integer", "tabu: how many iterations a flipped spin stays forbidden. Default 0, which scales it to the graph.")),
                ("restart_after", prop("integer", "tabu: restart from a fresh state after this many iterations with no improvement. Default 5000; 0 never restarts.")),
                ("population", prop("integer", "population: number of chains. Default 1000. rho is bounded by this, and rho near it means the population collapsed onto one ancestor.")),
                ("sweeps", prop("integer", "population: Gibbs sweeps per replica per rung. Default 4.")),
                ("stages", prop("integer", "population: ladder rungs from beta = 0. Default 100.")),
                ("beta_max", prop("number", "population: coldest inverse temperature. Default 6.0.")),
                ("rungs", prop("integer", "cluster: ladder rungs. Default 16. Two ladders are run, because a cluster move needs a partner at the same temperature.")),
                ("rounds", prop("integer", "cluster: rounds of sweeps. Default 400. Check \"cluster_moves\" in the reply -- zero means the replicas never disagreed and the move did nothing.")),
                ("trotter", prop("integer", "quantum: Trotter slices. Default 4. ONE is classical annealing, not a degenerate case -- use it as the control.")),
                ("gamma_max", prop("number", "quantum: transverse field at the start. Default 3.0.")),
                ("gamma_min", prop("number", "quantum: transverse field at the end. Default 0.05, and NOT zero: the Trotter coupling diverges there.")),
                ("steps", prop("integer", "quantum: anneal steps. Default 200.")),
                ("hyperplanes", prop("integer", "goemans_williamson: random hyperplanes to draw, best kept. Default 64. The guarantee is about the expectation of ONE draw, so keeping the best can only help and does not extend the hypothesis.")),
                ("max_nodes", prop("integer", "branch: node budget. Default 20000000. Running out means no proof, which the reply reports.")),
                ("sdp_depth", prop("integer", "branch: spend a certified semidefinite bound on the residual problem at every node down to this depth, instead of the cheap incremental one. Off by default. It buys a much tighter bound where the residual is still large and the nodes are still few -- the top of the tree -- and costs a Cholesky per node to do it. Whether that pays is instance-dependent: the reply reports sdp_prunes over sdp_calls, and a ratio of zero means every Cholesky was wasted.")),
                ("incumbent", prop("array", "A starting state, one entry per node, each -1 or +1. Taken by branch, tabu and breakout. For branch it prunes from the first node and is worth far more than a better bound; for the other two it is a warm start, so a good state from an earlier anneal or solve is built on rather than thrown away. Measured on a 2D glass: tabu from a short anneal was better on 7 of 12 seeds at 784 spins and worse on none, while breakout was mixed -- it calibrates its jump against how long it has stalled, so a deep starting basin changes the schedule it thinks it is on. A wrong length is refused by name rather than ignored.")),
                ("return_state", prop("boolean", "Include the state in the reply. Defaults to true under 4096 nodes.")),
            ],
            vec!["graph"],
        ),
        tool(
            "ferrotherm_solve",
            "State a constraint or optimisation problem in its own vocabulary and get NAMED VALUES \
             back. Prefer this over ferrotherm_anneal: that one wants a graph of spins and returns \
             an array of +/-1, which means computing spin indices yourself to say something as \
             simple as \"these two must differ\". Declare variables with domains, state \
             constraints, optionally give an objective, and read the answer by name. \
             The reply's \"objective\" is what the answer is WORTH, in your own units and the \
             direction you wrote it: read that to compare two answers or to say what a schedule is \
             worth. \"energy\" is a different number -- the compiled Ising energy with every \
             penalty and the constant folded in -- which orders two answers to ONE model and means \
             nothing across models or after a penalty change. Check \
             `feasible` before trusting the values. False means the answer breaks something you \
             asked for, and the reply says which: a variable in \"did_not_decode\", or a \
             HARD constraint in \"violated\" that the objective outbid. A penalty makes a \
             constraint expensive, not impossible -- so raise \"penalty\" and try again, and only \
             conclude the problem has no answer if it stays infeasible at a large one. Entries with \
             \"hard\": false are preferences you priced with \"soft\" and do NOT make the answer \
             infeasible; \"soft_cost\" totals what those trades cost. The reply also carries \"caveats\": things the compiler KNOWS are wrong with the model and cannot fix, today meaning an encoding no penalty can make exact (a binary encoding of k values spells 2^ceil(log2 k) codewords, and when k is not a power of two the spare ones decode to nothing while costing exactly what a valid state costs). An empty list is the normal case; a non-empty one means a value that reads back fine may have come from a codeword the penalty never excluded, so report it rather than trusting the number.",
            vec![
                ("variables", prop("array", "Each is {\"name\": \"x\", \"values\": 3} for a categorical, or {\"name\": \"t\", \"lo\": 0, \"hi\": 9} for an integer. Everywhere a \"value\" appears below it is the variable's OWN value, never a slot index: for an integer over 10..20, write 13 to mean 13. A value outside the range is an error naming the range, not a silent reinterpretation. Either kind may carry \"encoding\": how it is STORED -- \"one-hot\" (default, exact, k spins), \"domain-wall\" (exact, k-1 spins, cheaper for large k), or \"binary\" (ceil(log2 k) spins, and NOT exact unless k is a power of two: the spare codewords decode to nothing and cost exactly what a valid state costs, which is reported in \"caveats\"). An unknown name is refused rather than defaulted.")),
                ("constraints", prop("array", "Each is {\"type\": \"not_equal\"|\"equal\", \"a\": name, \"b\": name}, {\"type\": \"fix\", \"var\": name, \"value\": v}, or {\"type\": \"cardinality\"|\"at_most\"|\"at_least\", \"k\": n, \"of\": [{\"var\": name, \"value\": v}, ...]}, or {\"type\": \"exactly_one\"|\"at_most_one\", \"of\": [...]}, or {\"type\": \"all_different\", \"of\": [{\"var\": name}, ...]} which makes every named VARIABLE take a different value -- the workhorse of assignment, scheduling, colouring and puzzles. all_different lowers per shared value rather than per pair, so it needs no slack, costs nothing where two domains do not overlap, and its violation names which value collided and who took it; asking it of more variables than the values they share is refused when the model compiles, because that has no answer at any penalty. Cardinality is exactly k; at_most and at_least are inequalities and cost extra spins, because an inequality needs a slack variable to become an equality the sampler can square. exactly_one and at_most_one lower pairwise instead and need no slack, so prefer them over k=1. \"of\" takes any number of entries, and each may name a different variable AND a different value. ANY constraint may carry \"soft\": w, which makes it a PREFERENCE priced at w rather than a rule: the solver may trade it away, the answer stays feasible, and the reply reports it under \"violated\" with \"hard\": false plus a \"cost\". Use soft for what you would LIKE, hard for what an answer must satisfy to be an answer at all. The price is w x amount SQUARED, so missing by two costs four times missing by one, and unlike \"penalty\" it is absolute rather than scaled against the objective -- that is the point, since a preference is meant to be traded against it.")),
                ("objective", prop("object", "{\"maximize\": true, \"terms\": [{\"var\": name, \"value\": v, \"weight\": w}]}. A term with \"and_var\"/\"and_value\" rewards two variables taking values together, and a term with \"of\": [{\"var\": name, \"value\": v}, ...] rewards ANY number of them holding together. Three or more is a higher-order term: it is lowered with an extra spin per substituted pair, reported as \"ancillas\" in the reply, and its guarantee is about finding optima rather than about sampling. \"maximize\" takes true/false or 1/0; anything else is refused rather than read as false.")),
                ("method", prop("string", "Which solver to use. \"anneal\" (default) is simulated annealing down a ladder. \"tabu\" is tabu search, the strongest single arm in this stack's own matched-budget shootout. \"breakout\" is breakout local search. \"branch\" is branch and bound and is THE ONLY ONE THAT RETURNS A PROOF: it sets \"proved_optimal\" in the reply when it exhausts the tree. Read that together with \"feasible\" -- branch proves a statement about the compiled energy, and it becomes a statement about YOUR model exactly when the answer is also feasible, because a feasible assignment pays no penalty and its compiled energy is the objective plus a constant. Proved AND feasible is a real optimality proof and needs nothing from the penalty being large enough; proved and INFEASIBLE means the penalty is too small and no longer search will fix it.")),
                ("effort", prop("integer", "The chosen method's budget: iterations for tabu and breakout, a node ceiling for branch. 0 takes a default. Ignored by \"anneal\", which uses \"tries\" and \"schedule\" instead.")),
                ("tries", prop("integer", "Independent annealing restarts; the best feasible answer wins. Default 16. Applies to \"anneal\" only -- a deterministic search and a proof do not restart.")),
                ("penalty", prop("number", "Constraint strength. Omit it: by default it scales to twice the largest objective weight, because a constraint that ties with an objective gets traded away. Raise it when \"violated\" comes back non-empty.")),
                ("schedule", prop("object", "The annealing ladder: {\"beta_hot\": 0.05, \"beta_cold\": 8.0, \"stages\": 120, \"sweeps\": 40}, which are the defaults. Lengthen it -- more stages, more sweeps -- when a model stays infeasible at a large penalty: that is a model not being annealed long enough, and no penalty fixes it. Cooling runs hot to cold, so beta_cold must exceed beta_hot.")),
            ],
            vec!["variables"],
        ),
        tool(
            "ferrotherm_fit",
            "FIT a Boltzmann machine to data, and get the trained model back as a graph you can \
             feed straight to any other tool here. Every other tool takes a model as given -- this \
             is the one that PRODUCES one, which is what makes this a computing paradigm rather \
             than a solver behind an API: the argument for thermodynamic hardware is that it \
             samples Boltzmann distributions cheaply, and the distributions anyone actually wants \
             are fitted to data. The reply's \"graph\" is in exactly the shape ferrotherm_sample, \
             ferrotherm_anneal, ferrotherm_bound and ferrotherm_verify take, so there is no export \
             step and no second format. \
             READ \"learned_percent\", NOT the raw likelihood. A log-likelihood of -2.79 says \
             nothing on its own; the same number placed 95.8% of the way from an untrained model \
             to a perfect one says everything, and both ends need no calibration -- an untrained \
             machine is uniform over 2^visible images and scores exactly -visible*ln2, and a \
             perfect one is uniform over the rows you gave and scores -ln(rows). \
             The likelihood is EXACT, by enumeration, and comes back null above 22 spins, where \
             this REFUSES rather than substituting a cheaper estimate. That refusal is deliberate: \
             an ELBO, a reconstruction error and a pseudo-likelihood are each worst exactly where \
             sampling is worst, so a caller comparing two machines on one would be reading the \
             proxy's failure and calling it expressivity. A fit whose likelihood is null still \
             produced a real model -- only its quality is unmeasured. \
             ON SHAPE: \"hidden\": [12] is a restricted Boltzmann machine, where every latent unit \
             touches every visible one. \"hidden\": [6, 6] is a two-layer deep machine with the \
             same twelve latents and FEWER EDGES. Measured in this repository, deeper arrangements \
             of the same latent count learn strictly LESS at every count tried -- 96.3% wide \
             against 93.1% and 65.5% deep at twelve latents -- so prefer one wide layer unless you \
             have a reason not to.",
            vec![
                ("visible", prop("integer", "How many visible units the machine has. Must equal the width of a data row, because a row is CLAMPED onto them: the first `visible` spins hold the data and the rest are latent.")),
                ("hidden", prop("array", "Layer widths, outermost last. [12] is an RBM; [6, 6] is two layers of six. Every layer is fully connected to the one below it and to nothing else.")),
                ("data", prop("array", "Rows of -1 and +1, all the same width. Or the string \"bars-and-stripes-N\" for the standard tiny benchmark at N x N, which is what to reach for when you want to see the operation work before wiring your own data in: \"bars-and-stripes-3\" is 14 images over 9 visible units.")),
                ("epochs", prop("integer", "Passes over the data. Default 300.")),
                ("k", prop("integer", "k in CD-k: negative-phase sweeps started from the positive-phase state. Default 5. One is the biased extreme and larger is closer to the true gradient at proportional cost -- which is why the reply reports the EXACT likelihood rather than a training loss, since the training objective is biased by construction and cannot tell you it went wrong.")),
                ("positive_sweeps", prop("integer", "Sweeps used to settle the latent units with the visible ones clamped to a data row. Default 5.")),
                ("learning_rate", prop("number", "Starting step, default 0.05. It DECAYS to a tenth of this over the run, and that decay is not a refinement: the gradient's model term is one sample per row, so without it the weights random-walk around the optimum forever and the fit sits a constant distance from its own fixed point however long you run.")),
                ("batch", prop("integer", "Rows per gradient step. Default 8.")),
                ("seed", prop("integer", "Same seed reproduces the fit exactly. Default 1.")),
            ],
            vec!["visible", "hidden", "data"],
        ),
        tool(
            "ferrotherm_hubo",
            "Minimise a HIGHER-ORDER model natively: terms over any number of variables, with no \
             ancillas and no penalty weight to get right. Reach for this INSTEAD of a three-or-more \
             variable objective term in ferrotherm_solve whenever the model is genuinely \
             higher-order and you are running on a CPU. Both paths work and they are not \
             equivalent: ferrotherm_solve lowers a k-body term through Rosenberg's reduction, \
             adding one ancilla spin per substituted pair plus a penalty chosen as the sum of every \
             coefficient's magnitude -- around 1300 against term weights of 1 -- and that penalty \
             makes the landscape rigid rather than merely larger, because any single flip that \
             would move the search must first pay it. Measured on 60 three-body terms over 40 \
             spins: this path reaches -48.12 at its budget while the reduced path reaches -34.00 at \
             a THOUSAND TIMES that budget, with zero ancilla violations in either, so the reduced \
             search is stuck inside the feasible region rather than escaping it. Use the reduction \
             only when the target really is pairwise hardware, which is what it is for. Note that \
             \"ancillas_avoided\" in the reply is an UPPER BOUND on what a reduction would have \
             spent and not the cost: the reduction shares one ancilla across every term containing \
             the same pair, so on three terms sharing one it spends one where this reports three.",
            vec![
                ("spins", prop("integer", "How many variables the model has. They are numbered 0..spins-1, and every index in every term must be one of them.")),
                ("terms", prop("array", "Each is {\"vars\": [i, j, k, ...], \"weight\": w}, contributing -w times the product of those spins. ANY arity: one variable is a field, two is a coupling, three or more is what this operation exists for. A variable repeated inside one term is refused rather than accepted, because s*s = 1 would silently make the term a different order than the one written -- the refusal names the variable and how many times it appeared.")),
                ("beta_min", prop("number", "Hot end of the annealing ladder. Default 0.05.")),
                ("beta_max", prop("number", "Cold end. Must exceed beta_min; default 8.0.")),
                ("stages", prop("integer", "Ladder steps from hot to cold. Default 200.")),
                ("sweeps_per_stage", prop("integer", "Sweeps at each beta. Default 8.")),
                ("seed", prop("integer", "Same seed reproduces the run exactly. Default 1.")),
            ],
            vec!["spins", "terms"],
        ),
        tool(
            "ferrotherm_capabilities",
            "Describe this server: operations, graph specification, limits, and conventions.",
            vec![],
            vec![],
        ),
    ])
}

fn ok(id: Json, result: Json) -> Json {
    Json::obj(vec![("jsonrpc", Json::s("2.0")), ("id", id), ("result", result)])
}

fn rpc_err(id: Json, code: i32, msg: &str) -> Json {
    Json::obj(vec![
        ("jsonrpc", Json::s("2.0")),
        ("id", id),
        ("error", Json::obj(vec![("code", Json::n(code as f64)), ("message", Json::s(msg))])),
    ])
}

/// A tool result carries its payload as text content; an error sets `isError` rather than failing
/// the call, so the model sees what went wrong and can correct the arguments itself.
fn tool_result(text: String, is_error: bool) -> Json {
    Json::obj(vec![
        (
            "content",
            Json::Arr(vec![Json::obj(vec![("type", Json::s("text")), ("text", Json::Str(text))])]),
        ),
        ("isError", Json::Bool(is_error)),
    ])
}

/// Handle one JSON-RPC message. Returns None for notifications, which take no reply.
pub fn handle(line: &str) -> Option<String> {
    let msg = match parse(line) {
        Ok(m) => m,
        Err(e) => return Some(write(&rpc_err(Json::Null, -32700, &format!("parse error: {e}")))),
    };
    let id = msg.get("id").cloned().unwrap_or(Json::Null);
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(Json::Obj(Vec::new()));

    // Notifications carry no id and must not be answered.
    msg.get("id")?;

    let resp = match method {
        "initialize" => {
            let want = params.get("protocolVersion").and_then(|v| v.as_str()).unwrap_or(PROTOCOL);
            let use_ver = if SUPPORTED.contains(&want) { want } else { PROTOCOL };
            ok(
                id,
                Json::obj(vec![
                    ("protocolVersion", Json::s(use_ver)),
                    ("capabilities", Json::obj(vec![("tools", Json::obj(vec![]))])),
                    (
                        "serverInfo",
                        Json::obj(vec![
                            ("name", Json::s("ferrotherm")),
                            ("version", Json::s(env!("CARGO_PKG_VERSION"))),
                        ]),
                    ),
                    (
                        "instructions",
                        Json::s(
                            "Thermodynamic sampling over Ising graphs, and a modelling layer above \
                             it. Reach for ferrotherm_solve first: state a problem as variables, \
                             constraints and an objective, and read the answer back in your own \
                             names. The others work in spins -- ferrotherm_anneal to optimise a \
                             graph you have already encoded, ferrotherm_sample to draw from a \
                             distribution, ferrotherm_verify to check the sampler against exact \
                             enumeration on small graphs before trusting it on large ones. \
                             Whatever you call, check the result's own verdict field before \
                             trusting its numbers: `feasible` on a solve, `findings` on a verify.",
                        ),
                    ),
                ]),
            )
        }
        "ping" => ok(id, Json::obj(vec![])),
        "tools/list" => ok(id, Json::obj(vec![("tools", tools())])),
        "tools/call" => {
            let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(Json::Obj(Vec::new()));
            let op = name.strip_prefix("ferrotherm_").unwrap_or(name);
            match api::dispatch(op, &args) {
                Ok(r) => ok(id, tool_result(write(&r), false)),
                Err(e) => ok(id, tool_result(e, true)),
            }
        }
        "" => rpc_err(id, -32600, "invalid request: no method"),
        other => rpc_err(id, -32601, &format!("method not found: {other}")),
    };
    Some(write(&resp))
}

/// Read newline-delimited JSON-RPC from stdin until it closes.
pub fn serve() -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(resp) = handle(&line) {
            writeln!(out, "{resp}")?;
            out.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(line: &str) -> Json {
        parse(&handle(line).expect("expected a response")).unwrap()
    }

    #[test]
    fn initialize_reports_tools() {
        let r = call(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#);
        let res = r.get("result").unwrap();
        assert_eq!(res.get("protocolVersion").unwrap().as_str(), Some("2025-06-18"));
        assert!(res.get("capabilities").unwrap().get("tools").is_some());
    }

    #[test]
    fn honours_an_older_client_protocol() {
        let r = call(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#);
        assert_eq!(
            r.get("result").unwrap().get("protocolVersion").unwrap().as_str(),
            Some("2024-11-05")
        );
    }

    #[test]
    fn unknown_protocol_falls_back_to_ours() {
        let r = call(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1999-01-01"}}"#);
        assert_eq!(r.get("result").unwrap().get("protocolVersion").unwrap().as_str(), Some(PROTOCOL));
    }

    #[test]
    fn every_tool_has_a_usable_schema() {
        let r = call(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        let ts = r.get("result").unwrap().get("tools").unwrap().as_arr().unwrap();
        // A FLOOR, not an equality. The exact count was pinned here, so adding a tool failed this
        // test -- which reads as "the new tool is broken" and is fixed by editing the number, a
        // ritual that teaches nothing. What this test is for is that every advertised tool has a
        // usable schema; `every_operation_the_http_api_advertises_has_an_mcp_tool` is what checks
        // the set is complete, and it does so against the API rather than against a constant.
        assert!(ts.len() >= 6, "only {} tools advertised", ts.len());
        for t in ts {
            let name = t.get("name").unwrap().as_str().unwrap();
            assert!(name.starts_with("ferrotherm_"), "{name}");
            assert!(t.get("description").unwrap().as_str().unwrap().len() > 40, "{name} description too thin");
            let s = t.get("inputSchema").unwrap();
            assert_eq!(s.get("type").unwrap().as_str(), Some("object"));
            assert!(s.get("properties").is_some(), "{name} has no properties");
            assert!(s.get("required").is_some(), "{name} has no required list");
        }
    }

    #[test]
    fn tools_call_runs_the_sampler() {
        let r = call(
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ferrotherm_sample",
                "arguments":{"graph":{"builtin":"ring","n":8},"sweeps":20}}}"#,
        );
        let res = r.get("result").unwrap();
        assert_eq!(res.get("isError").unwrap().as_bool(), Some(false));
        let text = res.get("content").unwrap().as_arr().unwrap()[0].get("text").unwrap().as_str().unwrap();
        assert_eq!(parse(text).unwrap().get("nodes").unwrap().as_f64(), Some(8.0));
    }

    #[test]
    fn a_bad_call_is_a_correctable_tool_error_not_a_dead_request() {
        let r = call(
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ferrotherm_sample",
                "arguments":{"graph":{"n":4,"couplings":[[0,99,1.0]]}}}}"#,
        );
        // the model must still get a result it can read and retry from
        let res = r.get("result").unwrap();
        assert_eq!(res.get("isError").unwrap().as_bool(), Some(true));
        assert!(r.get("error").is_none(), "should not be a protocol error");
        let text = res.get("content").unwrap().as_arr().unwrap()[0].get("text").unwrap().as_str().unwrap();
        assert!(text.contains("out of range"), "{text}");
    }

    #[test]
    fn notifications_get_no_reply() {
        assert!(handle(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_none());
    }

    #[test]
    fn malformed_input_is_a_parse_error_not_a_crash() {
        let r = call("{ this is not json");
        assert_eq!(r.get("error").unwrap().get("code").unwrap().as_f64(), Some(-32700.0));
    }

    #[test]
    fn unknown_method_is_32601() {
        let r = call(r#"{"jsonrpc":"2.0","id":9,"method":"resources/list"}"#);
        assert_eq!(r.get("error").unwrap().get("code").unwrap().as_f64(), Some(-32601.0));
    }

    #[test]
    fn tool_names_match_the_dispatch_table() {
        // a tool advertised but not routable is the failure this catches
        let ts = tools();
        for t in ts.as_arr().unwrap() {
            let name = t.get("name").unwrap().as_str().unwrap();
            let op = name.strip_prefix("ferrotherm_").unwrap();
            let e = api::dispatch(op, &Json::Obj(Vec::new()));
            assert!(
                !matches!(&e, Err(m) if m.starts_with("unknown operation")),
                "{name} is advertised but not dispatchable"
            );
        }
    }

    /// And the other direction, which is the one that was not checked.
    ///
    /// The test above catches a tool with no operation behind it. It cannot catch an operation
    /// with no tool in front of it -- and that is the direction this project keeps failing in: a
    /// capability lands in Rust, every surface still compiles, and the thing users cannot say is
    /// invisible to every test. `bound` and `optimize` were dispatchable before they were
    /// advertised, and nothing here would have noticed.
    #[test]
    fn every_operation_the_http_api_advertises_has_an_mcp_tool() {
        let caps = api::capabilities();
        let ops = caps.get("operations").and_then(|o| o.as_arr()).expect("capabilities lists operations");
        let names: Vec<String> = tools()
            .as_arr()
            .unwrap()
            .iter()
            .map(|t| t.get("name").unwrap().as_str().unwrap().to_string())
            .collect();
        for o in ops {
            let op = o.get("name").and_then(|n| n.as_str()).expect("each operation is named");
            // `capabilities` describes the server rather than being an operation over a graph.
            if op == "capabilities" {
                continue;
            }
            let want = format!("ferrotherm_{op}");
            assert!(
                names.contains(&want),
                "the HTTP API offers {op:?} and MCP does not advertise {want:?}; the tools are \
                 documented as the same operations"
            );
        }
    }
}
