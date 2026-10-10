# =============================================================================
# stress_test_lowrank.jl — Newton vs Adam vs hybrid for lowrank_glm.jl
#
#   julia -t auto --project=. src/stress_test_lowrank.jl [options]   (from the repo root)
#     --sizes small,medium,large     small = 2000×500 dense, medium = 20000×2000 dense,
#                                    large = 100000×2000 sparse (Int32 indices)
#     --ks 5,20,50   --families poisson,bernoulli,zinb,nb   --solvers newton,adam,hybrid
#     --time-limit 600   (s per fit; fits predicted to exceed it are skipped)
#     --out stress_results.csv       (rows are appended; delete the file to start fresh)
#     --summary stress_summary.md    (regenerated from the whole CSV at the end)
#     --redo-skipped 1   rerun combos that were previously skipped by prediction
#     --force 1          do not skip by prediction (each fit is still capped by --time-limit)
# Re-running resumes: combos already in the CSV are not repeated.
#
# Data are scRNA-like and sparse for every size (Poisson mean ≈ 0.3, NB mean ≈ 0.4 with
# θ ≈ 3, ZINB with ≈ 20 % extra zeros, binary ≈ 20 % ones); 10 % of entries are held out (mask) for the test
# deviance. Peak memory = max RSS sampled every 50 ms during the fit (Linux
# /proc/self/statm; elsewhere Sys.maxrss), reported as absolute and above pre-fit.
# =============================================================================

include(joinpath(@__DIR__, "lowrank_glm.jl"))
using Printf

const SIZES = Dict("small" => (2_000, 500, false), "medium" => (20_000, 2_000, false), "large" => (100_000, 2_000, true))

function parse_args(args)
    o = Dict("sizes" => "small,medium,large", "ks" => "5,20,50", "families" => "poisson,bernoulli,zinb,nb",
             "solvers" => "newton,adam,hybrid", "time-limit" => "600",
             "out" => joinpath(@__DIR__, "stress_results.csv"), "summary" => joinpath(@__DIR__, "stress_summary.md"),
             "redo-skipped" => "0", "force" => "0")
    i = 1
    while i <= length(args)
        key = replace(args[i], "--" => ""); o[key] = args[i+1]; i += 2
    end
    return (sizes = split(o["sizes"], ","), ks = parse.(Int, split(o["ks"], ",")),
            families = Symbol.(split(o["families"], ",")), solvers = Symbol.(split(o["solvers"], ",")),
            time_limit = parse(Float64, o["time-limit"]), out = o["out"], summary = o["summary"],
            redo_skipped = o["redo-skipped"] == "1", force = o["force"] == "1")
end

rss_mb() = try
    parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 2^20
catch
    Sys.maxrss() / 2^20
end
# returns (result, peak RSS MB, peak live Julia heap above the pre-fit level MB)
function with_peak_mem(f)
    GC.gc(); GC.gc(); live0 = Base.gc_live_bytes()
    peak = Ref(rss_mb()); plive = Ref(live0); done = Ref(false)
    task = Threads.@spawn while !done[]
        peak[] = max(peak[], rss_mb()); plive[] = max(plive[], Base.gc_live_bytes()); sleep(0.05)
    end
    local r
    try
        r = f()
    finally
        done[] = true; wait(task)
    end
    plive[] = max(plive[], Base.gc_live_bytes())
    return r, max(peak[], rss_mb()), (plive[] - live0) / 2^20
end

# --- data: factors + row-blocked, threaded sampling (never materialises dense η) ---
function make_data(fam, n, p, k, sparse_out; seed = 1)
    rng = Xoshiro(seed)
    X = randn(rng, n, k); Y = randn(rng, p, k) ./ sqrt(k)
    o = fam === :bernoulli ? zeros(n) : 0.3 .* randn(rng, n)
    b = (fam === :poisson ? -1.5 : fam === :zinb ? -1.0 : fam === :nb ? -1.0 : -1.5) .+ 0.5 .* randn(rng, p)
    Ypi = 0.5 .* randn(rng, p, k) ./ sqrt(k); c = -1.5 .+ 0.5 .* randn(rng, p)
    θ = 3.0 .* exp.(0.5 .* randn(rng, p))
    if fam === :bernoulli; Y .*= 2; end
    blocks = collect(Iterators.partition(1:n, 2000))
    parts = Vector{Any}(undef, length(blocks))
    Threads.@threads for bi in eachindex(blocks)
        I = blocks[bi]; r = Xoshiro(seed * 1_000_003 + bi)
        η = o[I] .+ X[I, :] * Y' .+ b'
        Ab = zeros(Float32, length(I), p)
        if fam === :poisson
            for j in 1:p, i in axes(η, 1); Ab[i, j] = _rand_poisson(r, exp(η[i, j])); end
        elseif fam === :bernoulli
            for j in 1:p, i in axes(η, 1); Ab[i, j] = rand(r) < logistic(η[i, j]); end
        elseif fam === :nb
            for j in 1:p, i in axes(η, 1); Ab[i, j] = _rand_poisson(r, exp(η[i, j]) * _rand_gamma(r, θ[j]) / θ[j]); end
        else
            ηp = X[I, :] * Ypi' .+ c'
            for j in 1:p, i in axes(η, 1)
                rand(r) < logistic(ηp[i, j]) && continue
                Ab[i, j] = _rand_poisson(r, exp(η[i, j]) * _rand_gamma(r, θ[j]) / θ[j])
            end
        end
        parts[bi] = sparse_out ? SparseMatrixCSC{Float32,Int32}(sparse(Ab)) : Ab
    end
    A = sparse_out ? reduce(vcat, parts) : reduce(vcat, parts)
    sparse_out && (A = SparseMatrixCSC{Float32,Int32}(A))
    truth = (X = X, Y = Y, b = b, o = o)
    return A, truth
end

function holdout_mask(n, p, frac; seed = 2)
    rng = Xoshiro(seed); test = falses(n, p)
    for j in 1:p, i in 1:n; rand(rng) < frac && (test[i, j] = true); end
    return test
end

# correlation of true vs fitted η on (at most) 4000 rows
function eta_cor(fit, truth; nsub = 4000)
    n = size(truth.X, 1); rows = n <= nsub ? (1:n) : sort(randperm(Xoshiro(3), n)[1:nsub])
    ηt = truth.o[rows] .+ truth.X[rows, :] * truth.Y' .+ truth.b'
    ηf = fit.offset[rows] .+ fit.X[rows, :] * fit.Y' .+ fit.intercept'
    return cor(vec(ηt), vec(Float64.(ηf)))
end

const HEADER = "size,n,p,storage,family,k,solver,status,time_s,iters,converged,train_obj,heldout_dev,cor_eta,max_angle_deg,peak_rss_mb,peak_heap_above_prefit_mb,alloc_gb,threads"

function run_one(fam, A, truth, test, k, solver, time_limit)
    fitter = fam === :poisson ? fit_poisson_pca : fam === :bernoulli ? fit_logistic_pca :
             fam === :nb ? fit_nb_pca : fit_zinb_pca
    kw = fam === :bernoulli ? (;) : (offset = truth.o,)
    train = .!test
    (stats, peak, live) = with_peak_mem() do
        @timed fitter(A, k; mask = train, solver, time_limit, rng = Xoshiro(4), kw...)
    end
    f = stats.value
    evaldata = _Data(A, A, nothing)               # column access only, no copy
    dev = _heldout_deviance(f, evaldata, test)
    return (time = stats.time, iters = f.iters, conv = f.converged, obj = f.loss[end], dev = dev,
            cor = eta_cor(f, truth), ang = max_subspace_angle(truth.Y, f.Y), peak = peak, live = live,
            alloc = stats.bytes / 2^30)
end

function main(args = ARGS)
    opt = parse_args(args)
    isfile(opt.out) || write(opt.out, HEADER * "\n")
    # warm-up compile of every family × solver on a tiny problem (dense and sparse)
    println("warm-up compile ..."); flush(stdout)
    for fam in opt.families, sp in (false, true)
        A, tr = make_data(fam, 300, 120, 3, sp); te = holdout_mask(300, 120, 0.1)
        for s in opt.solvers; run_one(fam, A, tr, te, 3, s, 60.0); end
    end
    # resume: combos already in the CSV are not rerun; their times seed the skip prediction
    times = Dict{Any,Float64}()                   # (size, fam, solver, k) => seconds
    for l in readlines(opt.out)[2:end]
        isempty(l) && continue
        f = split(l, ",")
        skipped = isempty(f[9])
        skipped && opt.redo_skipped && continue
        t = skipped ? parse(Float64, match(r"skipped_pred_([0-9.]+)s", f[8])[1]) : parse(Float64, f[9])
        times[(String(f[1]), Symbol(f[5]), Symbol(f[7]), parse(Int, f[6]))] = t
    end
    sizes_sorted = sort(collect(opt.sizes); by = s -> prod(SIZES[s][1:2]))
    for (si, sz) in enumerate(sizes_sorted), fam in opt.families, k in sort(opt.ks)
        n, p, sp = SIZES[sz]
        k < min(n, p) || continue
        todo = [s for s in opt.solvers if !haskey(times, (sz, fam, s, k))]
        isempty(todo) && continue
        t_gen = @elapsed (A, truth) = make_data(fam, n, p, k, sp)
        test = holdout_mask(n, p, 0.1)
        nzf = sp ? nnz(A) / length(A) : count(!iszero, A) / length(A)
        @printf("\n== %s %d×%d %s k=%d  (data %.1fs, nonzero %.1f%%, RSS %.0f MB)\n", sz, n, p, fam, k, t_gen, 100nzf, rss_mb())
        flush(stdout)
        for solver in todo
            # predict runtime from the same solver at a smaller k (same size) or the previous size (same k)
            pred = 0.0
            ks_done = [kk for kk in opt.ks if haskey(times, (sz, fam, solver, kk)) && kk < k]
            if !isempty(ks_done)
                kk = maximum(ks_done); pred = max(pred, times[(sz, fam, solver, kk)] * (k / kk)^(solver === :adam ? 1.0 : 1.5))
            end
            allsz = sort(collect(keys(SIZES)); by = s -> prod(SIZES[s][1:2]))
            gi = findfirst(==(sz), allsz)
            if gi > 1 && haskey(times, (allsz[gi-1], fam, solver, k))
                n0, p0, _ = SIZES[allsz[gi-1]]
                pred = max(pred, times[(allsz[gi-1], fam, solver, k)] * (n * p) / (n0 * p0))
            end
            row = if !opt.force && pred > opt.time_limit
                @printf("  %-7s skipped (predicted %.0f s > limit)\n", solver, pred)
                times[(sz, fam, solver, k)] = pred
                [sz, n, p, sp ? "sparse" : "dense", fam, k, solver, @sprintf("skipped_pred_%.0fs", pred),
                 "", "", "", "", "", "", "", "", "", "", Threads.nthreads()]
            else
                r = run_one(fam, A, truth, test, k, solver, opt.time_limit)
                times[(sz, fam, solver, k)] = r.time
                status = r.conv ? "converged" : (r.time >= 0.95 * opt.time_limit ? "time_limit" : "maxiter")
                @printf("  %-7s %7.1f s  it %4d  %-10s obj %.6e  dev %.4f  cor %.4f  angle %5.2f°  RSS %.0f MB heap +%.0f MB\n",
                        solver, r.time, r.iters, status, r.obj, r.dev, r.cor, r.ang, r.peak, r.live)
                [sz, n, p, sp ? "sparse" : "dense", fam, k, solver, status, round(r.time; digits = 2), r.iters, r.conv,
                 r.obj, round(r.dev; digits = 6), round(r.cor; digits = 5), round(r.ang; digits = 3),
                 round(r.peak; digits = 0), round(r.live; digits = 0), round(r.alloc; digits = 3), Threads.nthreads()]
            end
            open(io -> println(io, join(row, ",")), opt.out, "a")
            flush(stdout)
        end
        A = nothing; test = nothing; GC.gc()
    end
    write_summary(opt.out, opt.summary)
end

# --- markdown summary from the CSV ---
function write_summary(csv, md)
    lines = readlines(csv); hdr = split(lines[1], ","); rows = [Dict(zip(hdr, split(l, ","))) for l in lines[2:end] if !isempty(l)]
    key(r) = (r["size"], r["family"], r["k"], r["solver"])
    real = Set(key(r) for r in rows if !isempty(r["time_s"]))
    rows = [r for r in rows if !isempty(r["time_s"]) || !(key(r) in real)]   # drop skip rows that were rerun
    num(x) = isempty(x) ? NaN : parse(Float64, x)
    io = IOBuffer()
    println(io, "# Stress test: Newton vs Adam vs hybrid (lowrank_glm.jl)\n")
    println(io, "Threads: $(isempty(rows) ? "?" : rows[1]["threads"]). Train objective = penalised NLL on the 90 % training entries; ",
                "held-out = mean per-entry deviance on the 10 % masked entries (NB / ZINB: -2 log-lik); cor(η) on ≤ 4000 rows; ",
                "angle = largest principal angle between true and fitted loadings; heap = peak live Julia heap during the fit ",
                "above the pre-fit level (includes the fitter's internal copy of the data); RSS = process peak (not per fit).\n")
    println(io, "| size | family | k | solver | status | time (s) | iters | Δobj vs best (rel) | held-out dev | cor(η) | angle (°) | heap (MB) | alloc (GB) |")
    println(io, "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    groups = unique([(r["size"], r["family"], r["k"]) for r in rows])
    for g in groups
        rs = [r for r in rows if (r["size"], r["family"], r["k"]) == g]
        objs = filter(!isnan, [num(r["train_obj"]) for r in rs]); bo = isempty(objs) ? NaN : minimum(objs)
        for r in rs
            o = num(r["train_obj"])
            @printf(io, "| %s %s×%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", r["size"], r["n"], r["p"],
                    r["family"], r["k"], r["solver"], r["status"], r["time_s"], r["iters"],
                    isnan(o) ? "" : @sprintf("%.1e", (o - bo) / abs(bo)), r["heldout_dev"], r["cor_eta"], r["max_angle_deg"],
                    r["peak_heap_above_prefit_mb"], r["alloc_gb"])
        end
    end
    println(io, "\n## Fastest solver per setting (among runs within 1e-3 relative of the best train objective)\n")
    for g in groups
        rs = [r for r in rows if (r["size"], r["family"], r["k"]) == g && !isnan(num(r["train_obj"]))]
        isempty(rs) && continue
        bo = minimum(num(r["train_obj"]) for r in rs)
        ok = [r for r in rs if (num(r["train_obj"]) - bo) / abs(bo) <= 1e-3]
        fast = ok[argmin([num(r["time_s"]) for r in ok])]
        others = join([@sprintf("%s %.1fs", r["solver"], num(r["time_s"])) for r in rs if r !== fast], ", ")
        println(io, "- $(g[1]) $(g[2]) k=$(g[3]): **$(fast["solver"])** $(fast["time_s"]) s ($(others))")
    end
    write(md, String(take!(io)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
