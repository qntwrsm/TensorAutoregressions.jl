#=
plotting.jl

    Provides a collection of visualization methods for tensor autoregressive models, such as
    impulse response functions, forecasts, data plots, and Kruskal coefficient tensor and
    tensor error covariance structure visualization.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/04/04
=#

"""
    data_plot(model[, labels, time]) -> fig

Plot the time series data, with optionally specified `labels` and `time` index.
"""
function data_plot(model::AbstractTensorAutoregression)
    d = dims(model)

    # sort modes
    order = sortperm(collect(d), rev = true)

    # setup subplots
    cols = ceil(Int, sqrt(d[order[1]]))
    rows = ceil(Int, d[order[1]] / cols)

    # setup figure
    fig = Figure(size = (cols * 600, cols * 450))
    grids = [GridLayout(fig[row, col]) for row in 1:rows
             for col in 1:cols if (row - 1) * cols + col <= d[order[1]]]
    axs = [Axis(grid[i, 1]) for grid in grids, i in 1:d[order[2]]]

    # layout
    for grid in grids
        colgap!(grid, 5)
        rowgap!(grid, 5)
    end

    # link axes
    linkxaxes!(axs...)
    for col in eachcol(axs)
        linkyaxes!(col...)
    end

    # decorations
    for row in eachrow(axs)
        row[end].xlabel = "time"
    end
    hidexdecorations!.(axs[:, 1:(end - 1)], grid = false)
    for (i, row) in pairs(eachrow(axs))
        if (i - 1) % cols != 0
            hideydecorations!.(row, grid = false)
        end
    end

    # data
    ncolors = length(d) > 2 ? prod(d[order[3:end]]) : 2
    colors = resample_cmap(:viridis, ncolors)
    for (idx, ax) in pairs(IndexCartesian(), axs)
        # identify 2nd largest mode
        k = order[2] - (order[1] < order[2] ? 1 : 0)
        # select largest and 2nd largest modes
        y = selectdim(selectdim(data(model), order[1], idx[1]), k, idx[2])
        # sort modes
        y_sorted = length(d) > 3 ? permutedims(y, order[3:end]) : y
        # plot
        series!(ax, reshape(y_sorted, :, nobs(model)), color = colors)
    end

    return fig
end
function data_plot(model::AbstractTensorAutoregression, labels, time)
    d = dims(model)

    # sort modes
    order = sortperm(collect(d), rev = true)

    # combine time series labels
    if length(d) > 2
        sub_labels = labels[order[3]]
        if length(d) > 3
            for mode in order[4:end]
                sub_labels = vec(sub_labels .* " - " .* reshape(labels[mode], 1, :))
            end
        end
    else
        sub_labels = nothing
    end

    # time ticks
    values = string.(unique(year.(time))[1:5:end])
    ticks = [findfirst(string.(year.(time)) .== value) for value in values]

    # setup subplots
    cols = ceil(Int, sqrt(d[order[1]]))
    rows = ceil(Int, d[order[1]] / cols)

    # setup figure
    fig = Figure(size = (cols * 600, cols * 450))
    grids = [GridLayout(fig[row, col]) for row in 1:rows
             for col in 1:cols if (row - 1) * cols + col <= d[order[1]]]
    axs = [Axis(grid[i, 1]) for grid in grids, i in 1:d[order[2]]]

    # layout
    for grid in grids
        colgap!(grid, 5)
        rowgap!(grid, 5)
    end

    # link axes
    linkxaxes!(axs...)
    for col in eachcol(axs)
        linkyaxes!(col...)
    end

    # decorations
    for row in eachrow(axs)
        row[end].xlabel = "time"
        row[end].xticks = (ticks, values)
    end
    hidexdecorations!.(axs[:, 1:(end - 1)], grid = false)
    for (i, row) in pairs(eachrow(axs))
        if (i - 1) % cols != 0
            hideydecorations!.(row, grid = false)
        end
    end

    # grid titles
    for (i, grid) in pairs(grids)
        Label(grid[1, :, Top()], labels[order[1]][i], valign = :bottom, font = :bold)
    end
    # axis titles
    for (i, grid) in pairs(grids)
        if i % cols == 0 || i == length(grids)
            for (j, label) in pairs(labels[order[2]])
                Box(grid[j, 2], color = :gray90)
                Label(grid[j, 2], label, rotation = π / 2, tellheight = false)
            end
            colgap!(grid, 0)
        end
    end

    # data
    ncolors = length(d) > 2 ? prod(d[order[3:end]]) : 2
    colors = resample_cmap(:viridis, ncolors)
    for (idx, ax) in pairs(IndexCartesian(), axs)
        # identify 2nd largest mode
        k = order[2] - (order[1] < order[2] ? 1 : 0)
        # select largest and 2nd largest modes
        y = selectdim(selectdim(data(model), order[1], idx[1]), k, idx[2])
        # sort modes
        y_sorted = length(d) > 3 ? permutedims(y, order[3:end]) : y
        # plot
        series!(ax, reshape(y_sorted, :, nobs(model)), color = colors,
                labels = sub_labels)
    end

    # add legend
    if length(d) > 2
        if d[order[1]] == cols * rows
            Legend(fig[:, cols + 1], axs[1])
        else
            Legend(fig[rows, cols], axs[1], tellwidth = false, halign = :left)
        end
    end

    return fig
end

"""
    kruskal_plot(A[, labels, time]) -> figs

Plot factors and loadings of the Kruskal coefficient tensor `A` with optionally specified
`labels` and `time`.
"""
function kruskal_plot(A::StaticKruskal)
    d = size(A)
    n = length(d) ÷ 2

    # setup figure
    fig = Figure(size = (prod(d[1:n]) * 20 + 200, prod(d[1:n]) * 20))
    ax = Axis(fig[1, 1])

    # Kruskal coefficient tensor factors and static loadings
    hm = heatmap!(ax, matricize(full(A), (n + 1):(2n))')
    Colorbar(fig[1, 2], hm)

    return fig
end
function kruskal_plot(A::StaticKruskal, labels)
    d = size(A)
    n = length(d) ÷ 2

    # setup figures
    fig = Figure(size = (prod(d[1:n]) * 20 + 200, prod(d[1:n]) * 20))
    # main grids
    gl = GridLayout(fig[1, 1])  # left label grid
    gb = GridLayout(fig[2, 2])  # bottom label grid
    gm = GridLayout(fig[1, 2])  # main plotting grid
    gc = GridLayout(fig[1, 3])  # colorbar grid
    # plotting axis
    ax = Axis(gm[1, 1])

    # label grids
    nested_labels!(gl, d[1:n], n, labels, :left)
    nested_labels!(gb, d[1:n], n, labels, :bottom)

    # Kruskal coefficient tensor
    hm = heatmap!(ax, matricize(full(A), (n + 1):(2 * n))')
    Colorbar(gc[1, 1], hm)

    # ticks
    ax.xticks = (1:prod(d[1:n]), repeat(labels[1], prod(d[2:n])))
    ax.yticks = (1:prod(d[1:n]), repeat(labels[1], prod(d[2:n])))

    return fig
end
function kruskal_plot(A::DynamicKruskal)
    d = size(A)
    n = length(d) ÷ 2

    # setup figures
    figs = [Figure(size = (prod(d[1:n]) * 20 + 200, prod(d[1:n]) * 20)) for _ in 1:rank(A)]
    push!(figs, Figure())
    axs = [Axis(fig[1, 1]) for fig in figs]

    # Kruskal coefficient tensor factors
    for r in 1:rank(A)
        hm = heatmap!(axs[r], matricize(full(A)[r], (n + 1):(2n))')
        Colorbar(figs[r][1, 2], hm)
    end

    # dynamic Kruskal loadings
    colors = resample_cmap(:viridis, max(rank(A), 2))
    series!(axs[end], loadings(A), color = colors, labels = ["rank $r" for r in 1:rank(A)])

    # add legend
    Legend(figs[end][1, 2], axs[end])

    return figs
end
function kruskal_plot(A::DynamicKruskal, labels, time)
    d = size(A)
    n = length(d) ÷ 2

    # setup figures
    figs = [Figure(size = (prod(d[1:n]) * 20 + 200, prod(d[1:n]) * 20)) for _ in 1:rank(A)]
    push!(figs, Figure())
    # main grids
    gl = [GridLayout(figs[r][1, 1]) for r in 1:rank(A)]  # left label grid
    gb = [GridLayout(figs[r][2, 2]) for r in 1:rank(A)]  # bottom label grid
    gm = [GridLayout(figs[r][1, 2]) for r in 1:rank(A)]  # main plotting grid
    gc = [GridLayout(figs[r][1, 3]) for r in 1:rank(A)]  # colorbar grid
    # plotting axes
    axs = [Axis(gm[r][1, 1]) for r in 1:rank(A)]
    push!(axs, Axis(figs[end][1, 1]))

    # label grids
    for r in 1:rank(A)
        nested_labels!(gl[r], d[1:n], n, labels, :left)
        nested_labels!(gb[r], d[1:n], n, labels, :bottom)
    end

    # Kruskal coefficient tensor factors
    for r in 1:rank(A)
        hm = heatmap!(axs[r], matricize(full(A)[r], (n + 1):(2n))')
        Colorbar(gc[r][1, 1], hm)
    end

    # ticks
    for r in 1:rank(A)
        axs[r].xticks = (1:prod(d[1:n]), repeat(labels[1], prod(d[2:n])))
        axs[r].yticks = (1:prod(d[1:n]), repeat(labels[1], prod(d[2:n])))
    end

    # dynamic Kruskal loadings
    colors = resample_cmap(:viridis, max(rank(A), 2))
    series!(axs[end], loadings(A), color = colors, labels = ["rank $r" for r in 1:rank(A)])

    # time ticks
    values = string.(unique(year.(time))[1:5:end])
    ticks = [findfirst(string.(year.(time)) .== value) for value in values]
    axs[end].xlabel = "time"
    axs[end].xticks = (ticks, values)

    # add legend
    Legend(figs[end][1, 2], axs[end])

    return figs
end

"""
    cov_plot(ε[, labels]) -> fig

Plot the tensor error covariance structure `ε` with optionally specified `labels`.
"""
function cov_plot(ε::AbstractTensorErrorDistribution)
    d = size.(cov(ε), 1)

    # setup figure
    fig = Figure(size = (prod(d) * 20 + 200, prod(d) * 20))
    ax = Axis(fig[1, 1])

    # covariance matrix
    hm = heatmap!(ax, cov(ε, full = true)')
    Colorbar(fig[1, 2], hm)

    return fig
end
function cov_plot(ε::AbstractTensorErrorDistribution, labels)
    d = size.(cov(ε), 1)
    n = length(d)

    # setup figures
    fig = Figure(size = (prod(d) * 20 + 200, prod(d) * 20))
    # main grids
    gl = GridLayout(fig[1, 1])  # left label grid
    gb = GridLayout(fig[2, 2])  # bottom label grid
    gm = GridLayout(fig[1, 2])  # main plotting grid
    gc = GridLayout(fig[1, 3])  # colorbar grid
    # plotting axis
    ax = Axis(gm[1, 1])

    # label grids
    nested_labels!(gl, d, n, labels, :left)
    nested_labels!(gb, d, n, labels, :bottom)

    # covariance matrix
    hm = heatmap!(ax, cov(ε, full = true)')
    Colorbar(gc[1, 1], hm)

    # ticks
    ax.xticks = (1:prod(d), repeat(labels[1], prod(d[2:n])))
    ax.yticks = (1:prod(d), repeat(labels[1], prod(d[2:n])))

    return fig
end

"""
    nested_labels!(grid, d, n, labels, loc)

Add recursively nested labels from `n`nd level from `labels` to a grid layout `grid`
with dimensions `d` for the levels and `loc` indicating the location of the labels.
"""
function nested_labels!(grid, d, n, labels, loc)
    # terminate at final to last level (last level are axis ticks)
    if n == 2
        for i in 1:d[n]
            # add all remaining labels
            if loc == :bottom
                Box(grid[1, i])
                Label(grid[1, i], labels[n][i], tellwidth = false)
            elseif loc == :left
                Box(grid[i, 1])
                Label(grid[i, 1], reverse(labels[n])[i], rotation = π / 2,
                      tellheight = false)
            end
        end
    else
        for i in 1:d[n]
            # subgrid
            g = loc == :left ? GridLayout(grid[length(d) - n + 1, 1]) :
                GridLayout(grid[n, 1])
            Box(g[1, 1])
            # add label of current level
            if loc == :bottom
                Label(g[1, 1], labels[n][i], tellwidth = false)
            elseif loc == :left
                Label(g[1, 1], reverse(labels[n])[i], rotation = π / 2, tellheight = false)
            end
            # move to next level
            nested_labels!(grid, d, n - 1, labels, loc)
        end
    end
end

"""
    irf_plot(irfs, response[, impulse]) -> fig

Plot the impulse response function `irfs` for the response `response`, when `irfs` is
dynamic an `impulse` can be provided.
"""
function irf_plot(irfs::StaticIRF, response, impulse)
    ψ = irf(irfs, impulse, response)
    periods = length(ψ) - 1

    # setup figure
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Generalized Impulse Response Function",
              titlealign = :left, titlecolor = :gray50, xlabel = "periods",
              xticks = (0:periods, ["$h" for h in 0:periods]))

    # impulse response function
    lines!(ax, 0:periods, ψ, color = :black)

    # confidence bands
    lines!(ax, 0:periods, lower(irfs, impulse, response), color = :gray50,
           linestyle = :dash)
    lines!(ax, 0:periods, upper(irfs, impulse, response), color = :gray50,
           linestyle = :dash)

    return fig
end
function irf_plot(irfs::DynamicIRF, response)
    ψ = irf(irfs, response)
    periods = length(ψ) - 1

    # setup figure
    fig = Figure()
    ax = Axis(fig[1, 1],
              title = "Generalized Impulse Response Function", titlealign = :left,
              titlecolor = :gray50, xlabel = "periods",
              xticks = (0:periods, ["$h" for h in 0:periods]))

    # impulse response function
    lines!(ax, 0:periods, ψ, color = :black)

    # confidence bands
    lines!(ax, 0:periods, lower(irfs, response), color = :gray50, linestyle = :dash)
    lines!(ax, 0:periods, upper(irfs, response), color = :gray50, linestyle = :dash)

    return fig
end
