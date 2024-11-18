#=
plotting.jl

    Provides a collection of visualization methods for tensor autoregressive 
    models, such as impulse response functions, forecasts, data plots, and 
    Kruskal coefficient tensor and tensor error covariance structure 
    visualization.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/04/04
=#

"""
    data_plot(model[, labels, time]) -> fig

Plot the time series data, with optionally specified `labels` and `time` index.
"""
function data_plot(model::AbstractTensorAutoregression)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # sort modes
    maxmode = argmax(dims[1:(n - 1)])
    m = setdiff(1:(n - 1), maxmode)

    # setup subplots
    cols = iseven(dims[maxmode]) ? 2 : 3
    rows = ceil(Int, dims[maxmode] / cols)
    indices = CartesianIndices((rows, cols))

    # setup figure
    fig = Figure(resolution = (cols * 800, cols * 600))
    grids = [GridLayout(fig[Tuple(idx)...]) for idx in indices[1:dims[maxmode]]]
    axs = [Axis(grid[i, 1]) for grid in grids, i in 1:dims[n]]

    # layout
    for grid in grids
        colgap!(grid, 0)
        rowgap!(grid, 0)
    end

    # link axes
    linkxaxes!(axs...)
    for i in 1:dims[n]
        linkyaxes!(axs[:, i]...)
    end

    # decorations
    for i in 1:dims[maxmode]
        axs[i, end].xlabel = "time"
        axs[i, end].xticks = (ticks, values)
    end
    hidexdecorations!.(axs[:, 1:(end - 1)], grid = false)
    hideydecorations!.(axs[(rows + 1):end, :], grid = false)

    # data
    colors = resample_cmap(:viridis, prod(dims[m]))
    for (idx, ax) in pairs(IndexCartesian(), axs)
        y = selectdim(data(model), maxmode, idx[1])
        series!(ax,
                reshape(selectdim(y, n - 1, idx[2]), :, last(dims)),
                color = colors)
    end

    return fig
end

function data_plot(model::AbstractTensorAutoregression, labels, time)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # sort modes
    maxmode = argmax(dims[1:(n - 1)])
    m = setdiff(1:(n - 1), maxmode)

    # combine time series labels
    sub_labels = labels[m[1]]
    for mode in m[2:end]
        sub_labels = vec(sub_labels .* " - " .* reshape(labels[mode], 1, :))
    end

    # time ticks
    values = string.(unique(year.(time))[1:5:end])
    ticks = [findfirst(string.(year.(time)) .== value) for value in values]

    # setup subplots
    cols = iseven(dims[maxmode]) ? 2 : 3
    rows = ceil(Int, dims[maxmode] / cols)
    indices = CartesianIndices((rows, cols))

    # setup figure
    fig = Figure(resolution = (cols * 800, cols * 600))
    grids = [GridLayout(fig[Tuple(idx)...]) for idx in indices[1:dims[maxmode]]]
    axs = [Axis(grid[i, 1]) for grid in grids, i in 1:dims[n]]

    # layout
    for grid in grids
        colgap!(grid, 0)
        rowgap!(grid, 0)
    end

    # link axes
    linkxaxes!(axs...)
    for i in 1:dims[n]
        linkyaxes!(axs[:, i]...)
    end

    # decorations
    for i in 1:dims[maxmode], j in 1:dims[n]
        axs[i, j].xlabel = "time"
        axs[i, j].xticks = (ticks, values)
    end
    hidexdecorations!.(axs[:, 1:(end - 1)], grid = false)
    hideydecorations!.(axs[(rows + 1):end, :], grid = false)

    # grid titles
    for (i, grid) in pairs(grids)
        Label(grid[1, :, Top()],
              labels[maxmode][i],
              valign = :bottom,
              font = :bold)
    end
    # axis titles
    for grid in grids[(end - rows + 1):end]
        for (i, label) in pairs(labels[end])
            Box(grid[i, 2], color = :gray90)
            Label(grid[i, 2], label, rotation = π / 2, tellheight = false)
        end
        colgap!(grid, 0)
    end

    # data
    colors = resample_cmap(:viridis, prod(dims[m]))
    for (idx, ax) in pairs(IndexCartesian(), axs)
        y = selectdim(data(model), maxmode, idx[1])
        series!(ax,
                reshape(selectdim(y, n - 1, idx[2]), :, last(dims)),
                color = colors,
                labels = sub_labels)
    end

    # add legend
    if dims[maxmode] == length(indices)
        Legend(fig[:, cols + 1], axs[1], "sectors")
    else
        Legend(fig[rows, cols],
               axs[1],
               "sectors",
               tellwidth = false,
               halign = :left)
    end

    return fig
end

"""
    kruskal_plot(A[, labels, time]) -> figs

Plot factors and loadings of the Kruskal coefficient tensor `A` with optionally
specified `labels` and `time`.
"""
function kruskal_plot(A::StaticKruskal)
    dims = size(A)
    n = length(dims) ÷ 2

    # setup figure
    fig = Figure(resolution = (1600, 1600))
    ax = Axis(fig[1, 1])

    # Kruskal coefficient tensor factors and static loadings
    hm = heatmap!(ax, matricize(full(A), (n + 1):(2n))')
    Colorbar(fig[1, 2], hm)

    return fig
end

function kruskal_plot(A::StaticKruskal, labels)
    dims = size(A)
    n = length(dims) ÷ 2

    # setup figures
    fig = Figure(resolution = (1600, 1600))
    # main grids
    gl = GridLayout(fig[1, 1])  # left label grid
    gb = GridLayout(fig[2, 2])  # bottom label grid
    gm = GridLayout(fig[1, 2])  # main plotting grid
    gc = GridLayout(fig[1, 3])  # colorbar grid
    # plotting axis
    ax = Axis(gm[1, 1])

    # label grids
    for i in 1:dims[n]
        nested_labels!(GridLayout(gl[i, 1]), dims, n, i, labels, :left)
        nested_labels!(GridLayout(gb[1, i]), dims, n, i, labels, :bottom)
    end

    # Kruskal coefficient tensor
    hm = heatmap!(ax, matricize(full(A), (n + 1):(2 * n))')
    Colorbar(gc[1, 1], hm)

    # ticks
    ax.xticks = (1:prod(dims[1:n]), repeat(labels[1], prod(dims[2:n])))
    ax.yticks = (1:prod(dims[1:n]), repeat(labels[1], prod(dims[2:n])))

    return fig
end

function kruskal_plot(A::DynamicKruskal)
    dims = size(A)
    n = length(dims) ÷ 2

    # setup figures
    figs = [Figure(resolution = (1600, 1600)), Figure()]
    axs = [Axis(fig[1, 1]) for fig in figs]

    # Kruskal coefficient tensor factors
    hm = heatmap!(axs[1], matricize(full(A), (n + 1):(2n))')
    Colorbar(figs[1][1, 2], hm)

    # dynamic Kruskal loadings
    series!(axs[2], loadings(A), color = :viridis)

    return figs
end

function kruskal_plot(A::DynamicKruskal, labels, time)
    dims = size(A)
    n = length(dims) ÷ 2

    # setup figures
    figs = [Figure(resolution = (1600, 1600)), Figure()]
    # main grids
    gl = GridLayout(figs[1][1, 1])  # left label grid
    gb = GridLayout(figs[1][2, 2])  # bottom label grid
    gm = GridLayout(figs[1][1, 2])  # main plotting grid
    gc = GridLayout(figs[1][1, 3])  # colorbar grid
    # plotting axes
    axs = [Axis(gm[1, 1]), Axis(figs[2][1, 1])]

    # label grids
    for i in 1:dims[n]
        nested_labels!(GridLayout(gl[i, 1]), dims, n, i, labels, :left)
        nested_labels!(GridLayout(gb[1, i]), dims, n, i, labels, :bottom)
    end

    # Kruskal coefficient tensor factors
    hm = heatmap!(axs[1], matricize(full(A), (n + 1):(2n))')
    Colorbar(gc[1, 1], hm)

    # ticks
    axs[1].xticks = (1:prod(dims[1:n]), repeat(labels[1], prod(dims[2:n])))
    axs[1].yticks = (1:prod(dims[1:n]), repeat(labels[1], prod(dims[2:n])))

    # dynamic Kruskal loadings
    series!(axs[2], loadings(A), color = :viridis)

    # time ticks
    values = string.(unique(year.(time))[1:5:end])
    ticks = [findfirst(string.(year.(time)) .== value) for value in values]
    axs[2].xlabel = "time"
    axs[2].xticks = (ticks, values)

    return figs
end

"""
    cov_plot(ε[, labels]) -> fig

Plot the tensor error covariance structure `ε` with optionally specified
`labels`.
"""
function cov_plot(ε::AbstractTensorErrorDistribution)
    # setup figure
    fig = Figure(resolution = (1600, 1600))
    ax = Axis(fig[1, 1])

    # covariance matrix
    hm = heatmap!(ax, cov(ε, full = true)')
    Colorbar(fig[1, 2], hm)

    return fig
end

function cov_plot(ε::AbstractTensorErrorDistribution, labels)
    dims = size.(cov(ε), 1)
    n = length(dims)

    # setup figures
    fig = Figure(resolution = (1600, 1600))
    # main grids
    gl = GridLayout(fig[1, 1])  # left label grid
    gb = GridLayout(fig[2, 2])  # bottom label grid
    gm = GridLayout(fig[1, 2])  # main plotting grid
    gc = GridLayout(fig[1, 3])  # colorbar grid
    # plotting axis
    ax = Axis(gm[1, 1])

    # label grids
    for i in 1:dims[n]
        nested_labels!(GridLayout(gl[i, 1]), dims, n, i, labels, :left)
        nested_labels!(GridLayout(gb[1, i]), dims, n, i, labels, :bottom)
    end

    # covariance matrix
    hm = heatmap!(ax, cov(ε, full = true)')
    Colorbar(gc[1, 1], hm)

    # ticks
    ax.xticks = (1:prod(dims[1:n]), repeat(labels[1], prod(dims[2:n])))
    ax.yticks = (1:prod(dims[1:n]), repeat(labels[1], prod(dims[2:n])))

    return fig
end

"""
    nested_labels!(grid, dims, n, i, labels, loc)

Add recursively nested `i`-th label from `n`nd level from `labels` to a grid
layout `grid` with dimensions `dims` for the levels and `loc` indicating the
location of the labels.
"""
function nested_labels!(grid, dims, n, i, labels, loc)
    # terminate at final to last level (last level are axis ticks)
    if n == 2
        for j in 1:dims[n]
            # add all remaining labels
            if loc == :bottom
                Box(grid[1, j])
                Label(grid[1, j], labels[n][j], tellwidth = false)
            elseif loc == :left
                Box(grid[j, 1])
                Label(grid[j, 1], reverse(labels[n])[j], rotation = π / 2,
                      tellheight = false)
            end
        end
    else
        # first subgrid
        ga = loc == :left ? GridLayout(grid[1, 1]) : GridLayout(grid[2, 1])
        Box(ga[1, 1])
        # add label of current level
        if loc == :bottom
            Label(ga[1, 1], labels[n][i], tellwidth = false)
        elseif loc == :left
            Label(ga[1, 1], reverse(labels[n])[i], rotation = π / 2, tellheight = false)
        end
        # second subgrid (will be recursively filled with new level)
        gb = loc == :left ? GridLayout(grid[1, 2]) : GridLayout(grid[1, 1])
        for j in 1:dims[n - 1]
            nested_labels!(gb, dims, n - 1, j, labels, loc)
        end
    end
end

"""
    irf_plot(irfs, impulse, response[, time]) -> fig

Plot the impulse response function `irfs` for the response `response` to the
impulse `impulse`. When `irfs` is dynamic `time` can be provided to plot the
impulse response function for a specific time period.
"""
function irf_plot(irfs::StaticIRF, impulse, response)
    ψ = irf(irfs, impulse, response)
    periods = length(ψ) - 1

    # setup figure
    fig = Figure()
    ax = Axis(fig[1, 1],
              title = orth(irfs) ? "Orthogonal Impulse Response Function" :
                      "Impulse Response Function",
              titlealign = :left,
              titlecolor = :gray50,
              xlabel = "periods",
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

function irf_plot(irfs::DynamicIRF, impulse, response)
    ψ = irf(irfs, impulse, response)
    periods = size(ψ, 1) - 1
    time = size(ψ, 2)

    # setup figure
    fig = Figure()
    ax = Axis3(fig[1, 1],
               azimuth = π / 4,
               title = orth(irfs) ? "Orthogonal Impulse Response Function" :
                       "Impulse Response Function",
               titlealign = :left,
               titlecolor = :gray50,
               xlabel = "time",
               ylabel = "periods",
               yticks = (0:periods, ["$h" for h in 0:periods]))

    # impulse response function
    surface!(ax, 1:time, 0:periods, ψ')

    return fig
end

function irf_plot(irfs::DynamicIRF, impulse, response, time)
    ψ = irf(irfs, impulse, response, time)
    periods = size(ψ, 1) - 1

    # setup figure
    fig = Figure()
    ax = Axis3(fig[1, 1],
               azimuth = π / 4,
               title = orth(irfs) ? "Orthogonal Impulse Response Function" :
                       "Impulse Response Function",
               titlealign = :left,
               titlecolor = :gray50,
               xlabel = "time",
               ylabel = "periods",
               yticks = (0:periods, ["$h" for h in 0:periods]))

    # impulse response function
    surface!(ax, time, 0:periods, ψ')

    return fig
end

function irf_plot(irfs::DynamicIRF, impulse, response, time::Integer)
    ψ = irf(irfs, impulse, response, time)
    periods = length(ψ) - 1

    # setup figure
    fig = Figure()
    ax = Axis(fig[1, 1],
              title = orth(irfs) ? "Orthogonal Impulse Response Function" :
                      "Impulse Response Function",
              titlealign = :left,
              titlecolor = :gray50,
              xlabel = "periods",
              xticks = (0:periods, ["$h" for h in 0:periods]))

    # impulse response function
    lines!(ax, 0:periods, ψ, color = :black)

    # confidence bands
    lines!(ax, 0:periods, lower(irfs, impulse, response, time),
           color = :gray50, linestyle = :dash)
    lines!(ax, 0:periods, upper(irfs, impulse, response, time),
           color = :gray50, linestyle = :dash)

    return fig
end
