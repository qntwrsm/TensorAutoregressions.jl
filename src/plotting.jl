#=
plotting.jl

    Provides a collection of plotting methods for tensor autoregressive models, 
    such as impulse response functions, forecasts, and data plots.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/04/04
=#

"""
    data_plot(model[, labels, time]) -> fig

Plot the time series data, with optionally specified `labels` and `time` index.
"""
function data_plot(model::TensorAutoregression)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # sort modes
    p = sortperm(collect(dims[1:n]), rev=true)
    m = setdiff(1:n, p[1:2])

    # setup subplots
    cols = iseven(dims[p[1]]) ? 2 : 3
    rows = ceil(Int, dims[p[1]] / cols)
    indices = CartesianIndices((rows, cols))

    # setup figure
    fig = Figure()
    grids = [GridLayout(fig[Tuple(idx)...]) for idx ∈ indices[1:dims[p[1]]]]
    axs = [Axis(grid[k, 1]) for grid ∈ grids, k = 1:dims[p[2]]]

    # layout
    for grid ∈ grids
        colgap!(grid, 0)
        rowgap!(grid, 0)
    end

    # link axes
    linkxaxes!(axs...)
    for k = 1:dims[p[2]]
        linkyaxes!(axs[:,k]...)
    end

    # decorations
    for i = 1:dims[p[1]]
        axs[i,end].xlabel = "time"
    end
    hidexdecorations!.(axs[:,1:end-1])

    # data
    colors = resample_cmap(:viridis, prod(dims[m]))
    for (idx, ax) ∈ pairs(IndexCartesian(), axs)
        y = selectdim(data(model), p[1], idx[1])
        series!(
            ax, 
            1:last(dims), 
            reshape(selectdim(y, p[2], idx[2]), :, last(dims)), 
            color=colors
        )
    end

    return fig
end

function data_plot(model::TensorAutoregression, labels, time)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # sort modes
    maxmode = argmax(dims[1:n-1])
    m = setdiff(1:n-1, maxmode)

    # combine time series labels
    sub_labels = labels[m[1]]
    for mode ∈ m[2:end]
        sub_labels = vec(sub_labels .* " - " .* reshape(labels[mode], 1, :))
    end

    # time ticks
    values = string.(unique(year.(time))[1:5:end])
    ticks = [findfirst(string.(year.(time)) .== value) for value ∈ values]

    # setup subplots
    cols = iseven(dims[maxmode]) ? 2 : 3
    rows = ceil(Int, dims[maxmode] / cols)
    indices = CartesianIndices((rows, cols))

    # setup figure
    fig = Figure(resolution=(cols * 800, cols * 600))
    grids = [GridLayout(fig[Tuple(idx)...]) for idx ∈ indices[1:dims[maxmode]]]
    axs = [Axis(grid[i, 1]) for grid ∈ grids, i = 1:dims[n]]
    
    # layout
    for grid ∈ grids
        colgap!(grid, 0)
        rowgap!(grid, 0)
    end

    # link axes
    linkxaxes!(axs...)
    for i = 1:dims[n]
        linkyaxes!(axs[:,i]...)
    end

    # decorations
    for i = 1:dims[maxmode]
        axs[i,end].xlabel = "time"
        axs[i,end].xticks = (ticks, values)
    end
    hidexdecorations!.(axs[:,1:end-1], grid=false)
    hideydecorations!.(axs[rows+1:end,:], grid=false)

    # grid titles
    for (i, grid) ∈ pairs(grids)
        Label(
            grid[1, :, Top()], 
            labels[maxmode][i], 
            valign=:bottom,
            font=:bold
        )
    end
    # axis titles
    for grid ∈ grids[end-rows+1:end]
        for (i, label) in pairs(labels[end])
            Box(grid[i, 2], color=:gray90)
            Label(grid[i, 2], label, rotation=π/2, tellheight=false)
        end
        colgap!(grid, 0)
    end

    # data
    colors = resample_cmap(:viridis, prod(dims[m]))
    for (idx, ax) ∈ pairs(IndexCartesian(), axs)
        y = selectdim(data(model), maxmode, idx[1])
        series!(
            ax,
            reshape(selectdim(y, n-1, idx[2]), :, last(dims)), 
            color=colors,
            labels=sub_labels
        )
    end

    # add legend
    if dims[maxmode] == length(indices)
        Legend(fig[:, cols+1], axs[1], "sectors")
    else
        Legend(
            fig[rows, cols], 
            axs[1], 
            "sectors", 
            tellwidth=false, 
            halign=:left,       
        )
    end

    return fig
end

"""
    irf_plot(irfs, response, impulse[, time]) -> fig

Plot the impulse response function `irfs` for the response `response` to the
impulse `impulse`. When `irfs` is dynamic `time` can be provided to plot the
impulse response function for a specific time period.
"""
function irf_plot(irfs::StaticIRF, response, impulse)
    ψ = irf(irfs, response, impulse)
    periods = length(ψ) - 1

    # setup figure
    fig = Figure()
    ax = Axis(
        fig[1,1], 
        title=orth(irfs) ? "Orthogonal Impulse Response Function" : "Impulse Response Function",
        titlealign=:left,
        titlecolor=:gray50,
        xlabel="periods",
        xticks=(0:periods, ["$h" for h = 0:periods])
    )

    # impulse response function
    lines!(ax, 0:periods, ψ, color=:black)

    # confidence bands
    lines!(ax, 0:periods, lower(irfs, response, impulse), color=:gray50, linestyle=:dash)
    lines!(ax, 0:periods, upper(irfs, response, impulse), color=:gray50, linestyle=:dash)

    return fig
end

function irf_plot(irfs::DynamicIRF, response, impulse)
    ψ = irf(irfs, response, impulse)
    periods = size(ψ, 1) - 1
    time = size(ψ, 2)

    # setup figure
    fig = Figure()
    ax = Axis3(
        fig[1,1], 
        azimuth = π/4, 
        title=orth(irfs) ? "Orthogonal Impulse Response Function" : "Impulse Response Function",
        titlealign=:left,
        titlecolor=:gray50,
        xlabel="time",
        ylabel="periods",
        yticks=(0:periods, ["$h" for h = 0:periods])
    )

    # impulse response function
    surface!(ax, 1:time, 0:periods, ψ')

    return fig
end

function irf_plot(irfs::DynamicIRF, response, impulse, time)
    ψ = irf(irfs, response, impulse, time)
    periods = size(ψ, 1) - 1

    # setup figure
    fig = Figure()
    ax = Axis3(
        fig[1,1], 
        azimuth = π/4, 
        title=orth(irfs) ? "Orthogonal Impulse Response Function" : "Impulse Response Function",
        titlealign=:left,
        titlecolor=:gray50,
        xlabel="time",
        ylabel="periods",
        yticks=(0:periods, ["$h" for h = 0:periods])
    )

    # impulse response function
    surface!(ax, time, 0:periods, ψ')

    return fig
end

function irf_plot(irfs::DynamicIRF, response, impulse, time::Integer)
    ψ = irf(irfs, response, impulse, time)
    periods = length(ψ) - 1

    # setup figure
    fig = Figure()
    ax = Axis(
        fig[1,1], 
        title=orth(irfs) ? "Orthogonal Impulse Response Function" : "Impulse Response Function",
        titlealign=:left,
        titlecolor=:gray50,
        xlabel="periods",
        xticks=(0:periods, ["$h" for h = 0:periods])
    )

    # impulse response function
    lines!(ax, 0:periods, ψ, color=:black)

    # confidence bands
    lines!(ax, 0:periods, lower(irfs, response, impulse, time), color=:gray50, linestyle=:dash)
    lines!(ax, 0:periods, upper(irfs, response, impulse, time), color=:gray50, linestyle=:dash)

    return fig
end