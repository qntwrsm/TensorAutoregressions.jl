#=
plotting.jl

    Provides a collection of plotting methods for tensor autoregressive models, 
    such as impulse response functions, forecasts, and data plots.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/04/04
=#

"""
    data_plot(model) -> fig

Plot the time series data.
"""
function data_plot(model::TensorAutoregression)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # maximum sized mode
    maxmode = argmax(dims[1:n])
    m = setdiff(1:n, maxmode)

    # setup subplots
    cols = iseven(dims[maxmode]) ? 2 : 3
    rows = ceil(Int, dims[maxmode] / cols)
    indices = CartesianIndices((rows, cols))

    # setup figure
    fig = Figure()
    axs = [Axis(fig[Tuple(idx)...]) for idx ∈ indices[1:dims[maxmode]]]
    
    # link y axes
    linkyaxes!(axs...)

    # data
    colors = resample_cmap(:viridis, prod(dims[m]))
    for (i, ax) ∈ enumerate(axs) 
        ax.xlabel = "time"
        series!(
            ax, 
            1:last(dims), 
            reshape(selectdim(data(model), maxmode, i), :, last(dims)), 
            color=colors
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