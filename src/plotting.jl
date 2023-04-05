#=
plotting.jl

    Provides a collection of plotting methods for tensor autoregressive models, 
    such as impulse response functions, forecasts, and data plots.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/04/04
=#

"""
    irf_plot(irfs, response, impulse[, time]) -> fig

Plot the impulse response function `irfs` for the response `response` to the
impulse `impulse`. When `irfs` is dynamic `time` can be provided to plot the
impulse response function for a specific time period.
"""
function irf_plot(irfs::StaticIRF, response, impulse)
    # setup figure
    fig = Figure()
    ax = Axis(
        fig[1,1], 
        title=orth(irfs) ? "Orthogonal Impulse Response Function" : "Impulse Response Function",
        titlealign=:left,
        titlecolor=:gray50,
        xlabel="periods" 
    )

    # impulse response function
    lines!(ax, irf(irfs, response, impulse), color=:black)

    # confidence bands
    lines!(ax, lower(irfs, response, impulse), color=:gray50, linestyle=:dash)
    lines!(ax, upper(irfs, response, impulse), color=:gray50, linestyle=:dash)

    return fig
end

function irf_plot(irfs::DynamicIRF, response, impulse)
    # setup figure
    fig = Figure()
    ax = Axis3(
        fig[1,1], 
        title=orth(irfs) ? "Orthogonal Impulse Response Function" : "Impulse Response Function",
        titlealign=:left,
        titlecolor=:gray50,
        xlabel="time",
        ylabel="periods"
    )

    # impulse response function
    surface!(ax, irf(irfs, response, impulse))

    return fig
end

function irf_plot(irfs::DynamicIRF, response, impulse, time)
    # setup figure
    fig = Figure()
    ax = Axis3(
        fig[1,1], 
        title=orth(irfs) ? "Orthogonal Impulse Response Function" : "Impulse Response Function",
        titlealign=:left,
        titlecolor=:gray50,
        xlabel="time",
        ylabel="periods" 
    )

    # impulse response function
    surface!(ax, irf(irfs, response, impulse, time))

    return fig
end

function irf_plot(irfs::DynamicIRF, response, impulse, time::Integer)
    # setup figure
    fig = Figure()
    ax = Axis(
        fig[1,1], 
        title=orth(irfs) ? "Orthogonal Impulse Response Function" : "Impulse Response Function",
        titlealign=:left,
        titlecolor=:gray50,
        xlabel="periods" 
    )

    # impulse response function
    lines!(ax, irf(irfs, response, impulse, time), color=:black)

    # confidence bands
    lines!(ax, lower(irfs, response, impulse, time), color=:gray50, linestyle=:dash)
    lines!(ax, upper(irfs, response, impulse, time), color=:gray50, linestyle=:dash)

    return fig
end