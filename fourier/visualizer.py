import numpy as np
from manim import *


class FourierMath:
    """Pure-numpy part of the original class (no plotting)."""

    def __init__(self, freqs, tp=3, n_samples=1000, amplitude=1.0):
        self.time = np.linspace(0, tp, n_samples)
        self.freqs = np.asarray(freqs, dtype=float)
        self.phi = np.zeros_like(self.freqs)
        # sum all frequency waveforms to create a combined waveform
        self.y = np.sum(
            [amplitude * np.sin(2 * np.pi * f * self.time + p) + 1 for f, p in zip(self.freqs, self.phi)],
            axis=0,
        )

    def wound(self, winding_freq):
        # Convert linear to polar coordinates
        return np.exp(-2j * np.pi * winding_freq * self.time) * self.y

    def center_of_mass(self, winding_freq):
        return np.mean(self.wound(winding_freq))


class FourierTransformScene(Scene):
    SIGNAL_FREQS = [2, 5]
    MAX_FREQ = 10
    STEP = 0.025
    RUN_TIME = 20

    def construct(self):
        fm = FourierMath(self.SIGNAL_FREQS, amplitude=1.5)

        # Precompute the center-of-mass curve
        grid = np.arange(0.01, self.MAX_FREQ + 1e-9, self.STEP)
        com_vals = np.array([abs(fm.center_of_mass(f)) for f in grid])

        left = Axes(
            x_range=[-5, 5, 1], y_range=[-5, 5, 1],
            x_length=6, y_length=6,
            axis_config={"include_ticks": False, "stroke_opacity": 0.5},
            tips=False,
        ).to_edge(LEFT, buff=0.7).shift(DOWN * 0.3)

        right = Axes(
            x_range=[0, self.MAX_FREQ, 1], y_range=[0, 2.2, 0.5],
            x_length=6.5, y_length=4,
            axis_config={"include_ticks": True, "tick_size": 0.06},
            tips=False,
        ).to_edge(RIGHT, buff=0.6).shift(DOWN * 0.3)

        x_ticks = VGroup(*[
            Text(str(i), font_size=16).next_to(right.c2p(i, 0), DOWN, buff=0.12)
            for i in range(0, self.MAX_FREQ + 1)
        ])
        x_label = Text("Frequency", font_size=22).next_to(right, DOWN, buff=0.45)
        left_label = Text("Wound signal", font_size=22).next_to(left, UP, buff=0.2)
        right_label = Text("Center of mass (distance from origin)", font_size=22).next_to(right, UP, buff=0.2)

        self.add(left, right, x_ticks, x_label, left_label, right_label)

        def to_scene(axes, re, im):
            """Vectorized data -> scene coords."""
            origin = axes.c2p(0, 0)
            ux = axes.c2p(1, 0) - origin
            uy = axes.c2p(0, 1) - origin
            return origin + np.outer(re, ux) + np.outer(im, uy)

        f_tracker = ValueTracker(0.01)

        def winding_curve():
            z = fm.wound(f_tracker.get_value())
            pts = to_scene(left, z.real, z.imag)
            curve = VMobject(stroke_width=1.5, stroke_color=BLUE)
            curve.set_points_as_corners(pts)
            return curve

        def com_dot():
            c = fm.center_of_mass(f_tracker.get_value())
            return Dot(to_scene(left, [c.real], [c.imag])[0], color=RED, radius=0.07)

        def com_trace():
            idx = np.searchsorted(grid, f_tracker.get_value(), side="right")
            trace = VMobject(stroke_width=3, stroke_color=RED)
            if idx >= 2:
                trace.set_points_as_corners(to_scene(right, grid[:idx], com_vals[:idx]))
            return trace

        def freq_title():
            return Text(
                f"Frequency: {f_tracker.get_value():.2f}", font_size=30
            ).to_edge(UP, buff=0.3)

        self.add(
            always_redraw(winding_curve),
            always_redraw(com_dot),
            always_redraw(com_trace),
            always_redraw(freq_title),
        )

        self.play(
            f_tracker.animate.set_value(self.MAX_FREQ),
            run_time=self.RUN_TIME,
            rate_func=linear,
        )
        self.wait(1)