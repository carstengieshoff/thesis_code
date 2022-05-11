from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_ecg_plotly(
    original: np.array,
    aa: Optional[np.array] = None,
    peaks: Optional[np.array] = None,
    front: int = 0,
    back: int = 0,
    H: int = 0,
    lw: float = 1.8,
    title: str = "QRST Cancellation",
) -> None:
    num_leads = original.shape[1]

    fig = make_subplots(rows=num_leads, cols=1, subplot_titles=[f"Lead {i + 1}" for i in range(num_leads)])

    for i in range(num_leads):
        row = i + 1
        fig.add_trace(go.Line(y=original[:, i], name="Original ECG", line_color="Blue", line_width=lw), row=row, col=1)
        if peaks is not None:
            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    x=peaks,
                    y=original[peaks, i],
                    name="R-Peaks",
                    marker=dict(
                        color="Red",
                        size=5,
                    ),
                ),
                row=row,
                col=1,
            )

        if aa is not None:
            fig.add_trace(go.Line(y=aa[:, i], name="AA", line_color="Green", line_width=lw), row=row, col=1)

            fig.add_trace(
                go.Line(y=original[:, i] - aa[:, i], name="VA", line_color="red", line_width=lw / 2), row=row, col=1
            )

    if peaks is not None and front is not None and back is not None:

        for peak in peaks:
            fig.add_vrect(
                x0=peak - front,
                x1=peak + back,
                fillcolor="Lightblue",
                opacity=1,
                layer="below",
                line_width=0,
                name="QRST-Window",
            )

    if peaks is not None and H is not None:
        for peak in peaks:
            fig.add_vrect(
                x0=peak - H,
                x1=peak + H,
                fillcolor="Darkblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

    fig.update_layout(height=num_leads * 300, width=1000, title_text=title)
    fig.show()
