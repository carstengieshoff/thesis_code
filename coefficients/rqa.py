from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyrqa
from pyrqa.analysis_type import Classic
from pyrqa.computation import RPComputation, RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius, Unthresholded
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries

param_names = [
    "Minimum diagonal line length (L_min)",
    "Minimum vertical line length (V_min)",
    "Minimum white vertical line length (W_min)",
    "Recurrence rate (RR)",
    "Determinism (DET)",
    "Average diagonal line length (L)",
    "Longest diagonal line length (L_max)",
    "Divergence (DIV)",
    "Entropy diagonal lines (L_entr)",
    "Laminarity (LAM)",
    "Trapping time (TT)",
    "Longest vertical line length (V_max)",
    "Entropy vertical lines (V_entr)",
    "Average white vertical line length (W)",
    "Longest white vertical line length (W_max)",
    "Longest white vertical line length inverse (W_div)",
    "Entropy white vertical lines (W_entr)",
    "Ratio determinism / recurrence rate (DET/RR)",
    "Ratio laminarity / determinism (LAM/DET)",
]


class RQAGenerator:
    """Wrapper for pyrqa: https://pypi.org/project/PyRQA/."""

    def __init__(
        self,
        min_diagonal_line_length: int = 2,
        min_vertical_line_length: int = 2,
        min_white_vertical_line_length: int = 2,
        params: Optional[List[str]] = None,
    ) -> None:
        self.min_diagonal_line_length = min_diagonal_line_length
        self.min_vertical_line_length = min_vertical_line_length
        self.min_white_vertical_line_length = min_white_vertical_line_length
        if params is not None and not all(param in param_names for param in params):
            raise ValueError("Unknown parameter names")
        self.params = params

    def generate(
        self, x: np.array, embedding_dim: int = 20, lag: int = 1, eps: float = 0.1, plot_rp: bool = False
    ) -> np.array:

        time_series = TimeSeries(x, embedding_dimension=embedding_dim, time_delay=lag)
        settings = Settings(
            time_series,
            analysis_type=Classic,
            neighbourhood=Unthresholded(),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1,
        )

        computation = RPComputation.create(settings)
        result = computation.run()

        if plot_rp:
            plt.imshow(result.recurrence_matrix_reverse_normalized, cmap="viridis")
            plt.colorbar()
            plt.show()
        _eps = np.quantile(result.recurrence_matrix.reshape(-1), q=eps)

        settings = Settings(
            time_series,
            analysis_type=Classic,
            neighbourhood=FixedRadius(_eps),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1,
        )
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
        result.min_diagonal_line_length = self.min_diagonal_line_length
        result.min_vertical_line_length = self.min_vertical_line_length
        result.min_white_vertical_line_length = self.min_white_vertical_line_length

        result_dict = {
            "Minimum diagonal line length (L_min)": result.min_diagonal_line_length,
            "Minimum vertical line length (V_min)": result.min_vertical_line_length,
            "Minimum white vertical line length (W_min)": result.min_white_vertical_line_length,
            "Recurrence rate (RR)": result.recurrence_rate,
            "Determinism (DET)": result.determinism,
            "Average diagonal line length (L)": result.average_diagonal_line,
            "Longest diagonal line length (L_max)": result.longest_diagonal_line,
            "Divergence (DIV)": result.divergence,
            "Entropy diagonal lines (L_entr)": result.entropy_diagonal_lines,
            "Laminarity (LAM)": result.laminarity,
            "Trapping time (TT)": result.trapping_time,
            "Longest vertical line length (V_max)": result.longest_vertical_line,
            "Entropy vertical lines (V_entr)": result.entropy_vertical_lines,
            "Average white vertical line length (W)": result.average_white_vertical_line,
            "Longest white vertical line length (W_max)": result.longest_white_vertical_line,
            "Longest white vertical line length inverse (W_div)": result.longest_white_vertical_line_inverse,
            "Entropy white vertical lines (W_entr)": result.entropy_white_vertical_lines,
            "Ratio determinism / recurrence rate (DET/RR)": result.ratio_determinism_recurrence_rate,
            "Ratio laminarity / determinism (LAM/DET)": result.ratio_laminarity_determinism,
        }
        if self.params is None:
            return np.array(list(result_dict.values()))
        else:
            return np.array(list(result_dict[key] for key in self.params))

    @property
    def available_params(self) -> List[str]:
        return param_names


if __name__ == "__main__":
    from signals import AAGP
    from signals.GP_kernels import organized_aa_args

    Fs = 120
    aa = AAGP(organized_aa_args, sampling_rate=Fs, sec=10)
    aa.generate(num_samples=1)
    aa.show()

    generator = RQAGenerator(
        params=[
            "Determinism (DET)",
            "Average diagonal line length (L)",
        ]
    )
    print(generator.generate(aa.data, plot_rp=True))
