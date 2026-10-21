from .interfaces import Grid, Pair, TaskContext, Prediction, SolverResult, BaseSolver
from .grid_codec import (
    grid_to_text, text_to_grid, grid_dims, grid_equal,
    copy_grid, rotate90, flip_h, flip_v, transpose,
    color_histogram, background_color,
)
from .augmentation import eight_views, inverse_view
from .submission import build_submission, validate_submission, write_submission, score_submission

__all__ = [
    "Grid", "Pair", "TaskContext", "Prediction", "SolverResult", "BaseSolver",
    "grid_to_text", "text_to_grid", "grid_dims", "grid_equal",
    "copy_grid", "rotate90", "flip_h", "flip_v", "transpose",
    "color_histogram", "background_color",
    "eight_views", "inverse_view",
    "build_submission", "validate_submission", "write_submission", "score_submission",
]
