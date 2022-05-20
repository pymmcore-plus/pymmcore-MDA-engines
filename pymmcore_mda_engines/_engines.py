from __future__ import annotations

__all__ = [
    "DriftCorrectionEngine",
]

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from pymmcore_plus.mda import MDAEngine

from ._image_generators import ImageGenerator

if TYPE_CHECKING:
    from pymmcore_plus import CMMCorePlus

from useq import MDASequence


class DriftCorrectionEngine(MDAEngine):
    def __init__(self, mmc: CMMCorePlus = None) -> None:
        super().__init__(mmc)
        self.drift_correction = defaultdict(lambda: np.array([0, 0], dtype=float))
        self._img_gen: ImageGenerator | None = None

    def register_image_generator(self, img_gen: ImageGenerator):
        """
        Register an ImageGenerator to use in place of the CMMCorePlus.snap method.

        This is useful when developing using the demo camera, as you can make more
        realistic images.

        Parameters
        ----------
        img_gen : ImageGenerator
        """
        self._img_gen = img_gen
        self._t = 0

    def _prepare_to_run(self, sequence: MDASequence):
        self._t = 0
        return super()._prepare_to_run(sequence)

    def run(self, sequence: MDASequence) -> None:
        """
        Run the multi-dimensional acquistion defined by `sequence`.
        Most users should not use this directly as it will block further
        execution. Instead use ``run_mda`` on CMMCorePlus which will run on
        a thread.

        Parameters
        ----------
        sequence : MDASequence
            The sequence of events to run.
        """
        self._prepare_to_run(sequence)

        for event in sequence:
            cancelled = self._wait_until_event(event, sequence)

            # If cancelled break out of the loop
            if cancelled:
                break

            logger.info(event)

            # change event to account for drift correction
            drift_adjustment = self.drift_correction.get(
                event.index.get("p", -1), np.array([0, 0])
            )
            event.x_pos -= drift_adjustment[0]
            event.y_pos -= drift_adjustment[1]

            self._prep_hardware(event)

            if self._img_gen is not None:
                event_t = event.index.get("t", 0)
                if event_t > self._t:
                    self._img_gen.step_positions()
                img = self._img_gen.snap_img((event.x_pos, event.y_pos), as_rgb=True)
            else:
                self._mmc.snapImage()
                img = self._mmc.getImage()

            self._events.frameReady.emit(img, event)
        self._finish_run(sequence)
