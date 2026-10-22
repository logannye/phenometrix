# One bounded browser YIN arithmetic repair

Declared after the first reserved-input arithmetic check and before a revised
implementation result. Independent natural reference files remain unacquired.
The frozen offline model, prediction protocol, sample selection, gates and raw
YIN reference streams are unchanged.

The first browser port used double-precision FFT followed by float32 energy,
autocorrelation and cumulative-difference intermediates. It passed 4,420 analytic
and full-development frames, then failed one of 10,812 reserved-input raw-Hz
checks. In F06_sx264 first-difference frame 666, two global CMND minima differ
by about 1e-7; small FFT rounding differences reverse their order, yielding
Python 50 Hz and browser approximately 53.3834 Hz. Both final outputs are
unvoiced, with raw MLP probability approximately 0.00393. This does not excuse
the failed all-frame raw-Hz gate. The first run and its numerical diagnosis
remain failed evidence under their exact source identity.

One implementation repair may be tested: use consistent float64 internal
arithmetic for FFT, energy, difference and cumulative normalization, preserving
float32 input samples and the unchanged YIN equations, period bounds, trough
threshold, parabolic rule and hybrid combination. This removes the mixed
precision stages instead of fitting a tie epsilon or selecting a preferred lag.
It is a numerical implementation revision, not a new model or reference shift.

First check the unchanged analytic/full-development fixtures. Only if they pass
the original tolerances, check all frames of the existing fixed reserved-input
fixtures, retaining every result. No precision/FFT-length sweep, endpoint
special case, input-specific override, voicing-based exclusion, replacement
fixture, threshold change or tolerance relaxation is allowed. Raw Hz remains
checked even on unvoiced frames. Final state parity alone cannot pass the raw
YIN component. If this single repair fails, preserve the discrepancy and report
that component as unqualified rather than continuing arithmetic search.

Any repaired implementation receives a distinct source/version identity and
repeats relevant unit, numerical, actual-worklet, cancellation and overload
checks. Sustained qualification requires the unchanged declared runtime gates.
The reserved natural accuracy estimate still uses the original frozen Python
estimator; any browser approximation remains separately identified and bounded.
