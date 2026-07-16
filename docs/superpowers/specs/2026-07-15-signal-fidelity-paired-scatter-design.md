# Paired signal-fidelity scatter design

## Conclusion

The figure tests whether FGM produces an empirical audio contribution, $E[\Delta_A]$, that is closer to the analytic conditional mutual information than the perturbed No-FGM baseline.

## Figure contract

- Archetype: single-panel quantitative comparison.
- Role: validation/comparison figure.
- Backend: Python (`pytorch2.5`) with matplotlib only.
- Final size: approximately 89 mm square, suitable for a single journal column.
- Inputs: `signal_fidelity_fgm.json` and `signal_fidelity_no_fgm_perturbed.json`.
- Exports: editable SVG, PDF, and 300-dpi PNG preview.

## Data mapping

- x: `true_cmi`, in bit.
- y: `delta_audio_tail / 0.6931471805599453`, converting nats to bit.
- Pair records by `(true_cmi, s, eta, seed)`; reject missing or duplicate pairs.
- Do not read accuracy fields from `final_val` for scatter coordinates.
- Use one shared `upper = max(max(x), max(y))` across both methods and plot the reference line from `(0, 0)` to `(upper, upper)`.

## Visual encoding

- FGM: open bright-blue circles.
- No-FGM + perturbation: open rose-pink circles.
- Matched records: thin neutral-grey segments drawn beneath the markers.
- Reference: thin black dashed `y=x` line.
- Equal x/y limits and equal aspect ratio; exact requested axis labels; frameless legend; white background; subtle light-grey grid.

## Validation

- Unit test the nats-to-bits conversion, exact pairing, duplicate/missing-pair rejection, shared upper bound, and use of the two requested JSON fields.
- Run the plotting script in `pytorch2.5` and verify all three outputs exist.
- Inspect the rendered PNG for clipping, overlap, legibility, equal axes, paired connectors, and correct legend labels.

## Reviewer risks

- Pairing by record order could silently connect unrelated conditions; use explicit condition keys instead.
- Plotting nats against bit would distort fidelity; enforce conversion before plotting.
- Marker overlap could hide one method; use distinct shape/fill encodings and draw connectors first.
