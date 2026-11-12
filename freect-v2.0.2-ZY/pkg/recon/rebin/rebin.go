package rebin

import (
	"gitlab.com/freect/freect/pkg/geometry"
	"gitlab.com/freect/freect/pkg/projections"
	"math"
)

// Some package-level definitions we'll use a fair bit

type RebinFunc func(*geometry.ScanGeometry, []*projections.Projection) ([]*projections.Projection, error)

// Interpolate between proj/chan (with fixed row index)
func RebinInterp(projs []*projections.Projection, proj_idx float64, chan_idx float64, row_idx int) float64 {

	//proj_idx_low := int(math.Floor(proj_idx))
	proj_idx_low := int(proj_idx)
	proj_idx_high := proj_idx_low + 1
	delta_proj := proj_idx - float64(proj_idx_low)

	//chan_idx_low := int(math.Floor(chan_idx))
	chan_idx_low := int(chan_idx)
	chan_idx_high := chan_idx_low + 1
	delta_chan := chan_idx - float64(chan_idx_low)

	// Project "At" method clamps row/channel values to projection borders
	// We still need to explicitly clamp values for the projection index
	n_projs := len(projs)
	proj_idx_low = int(math.Min(math.Max(float64(proj_idx_low), 0.0), float64(n_projs-1)))
	proj_idx_high = int(math.Min(math.Max(float64(proj_idx_high), 0.0), float64(n_projs-1)))

	interp_val := float64(projs[proj_idx_low].At(row_idx, chan_idx_low))*(1.0-delta_proj)*(1.0-delta_chan) +
		float64(projs[proj_idx_low].At(row_idx, chan_idx_high))*(1.0-delta_proj)*(delta_chan) +
		float64(projs[proj_idx_high].At(row_idx, chan_idx_low))*(delta_proj)*(1.0-delta_chan) +
		float64(projs[proj_idx_high].At(row_idx, chan_idx_high))*(delta_proj)*(delta_chan)

	return interp_val
}
