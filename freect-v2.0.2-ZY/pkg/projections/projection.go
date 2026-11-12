package projections

import (
	"math"
)

// Projection stores the pixel data and metadata for a single CT projection
//
// Projection pixel data shall be stored ROW MAJOR (i.e. data_idx :=
// row_idx * n_channels + channel_idx)
//
// Unlike ScanGeometry, projection metadata should be understood to
// reflect the *current* state of the projection data held, and evolve
// throughout the reconstruction process.  E.g., when upsampling our
// detector during rebinning, we (1) double the number of rows
// (NumRows -> 2*NumRows), (2) halve the physical channel size
// (ChannelWidth -> ChannelWidth/2.0), and (3) relocate the central
// channel (CentralChannel -> 2.0 * CentralChannel).
type Projection struct {
	Index          int
	Data           []float32
	NumRows        int
	NumChannels    int
	CentralRow     float64
	CentralChannel float64
	ChannelWidth   float64 // Physical size of one channel in millimeters
	RowWidth       float64 // Physical size of one row in millimeters
	TablePosition  float64 // Millimeters
	TubeAngle      float64 // radians
	Rebinned       bool
	Filtered       bool
}

// At returns the pixel data at coordinates row, channel.
//
// To limit the additional code complexity returning an error
// generates, any read outside of projection bounds gets CLAMPED to
// the edge values of the projection. This may not always be
// desirable.
// TODO Offer selectable behavior for reads outside of the data?
func (p *Projection) At(row int, channel int) float32 {
	row = int(math.Min(math.Max(float64(row), 0.0), float64(p.NumRows-1)))
	channel = int(math.Min(math.Max(float64(channel), 0.0), float64(p.NumChannels-1)))
	return p.Data[channel+row*p.NumChannels]
}

// ReverseChannelOrder flips the channel data for the projection.
// I.e. the data in index 0 is swapped with index N-1, 1 <-> N-2, ...
// We also update the central channel index accordingly.
//
// ReverseChannelOrder assumes 0-based indexing for the data.  If
// applying both CorrectMATLABIndexing *and* ReverseChannelOrder,
// CorrectMATLABIndexing *must* be applied first (ref. Battelle format
// reader)
func (p *Projection) ReverseChannelOrder() {
	for row := 0; row < p.NumRows; row++ {
		offset := row * p.NumChannels
		for i, j := 0, p.NumChannels-1; i < j; i, j = i+1, j-1 {
			p.Data[offset+i], p.Data[offset+j] = p.Data[offset+j], p.Data[offset+i]
		}
	}
	p.CentralChannel = (float64(p.NumChannels) - 1.0) - p.CentralChannel
}

// ReverseChannelOrder flips the row ordering for the projection
// I.e. the data in index 0 is swapped with index N-1, 1 <-> N-2, ...
// We also update the central row index accordingly.
func (p *Projection) ReverseRowOrder() {

	tmp := make([]float32, p.NumChannels)

	for i := 0; i < p.NumRows/2; i++ {
		a_row_idx := i
		b_row_idx := p.NumRows - 1 - i

		a_offset := a_row_idx * p.NumChannels
		a_row := p.Data[a_offset : a_offset+p.NumChannels]

		b_offset := b_row_idx * p.NumChannels
		b_row := p.Data[b_offset : b_offset+p.NumChannels]

		copy(tmp, a_row)
		copy(a_row, b_row)
		copy(b_row, tmp)
	}

}

// CorrectMATLABIndexing converts projection's central detector
// coordinates to zero-based indexing. Note: NOT idempotent.
func (p *Projection) CorrectMATLABIndexing() {
	p.CentralChannel -= 1.0
	p.CentralRow -= 1.0
}
