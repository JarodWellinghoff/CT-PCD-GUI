package rebin

import (
    "gitlab.com/freect/freect/pkg/geometry"
    "gitlab.com/freect/freect/pkg/projections"
    "gitlab.com/freect/freect/pkg/util"
    "math"
    "sync"
)

// Zitong: 
// RebinAxial handles sequential projections by oversampling
// detector channels exactly like RebinNoFFS but NOT performing angular
// interpolation between neighboring projection angles. This preserves
// output shape and metadata while avoiding mixing information across
// projection angles which is inappropriate for axial data.
// It is called RebinAxial to follow the naming convention of the one 
// used for helical scan mode, although it is not a "rebin" method.
func RebinAxial(scan_geom *geometry.ScanGeometry, projs []*projections.Projection) ([]*projections.Projection, error) {

	n_rows := scan_geom.DetectorRows
	n_chan := scan_geom.DetectorChannels
	n_proj := len(projs)

	// Preallocate the output array (channels oversampled by factor 2)
	rebinned_projs := make([]float32, n_rows*(2*n_chan)*n_proj)

	// Setup input/output channel parameters to match RebinNoFFS
	input_central_channel := projs[0].CentralChannel
	input_channel_width := projs[0].ChannelWidth
	output_central_channel := 2.0 * projs[0].CentralChannel
	output_channel_width := 0.5 * projs[0].ChannelWidth
	output_num_channels := 2 * n_chan

	rebinRows := func(row_idxs []int, wg *sync.WaitGroup) {
		for _, curr_row := range row_idxs {
			util.LogUpdate(curr_row+1, n_rows, 10, "rebinning (axial)")

			for curr_chan := 0; curr_chan < output_num_channels; curr_chan++ {
				for curr_proj := 0; curr_proj < n_proj; curr_proj++ {

					p := (float64(curr_chan) - output_central_channel) * (output_channel_width)

					beta := math.Asin(p / scan_geom.DistSourceToDetector)
					// For axial, do not shift between projection angles: use
					// the current projection directly (no alpha interpolation).
					beta_idx := beta*scan_geom.DistSourceToDetector/(input_channel_width) + input_central_channel
					alpha_idx := float64(curr_proj)

					out_idx := curr_proj*n_rows*(output_num_channels) + curr_row*(output_num_channels) + curr_chan

					rebinned_projs[out_idx] = float32(RebinInterp(projs, alpha_idx, beta_idx, curr_row))
				}
			}
		}
		wg.Done()
	}

	rows := make([]int, n_rows)
	for i := 0; i < n_rows; i++ {
		rows[i] = i
	}
	util.LaunchParallel(rebinRows, rows, util.MAX_PROC)

	// Repack our projections and update metadata
	for i, p := range projs {
		p.CentralChannel = output_central_channel
		p.ChannelWidth = output_channel_width
		p.NumChannels = output_num_channels
		p.Rebinned = true

		start_slice := i * n_rows * output_num_channels
		end_slice := start_slice + n_rows*output_num_channels

		p.Data = rebinned_projs[start_slice:end_slice]
	}

	return projs, nil
}
