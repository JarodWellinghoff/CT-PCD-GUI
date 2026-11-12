package rebin

import (
	// log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/geometry"
	"gitlab.com/freect/freect/pkg/projections"
	"gitlab.com/freect/freect/pkg/util"
	"math"
	"sync"
	//"time"
)

var _ RebinFunc = RebinNoFFS // Compile time check that we satisfy the rebin function signature

// During rebinning, channels are always oversampled to improved
// spatial resolution in the final image.

// All math on the CPU should be done with Float64 precision and only
// cast back to Float32 for the final storage back into our array.
// This is won't be true for GPU.

// TODO: I can't help but worry about the cache hit rate with this
// implementation.  Looks like it'll be super ugly.

// Although this implementation avoids an extra copy of the data, I
// think I'm likely to cause some confusion in here since I'm updating
// the projection data pointers quietly behind the scenes, and then
// still returning the original raw projection array.  I think this is
// likely to cause issues at some point in the future.
func RebinNoFFS(scan_geom *geometry.ScanGeometry, projs []*projections.Projection) ([]*projections.Projection, error) {

	n_rows := scan_geom.DetectorRows
	n_chan := scan_geom.DetectorChannels
	n_proj := len(projs)

	// Preallocate the output array (see note above regarding the 2*n_chan oversampling)
	rebinned_projs := make([]float32, n_rows*(2*n_chan)*n_proj)

	delta_theta := 2.0 * math.Pi / float64(scan_geom.ProjectionsPerRotation)

	// Currently, we only support upsampling to a factor of 2.0.  In
	// the future, it might be worthwhile to be able to rebin to
	// arbitrary upsampling dimensions.
	input_central_channel := projs[0].CentralChannel
	input_channel_width := projs[0].ChannelWidth
	output_central_channel := 2.0 * projs[0].CentralChannel
	output_channel_width := 0.5 * projs[0].ChannelWidth
	output_num_channels := 2 * n_chan

	// John's note: The cache hit efficiency of this (as is) is probably VERY low
	// Our main loop is over the *output* array
	//start := time.Now()
	rebinRows := func(row_idxs []int, wg *sync.WaitGroup) {
		for _, curr_row := range row_idxs {
			//log.Infof("Rebinning row %d/%d", curr_row+1, n_rows)
			util.LogUpdate(curr_row + 1, n_rows, 10, "rebinning")
			
			for curr_chan := 0; curr_chan < output_num_channels; curr_chan++ {
				for curr_proj := 0; curr_proj < n_proj; curr_proj++ {

					p := (float64(curr_chan) - output_central_channel) * (output_channel_width)

					beta := math.Asin(p / scan_geom.DistSourceToDetector)
					theta := float64(curr_proj) * delta_theta
					alpha := theta - beta

					beta_idx := beta*scan_geom.DistSourceToDetector/(input_channel_width) + input_central_channel
					alpha_idx := alpha / delta_theta

					out_idx := curr_proj*n_rows*(output_num_channels) + curr_row*(output_num_channels) + curr_chan

					rebinned_projs[out_idx] = float32(RebinInterp(projs, alpha_idx, beta_idx, curr_row))
				}
			}
		}
		wg.Done()
	}
	rows := make([]int, n_rows) // Changed this from rows := make([]int, 64) to avoid out of range error when n_rows > 64, e.g., for a Siemens Alpha scanner with 144 rows.
	for i := 0; i < n_rows; i++ {
		rows[i] = i
	}
	util.LaunchParallel(rebinRows, rows, util.MAX_PROC)

	// Repack our projections
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
