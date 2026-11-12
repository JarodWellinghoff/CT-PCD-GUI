package wfbp

import (
	"fmt"
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/geometry"
	"gitlab.com/freect/freect/pkg/image"
	"gitlab.com/freect/freect/pkg/projections"
	"gitlab.com/freect/freect/pkg/recon"
	"gitlab.com/freect/freect/pkg/util"
	"math"
	"sync"
)

// All of the mathematics for this backprojection function comes from
// "Stierstorfer, K. et al. Weighted FBP - a simple approximate 3D FBP
// algorithm for multislice spiral CT with good dose usage for
// arbitrary pitch. Physics in Medicine and Biology 49, 2209–2218
// (2004)."

// Q == 0.6 is the recommended value from Stierstorfer et al. to balance
// minimization of helical artifacts with good dose utilization.
var Q float64 = 0.6

// Weighting
func W(q float64) float64 {
	q = math.Abs(q)
	if q < Q {
		return 1.0
	} else if Q <= q && q < 1.0 {
		wq := math.Cos(0.5 * math.Pi * (q - Q) / (1.0 - Q))
		return wq * wq
	} else {
		return 0.0
	}
}

// Weighting - Axial version
// Q = 1.0 since all data is wanted in an axial scan.
func W_axial(q float64) float64 {
	q = math.Abs(q)
	if q < 1.0 {
		return 1.0
	} else {
		return 0.0
	}
}

func TriangleWeight(dz float64, nominal_slice_thickness float64) float64 {
	return math.Max(0.0, 1.0-(math.Abs(dz)/nominal_slice_thickness))

}

func BackprojectInterp(projs []*projections.Projection, proj_idx int, chan_idx float64, row_idx float64) float64 {

	curr_proj := projs[proj_idx]

	q_idx := row_idx
	p_idx := chan_idx

	q_floor := int(math.Floor(q_idx))
	q_ceil := int(math.Ceil(q_idx))

	p_floor := int(math.Floor(p_idx))
	p_ceil := int(math.Ceil(p_idx))

	// Interpolate (note that the "At" method on the projection will handle clamping for us if we accidentally read out of bounds)
	w_q := q_idx - float64(q_floor)
	w_p := p_idx - float64(p_floor)

	return (1.0-w_q)*(1.0-w_p)*float64(curr_proj.At(q_floor, p_floor)) +
		(1.0-w_q)*(w_p)*float64(curr_proj.At(q_floor, p_ceil)) +
		(w_q)*(1.0-w_p)*float64(curr_proj.At(q_ceil, p_floor)) +
		(w_q)*(w_p)*float64(curr_proj.At(q_ceil, p_ceil))

	// NN interpolation
	//return float64(projs[proj_idx].At(int(math.Round(row_idx)), int(math.Round(chan_idx))))
}

func Backproject(volume *recon.ReconVolume, scan_geom *geometry.ScanGeometry, projs []*projections.Projection) ([]*image.ReconstructedImage, error) {

	// Warn if providing "unfinished" data
	if projs[0].Rebinned == false {
		log.Warn("Projections have not been rebinned prior to backprojection")
	}

	if projs[0].Filtered == false {
		log.Warn("Projections have not been filtered prior to backprojection")
	}

	// Determine where we need to reconstruct "native" slices
	// These will then be thicknessed to return our final reconstruction
	native_slice_thickness := scan_geom.DistSourceToIsocenter * scan_geom.DetectorAxialSpacing / scan_geom.DistSourceToDetector
	log.Info("(DEBUG) Computed native slice thickness for backprojection: ", native_slice_thickness)
	slice_locations := image.GenerateSliceLocations(volume.StartPosition, volume.EndPosition, native_slice_thickness)

	images := make([]*image.ReconstructedImage, len(slice_locations))

	p_nodata := *projs[0]
	p_nodata.Data = nil
	log.Info(fmt.Sprintf("Projection data: %+v", p_nodata))

	// Core backprojection mathematics
	for i, curr_slice_location := range slice_locations {

		curr_image := image.NewReconstructedImage(volume.Nx, volume.Ny)
		curr_image.XOrigin = volume.XOrigin
		curr_image.YOrigin = volume.YOrigin
		curr_image.Z = curr_slice_location
		curr_image.FieldOfView = volume.FieldOfView
		images[i] = curr_image

		acq_fov := scan_geom.AcquisitionFieldOfView
		collimated_slice_width := scan_geom.CollimatedSliceWidth
		detector_cone_offset := (float64(scan_geom.DetectorRows) - 1.0) / 2.0
		theta_cone := 2.0 * math.Atan(detector_cone_offset*collimated_slice_width/scan_geom.DistSourceToIsocenter)

		backprojectVoxel := func(workload []int, wg *sync.WaitGroup) {
			for _, curr_voxel_idx := range workload {

				// Determine the x/y locations of the current index
				x_idx := curr_voxel_idx % curr_image.Nx
				y_idx := curr_voxel_idx / curr_image.Nx

				x_loc := (float64(x_idx)-0.5*(float64(curr_image.Nx)-1.0))*curr_image.FieldOfView/float64(curr_image.Nx) + curr_image.XOrigin
				y_loc := (float64(y_idx)-0.5*(float64(curr_image.Ny)-1.0))*curr_image.FieldOfView/float64(curr_image.Ny) + curr_image.YOrigin
				z_loc := curr_image.Z

				z_rot := scan_geom.TableFeedPerRotation

				if (x_loc*x_loc + y_loc*y_loc) > ((0.5 * acq_fov) * (0.5 * acq_fov)) {
					continue
				}

				// Precompute some values
				projections_per_half_turn := scan_geom.ProjectionsPerRotation / 2
				n_half_turns := int(math.Floor(float64(scan_geom.NumProjectionsTotal) / float64(projections_per_half_turn)))
				// projection_central_channel := projs[0].CentralChannel - 4.0
				projection_central_channel := projs[0].CentralChannel // Zitong modified
				projection_num_rows := projs[0].NumRows
				projection_channel_width := projs[0].ChannelWidth

				//log.Info(projection_central_channel)
				//log.Info(projection_central_channel)

				for theta_tilde_idx := 0; theta_tilde_idx < projections_per_half_turn; theta_tilde_idx++ {
					backprojected_value := 0.0
					normalization_factor := 0.0

					for k := 0; k < n_half_turns; k++ {
						for q_idx := 0; q_idx < projection_num_rows; q_idx++ {

							curr_theta_idx := theta_tilde_idx + k*(projections_per_half_turn)
							curr_theta := projs[curr_theta_idx].TubeAngle + .5*math.Pi // This offset of pi/2 radians gives us correct alignment between the reconstructed image and the "intuitive" axes
							curr_table_position := projs[curr_theta_idx].TablePosition

							if math.Abs(curr_table_position-z_loc) > z_rot {
								continue
							}

							// For speed and simplicity, we leverage
							// the small angle approximation here to
							// compute q.  This may or may not be an appropriate
							// choice in all modern detectors.
							q := 2.0 * (float64(q_idx) - projs[curr_theta_idx].CentralRow) / (float64(projs[curr_theta_idx].NumRows) - 1.0)

							p_hat := x_loc*math.Sin(curr_theta) - y_loc*math.Cos(curr_theta)
							l_hat := math.Sqrt(scan_geom.DistSourceToIsocenter*scan_geom.DistSourceToIsocenter-p_hat*p_hat) - x_loc*math.Cos(curr_theta) - y_loc*math.Sin(curr_theta)

							z_ray_l_hat := curr_table_position - (z_rot/(2.0*math.Pi))*math.Asin(p_hat/scan_geom.DistSourceToIsocenter) + q*l_hat*math.Tan(0.5*theta_cone)
							dz := z_ray_l_hat - z_loc
							//h := TriangleWeight(dz, collimated_slice_width)
							h := TriangleWeight(dz, 10.0)
							if h == 0 {
								continue
							}

							// Interpolation
							pixel_scale := scan_geom.DistSourceToDetector / (scan_geom.DistSourceToIsocenter * projection_channel_width)
							p_idx := p_hat*pixel_scale + (projection_central_channel)

							interpolated_value := BackprojectInterp(projs, curr_theta_idx, p_idx, float64(q_idx))
							weight := W(q)

							backprojected_value += h * weight * interpolated_value
							normalization_factor += h * weight

						}
					}

					// ADD RESULT BACK INTO THE IMAGE
					curr_image.Data[curr_voxel_idx] += float32((1.0 / normalization_factor) * (2.0 * math.Pi / float64(scan_geom.ProjectionsPerRotation)) * backprojected_value)
				}
			}

			wg.Done()
		}
		util.LaunchParallelN(backprojectVoxel, curr_image.Nx*curr_image.Ny, util.MAX_PROC)
		//util.LaunchParallelN(backprojectVoxel, curr_image.Nx*curr_image.Ny, 1)
	}

	return images, nil
}
