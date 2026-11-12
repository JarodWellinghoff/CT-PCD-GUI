package wfbp

import (
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/image"
	"gitlab.com/freect/freect/pkg/projections"
	"gitlab.com/freect/freect/pkg/recon"
	"gitlab.com/freect/freect/pkg/util"
	"math"
	"sync"
	"fmt"
)

// All of the mathematics for this backprojection function comes from
// "Stierstorfer, K. et al. Weighted FBP - a simple approximate 3D FBP
// algorithm for multislice spiral CT with good dose usage for
// arbitrary pitch. Physics in Medicine and Biology 49, 2209–2218
// (2004)."

//func BackprojectNative(volume *recon.ReconVolume, scan_geom *geometry.ScanGeometry, projs []*projections.Projection) ([]*image.ReconstructedImage, error) {
func BackprojectNative(config *recon.ReconConfig, projs []*projections.Projection) ([]*image.ReconstructedImage, error) {

	image_output := config.ImageOutput
	volume := config.ReconVolume
	scan_geom := config.ScanGeometry
	
	// Warn the user of any of the following cases
	if image_output.AngleOffset != 0 {
		log.Warn("ImageOutput.AngleOffset is non-zero.  This can break the intuitive understanding of XOrigin and YOrigin directions. Please double check your results.")
	}

	if projs[0].Rebinned == false {
		log.Warn("Projections have not been rebinned prior to backprojection")
	}

	if projs[0].Filtered == false {
		log.Warn("Projections have not been filtered prior to backprojection")
	}

	// Determine where we need to reconstruct "native" slices These
	// will then be thicknessed to return our final reconstruction
	// Note that we pad the start/stop position since extra slices
	// will be needed to correctly thickness the first/last slices.
	native_slice_thickness := scan_geom.DistSourceToIsocenter * scan_geom.DetectorAxialSpacing / scan_geom.DistSourceToDetector
	pad := volume.SliceThickness
	slice_locations := image.GenerateSliceLocationsForNativeBackprojection(volume.StartPosition, volume.EndPosition, native_slice_thickness, pad)
	
	images := make([]*image.ReconstructedImage, len(slice_locations))

	p_nodata := *projs[0]
	p_nodata.Data = nil
	
	// Core backprojection mathematics
	for i, curr_slice_location := range slice_locations {

		//pct := 100.0 * float64(i)/float64(len(slice_locations))
		//log.Infof(, curr_slice_location, i+1, len(slice_locations), pct)

		util.LogUpdate(i+1, len(slice_locations), len(slice_locations), fmt.Sprintf("reconstructing slice at %.03f ", curr_slice_location))

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

				for theta_tilde_idx := 0; theta_tilde_idx < projections_per_half_turn; theta_tilde_idx++ {
					backprojected_value := 0.0
					normalization_factor := 0.0

					for k := 0; k < n_half_turns; k++ {

						curr_theta_idx := theta_tilde_idx + k*(projections_per_half_turn)
						// John's thought: Based on the geometry
						// specified in Battelle, I had sort of
						// convinced myself that we needed to take the
						// negative of the tube angle to get correct
						// geometry for WFBP.  That however does *not*
						// work and just creates that really annoying,
						// tricky rib overlay artifact/wrong
						// reconstruction.  I'm not 100% sure that
						// what's in the file is 100% correct, so I
						// could be right that we need to flip *if*
						// the projection metadata is exactly what's
						// specified in the paper.  Should discuss this with Fred.
						//
						// THIS DOES NOT WORK, but would be what I would theoretically expect
						// curr_theta := 1.5 * math.Pi - projs[curr_theta_idx].TubeAngle 

						// Zitong's note:
						// What John had originally thought is correct.
						// However, curr_theta that we set for DICOM-CT-PD is already flipped,
						// so we don't need to flip it again.

						// This offset of pi/2 radians gives us correct alignment
						// between the reconstructed image and the "intuitive" axes
						// curr_theta := projs[curr_theta_idx].TubeAngle + .5*math.Pi
						curr_theta := projs[curr_theta_idx].TubeAngle + .5*math.Pi

						
						curr_table_position := projs[curr_theta_idx].TablePosition

						// Rotate the image at user request
						curr_theta += image_output.AngleOffset

						if math.Abs(curr_table_position-z_loc) > z_rot {
							continue
						}

						p_hat := x_loc*math.Sin(curr_theta) - y_loc*math.Cos(curr_theta)
						l_hat := math.Sqrt(scan_geom.DistSourceToIsocenter*scan_geom.DistSourceToIsocenter-p_hat*p_hat) - x_loc*math.Cos(curr_theta) - y_loc*math.Sin(curr_theta)
						q_hat := (z_loc - curr_table_position + (z_rot/(2.0*math.Pi))*math.Asin(p_hat/scan_geom.DistSourceToIsocenter)) / (l_hat * math.Tan(0.5*theta_cone))

						// Validate that the current projection has a ray that intersects the voxel
						if (q_hat > 1.0) || (q_hat < -1.0) {
							continue
						}

						// Interpolation
						pixel_scale := scan_geom.DistSourceToDetector / (scan_geom.DistSourceToIsocenter * projection_channel_width)
						p_idx := p_hat*pixel_scale + (projection_central_channel)
						q_idx := 0.5 * (q_hat + 1.0) * (float64(projection_num_rows) - 1.0) // TODO: Need to account for ZFFS

						interpolated_value := BackprojectInterp(projs, curr_theta_idx, p_idx, q_idx)
						// weight := W(q_hat)
						weight := W_axial(q_hat)

						backprojected_value += weight * interpolated_value
						normalization_factor += weight

					}


					// Add result back into the image
					curr_image.Data[curr_voxel_idx] += float32((1.0 / normalization_factor) * (2.0 * math.Pi / float64(scan_geom.ProjectionsPerRotation)) * backprojected_value)
				}
			}

			wg.Done()
		}
		util.LaunchParallelN(backprojectVoxel, curr_image.Nx*curr_image.Ny, util.MAX_PROC)
	}

	// Thickness
	final_images, err := recon.Thickness(volume, images, recon.TriangleWindow)
	if err != nil {
		return nil, err
	}

	// up-down flip of backprojected value.
	// Zitong: I found that the reconstructed image is flipped up-down compared to the reference image, so I added this flip here.
	for _, img := range final_images {
		for row := 0; row < img.Ny/2; row++ {
			offset_top := row * img.Nx
			offset_bottom := (img.Ny - 1 - row) * img.Nx
			for col := 0; col < img.Nx; col++ {
				img.Data[offset_top+col], img.Data[offset_bottom+col] = img.Data[offset_bottom+col], img.Data[offset_top+col]
			}
		}
	}

	// left-right flip of backprojected value.
	// Zitong: I found that the reconstructed image is flipped left-right compared to the reference image, so I added this flip here.
	for _, img := range final_images {
		for col := 0; col < img.Nx/2; col++ {
			offset_left := col
			offset_right := img.Nx - 1 - col
			for row := 0; row < img.Ny; row++ {
				img.Data[row*img.Nx+offset_left], img.Data[row*img.Nx+offset_right] = img.Data[row*img.Nx+offset_right], img.Data[row*img.Nx+offset_left]
			}
		}
	}

	return final_images, nil
}
