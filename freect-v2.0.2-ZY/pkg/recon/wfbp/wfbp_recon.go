package wfbp

import (
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/image"
	"gitlab.com/freect/freect/pkg/projections"
	"gitlab.com/freect/freect/pkg/recon"
	"gitlab.com/freect/freect/pkg/recon/filter"
	"gitlab.com/freect/freect/pkg/recon/rebin"
	"strings"
	"time"
)

func LogProgress(message string, silent bool) {
	if !silent {
		log.Info(message)
	}
}

func LogExecutionTime(message string, start time.Time, silent bool) {
	if !silent {
		log.Infof("%s in %v ", message, time.Since(start))
	}
}

// Verify the recon func meets the ReconstructionFunc signature
var _ recon.ReconstructionFunc = WFBPReconstructionFunc

func WFBPReconstructionFunc(config *recon.ReconConfig, wfbp_config interface{}) ([]*image.ReconstructedImage, error) {

	// Repack the ReconSpecific structure
	//
	// KLUDGE: Unfortunately, my brilliant plan for config handling of
	// different types of reconstruction configuration is a no-go
	// since the yaml decoder doesn't unpack the ReconSpecific struct
	// into the underlying type, but rather into a generic
	// map[string]any.  I sorta shoulda seen that coming.  For the
	// time being, since WFBP is the only one we're supporting, I'm
	// going to plow ahead with the current setup (still using the
	// interface) but just manually reset the structure in the
	// reconFunc.
	RestoreWFBPConfig(wfbp_config.(*WFBPConfig), config.ReconSpecific.(map[string]interface{}), config)

	// Load Raw Data
	r, err := config.RawInput.GetReader()
	if err != nil {
		return nil, err
	}

	log.Info("Reading raw data...")
	log.Info("Reader info: ", r.String())
	config_geom := config.ScanGeometry

	// We need a way to either
	start := time.Now()
	projs, geom, err := r.Read(config_geom)
	if err != nil {
		return nil, err
	}

	log.Info("Read raw data in ", time.Since(start))

	// Preprocessing
	preprocessed_projs, err := recon.Preprocess(config.Preprocessing, projs)
	if err != nil {
		return nil, err
	}

	// log.Infof("Saving preprocessed data ...")
	// projections.ProjectionStack(preprocessed_projs).SaveRaw("//mfad.mfroot.org/rchapp/eb028591/CT_CIC_Group_Server/Staff_Folders/Yu_Zitong/code_repositories/freeCT/siemens/Alpha/Head_phantom_sequential/preprocessed_projections.raw")

	// Rebin
	log.Info("Rebinning...")
	start = time.Now()
	var rebinned []*projections.Projection

	// Detect axial acquisitions and use an axial "rebin"
	// that avoids angular interpolation between projections.
	if strings.ToUpper(geom.ScanType) == "AXIAL" {
		rebinned, err = rebin.RebinAxial(geom, preprocessed_projs)
		if err != nil {
			return nil, err
		}
	} else {
		switch m := geom.FlyingFocalSpotMode; m {
		case "FFSNONE":
			rebinned, err = rebin.RebinNoFFS(geom, preprocessed_projs)
			if err != nil {
				return nil, err
			}
		default:
			log.Fatalf("FFS Mode \"%v\" unsupported", m)
		}
	}

	log.Infof("Rebinned data in %v", time.Since(start))

	// Clumsy.  Not 100% sure why this has to go here to get correct geometry.
	// TODO: Sort this question out.
	// Zitong: why do we need this?
	projections.ProjectionStack(rebinned).ReverseChannelOrder()

	// log.Infof("Saving rebinned data ...")
	// projections.ProjectionStack(rebinned).SaveRaw("//mfad.mfroot.org/rchapp/eb028591/CT_CIC_Group_Server/Staff_Folders/Yu_Zitong/code_repositories/freeCT/siemens/Alpha/Head_phantom_sequential/rebinned_projections.raw")

	log.Info("Filtering data...")
	start = time.Now()
	f := filter.NewNooFilter(geom, config.ReconSpecific.(*WFBPConfig).KernelC, config.ReconSpecific.(*WFBPConfig).KernelA)
	filtered_projs, err := filter.Filter(geom, rebinned, f)
	log.Infof("Filtered data in %v", time.Since(start))

	// Backproject
	log.Info("Backprojecting data...")
	start = time.Now()
	images, err := BackprojectNative(config, filtered_projs)
	if err != nil {
		return nil, err
	}
	log.Infof("Backprojected data in %v", time.Since(start))

	// Rescale to HU if configured to do so
	if !config.ImageOutput.RescaleToHU {
		return images, nil
	}

	for _, img := range images {
		err := img.RescaleToHU(config.ScanGeometry.HUCalibrationFactor)
		if err != nil {
			log.Error("Could not rescale to HU: ", err)
			break
		}
	}

	return images, nil
}
