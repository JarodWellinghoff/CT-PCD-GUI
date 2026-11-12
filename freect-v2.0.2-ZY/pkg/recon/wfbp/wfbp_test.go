package wfbp

import (
	"fmt"
	"gitlab.com/freect/freect/pkg/recon"
	"gopkg.in/yaml.v3"
	"math"
	"testing"
)

var test_data_dir = "/Users/jhoffman/Data/ldct/sample_case/"

func TestWFBP(t *testing.T) {

	wfbp_config := DefaultWFBPConfig()
	wfbp_config.KernelA = 0.5
	wfbp_config.KernelC = 0.75

	recon_config := recon.NewReconConfig(wfbp_config)
	recon_config.ReconSpecific = wfbp_config

	recon_config.RawInput.Battelle = test_data_dir
	recon_config, err := recon.GenerateBaseConfig(recon_config)
	if err != nil {
		t.Fatal(err)
	}

	// Adjust the recon volume to be something reasonable for a test.
	// We'll do a few slices around the middle of the total volume
	midpoint := (recon_config.ReconVolume.StartPosition + recon_config.ReconVolume.EndPosition) / 2.0
	recon_config.ReconVolume.StartPosition = midpoint + 2.0 // expect 3 slices?
	recon_config.ReconVolume.EndPosition = midpoint - 2.0
	recon_config.ReconVolume.SliceThickness = 2.0
	recon_config.ReconVolume.SliceSpacing = 2.0
	recon_config.ReconVolume.Nx = 512
	recon_config.ReconVolume.Ny = 512
	recon_config.ReconVolume.FieldOfView = 300.0

	recon_config.ImageOutput.RescaleToHU = true
	recon_config.ImageOutput.AngleOffset = math.Pi / 2

	//recon_config.Preprocessing.FlipDetectorChannels = true
	//recon_config.Preprocessing.FlipDetectorChannels = true

	data, err := yaml.Marshal(recon_config)
	if err != nil {
		t.Error(err)
	}
	t.Log(string(data))

	images, err := WFBPReconstructionFunc(recon_config)
	if err != nil {
		t.Log(err)
	}

	for i, img := range images {
		err := img.SaveBinary(fmt.Sprintf("slice_%03d.bin", i))
		if err != nil {
			t.Error(err)
		}
	}

}
