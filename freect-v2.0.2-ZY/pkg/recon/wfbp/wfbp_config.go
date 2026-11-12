package wfbp

import (
	log "github.com/sirupsen/logrus"
	"gitlab.com/freect/freect/pkg/recon"
)

// For a Ramp kernel KernelA == 1.0 and KernelC == 1.0
type WFBPConfig struct {
	KernelA float64 `json:"KernelA" yaml:"KernelA"` // a := {0.5, 0.54, 1.0}
	KernelC float64 `json:"KernelC" yaml:"KernelC"` // c := [0, 1.0]
}

func DefaultWFBPConfig() *WFBPConfig {
	return &WFBPConfig{
		KernelA: 0.5,
		KernelC: 0.5,
	}
}

// Inherent problems with this approach: we have no type safety about
// the incoming map parameters
//
// We could do this more generally with reflection, but that's going
// to be challenging
func RestoreWFBPConfig(overrides *WFBPConfig, file_parameters map[string]interface{}, config *recon.ReconConfig) {

	// Precedence here is:
	// (1) non-zero value in "overrides"
	// (2) non-zero value in file_parameters[]
	final_wfbp_config := DefaultWFBPConfig()

	// Check for values set in the config file
	v, exists := file_parameters["KernelA"]
	if exists {
		var ok bool
		final_wfbp_config.KernelA, ok = v.(float64) // Cannot guarantee that we won't get a panic here
		if !ok {
			log.Fatal("Expected wfbp-specific parameter KernelA to be a float value of 0.5, 0.54, or 1.0.  Please verify setting in configuration.")
		}
	}

	v, exists = file_parameters["KernelC"]
	if exists {
		var ok bool
		final_wfbp_config.KernelC, ok = v.(float64) // Cannot guarantee that we won't get a panic here
		if !ok {
			log.Fatal("Expected wfbp-specific parameter KernelC to be a float value in the range (0.0, 1.0].  Please verify setting in configuration.")
		}
	}

	// Check for non-zero override values
	if overrides.KernelA != 0.0 {
		final_wfbp_config.KernelA = overrides.KernelA
	}

	if overrides.KernelC != 0.0 {
		final_wfbp_config.KernelC = overrides.KernelC
	}

	config.ReconSpecific = final_wfbp_config

}
