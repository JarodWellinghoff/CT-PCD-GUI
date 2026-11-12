package recon

import (
	"gitlab.com/freect/freect/pkg/raw"
	"math"
)

func RoundToNearest(x float64, unit float64) float64 {
	return math.Round(x/unit) * unit
}

func GenerateBaseConfig(c *ReconConfig) (*ReconConfig, error) {

	r, err := c.RawInput.GetReader()
	if err != nil {
		return nil, err
	}

	// Extract scan geometry
	g, err := r.ReadGeometry()
	if err != nil {
		return nil, err
	}

	c.ScanGeometry = g

	// Estimate reconstruction parameters
	// Estimate of FOV; does not take into account finite detector width or 1/4 detector offset
	c.ReconVolume.FieldOfView = RoundToNearest(2.0*g.DistSourceToIsocenter*math.Sin(float64(g.DetectorChannels)*g.DetectorTransverseSpacing/(2.0*g.DistSourceToDetector)), 0.001)
	c.ReconVolume.SliceThickness = RoundToNearest(g.DetectorAxialSpacing*g.DistSourceToIsocenter/g.DistSourceToDetector, 0.001)
	c.ReconVolume.SliceSpacing = c.ReconVolume.SliceThickness
	c.ReconVolume.Nx = 512
	c.ReconVolume.Ny = 512
	// Estimate of start/stop locations (crude, does not account for scan pitch)
	scan_direction := (g.EndTablePosition - g.StartTablePosition) / math.Abs(g.EndTablePosition-g.StartTablePosition)
	half_detector_width := g.DetectorAxialSpacing * float64(g.DetectorRows) / 2.0
	c.ReconVolume.StartPosition = RoundToNearest(g.StartTablePosition+scan_direction*half_detector_width, 0.01)
	c.ReconVolume.EndPosition = RoundToNearest(g.EndTablePosition-scan_direction*half_detector_width, 0.01)

	// If reader type is Battelle, then we want to set appropriate
	// preprocessing flags
	if _, ok := r.(*raw.BattelleReader); ok {
		c.Preprocessing.CorrectMATLABIndexing = true
		c.Preprocessing.FlipDetectorChannels = true
		c.Preprocessing.FlipDetectorRows = true
	}

	return c, nil
}
