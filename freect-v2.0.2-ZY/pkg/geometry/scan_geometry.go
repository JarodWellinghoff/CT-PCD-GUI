package geometry

// ScanGeometry describes how the scan was acquired and should not be
// modified once read from the raw data.
//
// Scan geometry should be treated as a read-only struct throughout
// FreeCT applications.
type ScanGeometry struct {
	Manufacturer              string  `json:"Manufacturer" yaml:"Manufacturer"`
	DetectorRows              int     `json:"DetectorRows" yaml:"DetectorRows"`
	DetectorChannels          int     `json:"DetectorChannels" yaml:"DetectorChannels"`
	DetectorTransverseSpacing float64 `json:"DetectorTransverseSpacing" yaml:"DetectorTransverseSpacing"`
	DetectorAxialSpacing      float64 `json:"DetectorAxialSpacing" yaml:"DetectorAxialSpacing"`
	DetectorShape             string  `json:"DetectorShape" yaml:"DetectorShape"`

	DistSourceToDetector   float64 `json:"DistSourceToDetector" yaml:"DistSourceToDetector"`
	DistSourceToIsocenter  float64 `json:"DistSourceToIsocenter" yaml:"DistSourceToIsocenter"`
	DetectorCentralRow     float64 `json:"DetectorCentralRow" yaml:"DetectorCentralRow"`
	DetectorCentralChannel float64 `json:"DetectorCentralChannel" yaml:"DetectorCentralChannel"`
	ProjectionGeometry     string  `json:"ProjectionGeometry" yaml:"ProjectionGeometry"`

	ScanType               string `json:"ScanType" yaml:"ScanType"`
	ProjectionsPerRotation int    `json:"ProjectionsPerRotation" yaml:"ProjectionsPerRotation"`
	FlyingFocalSpotMode    string `json:"FlyingFocalSpotMode" yaml:"FlyingFocalSpotMode"`

	StartTablePosition     float64 `json:"StartTablePosition" yaml:"StartTablePosition"`
	EndTablePosition       float64 `json:"EndTablePosition" yaml:"EndTablePosition"`
	TableFeedPerRotation   float64 `json:"TableFeedPerRotation" yaml:"TableFeedPerRotation"`
	AcquisitionFieldOfView float64 `json:"AcquisitionFieldOfView" yaml:"AcquisitionFieldOfView"`
	CollimatedSliceWidth   float64 `json:"CollimatedSliceWidth" yaml:"CollimatedSliceWidth"`

	HUCalibrationFactor float64 `json:"HUCalibrationFactor" yaml:"HUCalibrationFactor"` // This probably shouldn't be in a ScanGeometry struct, but here we are

	NumProjectionsTotal int `json:"NumProjectionsTotal" yaml:"NumProjectionsTotal"`
}
