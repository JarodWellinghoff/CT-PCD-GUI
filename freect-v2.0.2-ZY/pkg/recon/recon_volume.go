package recon

// The ReconVolume structure fully specifies the output voxel grid
type ReconVolume struct {
	StartPosition  float64 `json:"StartPosition" yaml:"StartPosition"`
	EndPosition    float64 `json:"EndPosition" yaml:"EndPosition"`
	FieldOfView    float64 `json:"FieldOfView" yaml:"FieldOfView"`
	SliceThickness float64 `json:"SliceThickness" yaml:"SliceThickness"`
	SliceSpacing   float64 `json:"SlicePitch" yaml:"SlicePitch"`
	Nx             int     `json:"Nx" yaml:"Nx"`
	Ny             int     `json:"Ny" yaml:"Ny"`
	XOrigin        float64 `json:"XOrigin" yaml:"XOrigin"`
	YOrigin        float64 `json:"YOrigin" yaml:"YOrigin"`
}
