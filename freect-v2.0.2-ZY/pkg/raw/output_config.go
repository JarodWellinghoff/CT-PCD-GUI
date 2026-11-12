package raw

// Configure where the reconstruction should source its raw data
// Only ONE of these fields can be filled out for a reconstruction
type RawOutputConfig struct {
	Battelle string `json:"Batelle" yaml:"Battelle"`
	Binary   string `json:"Binary" yaml:"Binary"`
	//Ptr      string `json:"Ptr" yaml:"Ptr"`
	//Ctd      string `json:"Ctd" yaml:"Ctd"`
}
