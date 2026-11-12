package recon

type PreprocessingConfig struct {
	CorrectMATLABIndexing bool `json:"CorrectMATLABIndexing" yaml:"CorrectMATLABIndexing"`
	FlipDetectorChannels  bool `json:"FlipDetectorChannels" yaml:"FlipDetectorChannels"`
	FlipDetectorRows      bool `json:"FlipDetectorRows" yaml:"FlipDetectorRows"`
	NegateTubeAngle       bool `json:"NegateTubeAngle" yaml:"NegateTubeAngle"`

	AdapativeFiltration float64 `json:"AdapativeFiltration" yaml:"AdapativeFiltration"`
}
