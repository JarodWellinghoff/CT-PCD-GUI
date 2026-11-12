// TODO: Probably should rethink this approach.  Don't love the
// global/singleton thinking here.

package recon

type RuntimeConfig struct {
	Silent         bool
	Verbose        bool
	NoTelemetry    bool
	GPUMemoryLimit float64 // gigabytes
	CacheEnabled   bool
	CacheDirectory string
}

var GlobalRuntimeConfig = RuntimeConfig{
	Silent:         false,
	Verbose:        false,
	NoTelemetry:    false,
	GPUMemoryLimit: 4.0,
	CacheEnabled:   false,
	CacheDirectory: "",
}
