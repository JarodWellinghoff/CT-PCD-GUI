package recon

import (
	"gopkg.in/yaml.v3"
	"testing"
	//"gitlab.com/freect/freect/pkg/recon/wfbp"
)

func TestConfigSerialization(t *testing.T) {

	w := wfbp.WFBPConfig{}

	r := NewReconConfig(w)

	data, err := yaml.Marshal(r)
	if err != nil {
		t.Error(err)
	}
	t.Log(string(data))

}
