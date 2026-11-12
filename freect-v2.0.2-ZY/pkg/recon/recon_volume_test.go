package recon

import (
	"encoding/json"
	"gopkg.in/yaml.v3"
	"testing"
)

func TestReconVolume(t *testing.T) {
	r := &ReconVolume{}

	data, err := yaml.Marshal(r)
	if err != nil {
		t.Error(err)
	}
	t.Log(string(data))

	data, err = json.Marshal(r)
	if err != nil {
		t.Error(err)
	}
	t.Log(string(data))
}
