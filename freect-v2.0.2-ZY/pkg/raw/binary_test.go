package raw

import (
	"fmt"
	"gopkg.in/yaml.v3"
	"strconv"
	"testing"
)

func TestBinaryRead(t *testing.T) {

	//d := NewLocalDirectoryStore("/Users/jhoffman/Code/FreeCT/cmd/fct-read-util/tmp/")
	d := NewLocalDirectoryStore("test_data/binary")
	b, err := NewBinaryReader(d)
	if err != nil {
		t.Fatal(err)
	}

	p, g, err := b.Read()
	if err != nil {
		t.Error(err)
	}

	if len(p) != 3 {
		t.Errorf("Expected 3 projections but found %d", len(p))
	}

	test_proj := p[0]
	if test_proj.NumRows != 64 {
		t.Error("expected 64 rows but got ", test_proj.NumRows)
	}
	t.Logf("projection has %d rows", test_proj.NumRows)

	if test_proj.NumChannels != 888 {
		t.Error("expected 888 channels but got ", test_proj.NumRows)
	}
	t.Logf("projection has %d channels", test_proj.NumChannels)

	t.Log("Sample attenuation values read in: ")
	for i, v := range test_proj.Data {
		idx := strconv.Itoa(i)
		val := strconv.FormatFloat(float64(v), 'f', 5, 64)

		t.Log(fmt.Sprintf("%05s %10s", idx, val))

		if i == 10 {
			break
		}
	}

	data, _ := yaml.Marshal(g)
	if g.Manufacturer != "GE" {
		t.Error("expected manufacturer to be GE but got: ", g.Manufacturer)
	}

	t.Log("Geometry read in:\n", string(data))
}
