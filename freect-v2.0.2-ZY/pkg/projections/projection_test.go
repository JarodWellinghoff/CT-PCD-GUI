package projections_test

import (
	"gitlab.com/freect/freect/pkg/projections"
	"reflect"
	"testing"
)

var test_data_dir = "../raw/test_data/battelle/"

func TestReverseRowOrder(t *testing.T) {

	data_org := []float32{1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4}
	data_exp := []float32{4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1}

	tmp := make([]float32, len(data_org))
	copy(tmp, data_org)

	p := &projections.Projection{
		NumRows:     4,
		NumChannels: 3,
		Data:        tmp,
	}

	p.ReverseRowOrder()

	if !reflect.DeepEqual(p.Data, data_exp) {
		t.Error("Expected ", data_exp)
		t.Error("But got: ", p.Data)
	}

	t.Log("Original data: ", data_org)
	t.Log("Flipped rows:  ", p.Data)

}
