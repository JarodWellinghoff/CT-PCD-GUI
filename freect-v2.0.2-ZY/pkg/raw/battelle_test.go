package raw

import (
	"fmt"
	"gopkg.in/yaml.v3"
	"strconv"
	"testing"
)

var test_dir = "test_data/battelle/"

func TestBattelleProjection(t *testing.T) {

	ls := NewLocalDirectoryStore(test_dir)
	r, err := NewBattelleReader(ls)
	if err != nil {
		t.Fatal(err)
	}

	p, _, err := r.Read()
	if err != nil {
		t.Error("dicom read failed: ", err)
	}

	t.Logf("read %d projections", len(p))

	test_proj := p[0]
	if test_proj.NumRows != 64 {
		t.Error("expected 64 rows but got ", test_proj.NumRows)
	}
	t.Logf("projection has %d rows", test_proj.NumRows)

	if test_proj.NumChannels != 888 {
		t.Error("expected 888 channels but got ", test_proj.NumRows)
	}
	t.Logf("projection has %d channels", test_proj.NumChannels)

	if test_proj.TubeAngle != 3.8016846 {
		t.Errorf("expected %0.9f tube angle but got %0.9f", 3.8016846, test_proj.TubeAngle)
	}
	t.Logf("projection has %f tube angle", test_proj.TubeAngle)

	if test_proj.TablePosition != 25.279000 {
		t.Errorf("expected %f table position but got %v", 25.279, test_proj.TubeAngle)
	}
	t.Logf("projection has %f table position", test_proj.TablePosition)

	for i, v := range test_proj.Data {

		idx := strconv.Itoa(i)
		val := strconv.FormatFloat(float64(v), 'f', 5, 64)

		fmt.Printf("%05s %10s\n", idx, val)

		if i == 10 {
			break
		}

	}
	//writeFloat64ArrayToDisk(test_proj.Data, "test.bin")
}

func TestBattelleGeometry(t *testing.T) {

	ls := NewLocalDirectoryStore(test_dir)

	r, err := NewBattelleReader(ls)
	if err != nil {
		t.Fatal(err)
	}

	g, err := r.ReadGeometry()
	if err != nil {
		t.Error(err)
	}

	data, err := yaml.Marshal(g)

	t.Log(string(data))
}

//func BenchmarkBattelleRead(b *testing.B) {
//	test_dir := "/Users/jhoffman/Data/ldct/sample_case"
//	for n:=0; n< b.N; n++ {
//		r := NewRawDataReader(FromDicom(test_dir))
//		r.Read()
//	}
//}
